"""TRT Classifier Component
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image

from logzero import logger
from dyda_utils import lab_tools
from dyda.core import classifier_base

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class ClassifierTensorRT(classifier_base.ClassifierBase):
    def __init__(self, dyda_config_path='', debug=False):
        """ __init__ of ClassifierTensorRT

        Trainer Variables:
            input_data: a list of image array
            results: defined by lab_tools.output_pred_classification (dict)

        Arguments:
            dyda_config_path -- Trainer config filepath
        """

        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Setup dyda config
        super(ClassifierTensorRT, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

        self.check_param_keys()

        # Create engine
        self.engine = self._load_engine(self.param['model_file'])

        # Get Dim:
        self.c, self.h, self.w = self._get_input_shape()

        # Create execution context
        self.context = self.engine.create_execution_context()
        (self.inputs,
         self.outputs,
         self.bindings,
         self.stream) = self._allocate_buffers(self.engine)

        # Setup DL model
        with open(self.param['label_file'], 'r') as f:
            self.labels_map = f.read().split('\n')

    def check_param_keys(self):
        """
        Check if any default key is missing in the self.param.
        """
        default_keys = ["model_file",
                        "model_description",
                        "label_file",
                        ]
        for _key in default_keys:
            if _key not in self.param.keys():
                logger.info("%s missing in self.param" % _key)
                self.terminate_flag = True
            else:
                continue
        logger.info("keys of self.param are checked")

    def process_input(self, tensor):
        """Resize tensor (if needed) and change layout from HWC to CHW.

        Args:
            tensor: Input BGR tensor (OpenCV convention)

        Returns:
            Resized and transposed tensor
        """
        if tensor.shape[:-1] != (self.h, self.w):
            logger.warning("Input tensor is resized from {} to {}".format(
                tensor.shape[:-1], (self.h, self.w)))
            tensor = cv2.resize(tensor, (self.w, self.h))
        # Change data layout from HWC to CHW
        tensor = tensor.transpose((2, 0, 1))
        tensor = (2.0 / 255.0) * tensor - 1.0
        return tensor.ravel()

    def inference(self, tensor):
        logger.debug("Starting inference")
        np.copyto(self.inputs[0].host, tensor)
        res = self._do_inference(self.context,
                                 self.bindings,
                                 self.inputs,
                                 self.outputs,
                                 self.stream)
        return res

    def process_output(self, output):
        """
        Args:
            output: result fo inference engine

        Retruns:
            top-1 inference result in the format:

                {
                    'label': str,
                    'confidence': float
                }
        """
        top_k = 1

        logger.debug("Processing output blob")
        logger.debug("Top {} results: ".format(top_k))

        annotations = []
        for probs in output:
            probs = self._softmax(np.squeeze(probs))
            top_ind = np.argsort(probs)[-top_k:][::-1]
            for lid in top_ind:
                if self.labels_map:
                    det_label = self.labels_map[lid]
                else:
                    det_label = '#{}'.format(lid)
                logger.debug("\t{:.7f} label {}".format(probs[lid], det_label))

                annotations.append({
                    'label': det_label,
                    'confidence': float(probs[lid])
                })
        logger.debug('process_output return: {}'.format(annotations[0]))
        return annotations[0]

    def main_process(self):
        if len(self.input_data) == 0:
            logger.debug('no input_data found')
            self.terminate_flag = True

        logger.debug('self.input_data len: {}'.format(len(self.input_data)))
        for img_array in self.input_data:
            orig_h, orig_w = img_array.shape[:-1]
            img_array = self.process_input(img_array)
            inf_results = self.inference(img_array)
            top_result = self.process_output(inf_results)

            # orders, labinfo = self.create_labinfo(inf_results)
            res = lab_tools.output_pred_classification(
                # input_path=self.orig_input_path,
                input_path='',
                conf=top_result['confidence'],
                label=top_result['label'],
                img_size=(orig_h, orig_w),
                # labinfo=labinfo
                labinfo={}
            )
            self.results.append(res)
        logger.debug('self.results: {}'.format(self.results))

    def _get_input_idx(self):
        for i in range(1000):
            if self.engine.binding_is_input(i):
                return i
        return -1

    def _get_input_shape(self):
        """Dim in (C, H, W)
        """
        return self.engine.get_binding_shape(self._get_input_idx())

    def _load_engine(self, engine_file):
        with open(engine_file, 'rb') as eng_fh, \
             trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(eng_fh.read())

    def _allocate_buffers(self, engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(
                engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def _do_inference(self,
                      context,
                      bindings,
                      inputs,
                      outputs,
                      stream,
                      batch_size=1):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream)
            for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=batch_size,
                              bindings=bindings,
                              stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream)
            for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def _softmax(self, probs):
        return np.exp(probs) / np.sum(np.exp(probs))
