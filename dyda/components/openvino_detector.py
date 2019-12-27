"""OpenVINO Detector Component

Note: OpenVINO environment has to be set before using this component.

    To setup OpenVINO environment, the script below has to be run:

        $ source /opt/intel/computer_vision_sdk_2018.4.420/bin/setupvars.sh
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys

import cv2
import numpy as np

from time import time
from berrynet import logger
from dyda_utils import lab_tools
from openvino.inference_engine import IENetwork, IEPlugin
from dyda.core import detector_base


class DetectorOpenVINO(detector_base.DetectorBase):

    def __init__(self, dyda_config_path='', debug=False):
        """ __init__ of DetectorOpenVINO

        Trainer Variables:
            input_data: a list of image array
            results: defined by lab_tools.output_pred_detection()

        Arguments:
            dyda_config_path -- Trainer config filepath
        """
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Setup dyda config
        super(DetectorOpenVINO, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

        self.check_param_keys()

        if "threshold" in self.param.keys():
            self.threshold = self.param["threshold"]
        else:
            self.threshold = 0.3

        # Setup DL model
        model_xml = self.param['model_description']
        model_bin = self.param['model_file']
        with open(self.param['label_file'], 'r') as f:
            self.labels_map = [x.strip() for x in f]

        # Setup OpenVINO
        #
        # Plugin initialization for specified device and
        # load extensions library if specified
        #
        # Note: MKLDNN CPU-targeted custom layer support is not included
        #       because we do not use it yet.
        self.plugin = IEPlugin(
            device=self.param['device'], plugin_dirs=self.param['plugin_dirs'])
        if self.param['device'] == 'CPU':
            for ext in self.param['cpu_extensions']:
                logger.info('Add cpu extension: {}'.format(ext))
                self.plugin.add_cpu_extension(ext)
        logger.debug("Computation device: {}".format(self.param['device']))

        # Read IR
        logger.debug("Loading network files:\n\t{}\n\t{}".format(
            model_xml, model_bin))
        net = IENetwork(model=model_xml, weights=model_bin)

        if self.plugin.device == "CPU":
            supported_layers = self.plugin.get_supported_layers(net)
            not_supported_layers = [
                l for l in net.layers.keys()
                if l not in supported_layers
            ]
            if len(not_supported_layers) != 0:
                logger.error((
                    'Following layers are not supported '
                    'by the plugin for specified device {}:\n {}').format(
                        self.plugin.device,
                        ', '.join(not_supported_layers)))
                logger.error("Please try to specify cpu "
                             "extensions library path in demo's "
                             "command line parameters using -l "
                             "or --cpu_extension command line argument")
                sys.exit(1)

        assert len(net.inputs.keys()) == 1, (
            'Demo supports only single input topologies')
        assert len(net.outputs) == 1, (
            'Demo supports only single output topologies')

        # input_blob and and out_blob are the layer names in string format.
        logger.debug("Preparing input blobs")
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))

        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape

        # Loading model to the plugin
        self.exec_net = self.plugin.load(network=net, num_requests=2)

        del net

        # Initialize engine mode: sync or async
        #
        # FIXME: async mode does not work currently.
        #        process_input needs to provide two input tensors for async.
        self.is_async_mode = False
        self.cur_request_id = 0
        self.next_request_id = 1

    def __delete__(self, instance):
        del self.exec_net
        del self.plugin

    # def create_labinfo(self, results):
    #     """Create DT42 labinfo based on def in spec"""
    #     orders = results.argsort()[::-1]
    #     labinfo = {'classifier': {}}
    #     for i in range(0, len(orders)):
    #         index = orders[i]
    #         labinfo['classifier'][self.labels[index]] = results[index]
    #     return orders, labinfo

    def check_param_keys(self):
        """
        Check if any default key is missing in the self.param.
        """
        default_keys = ["model_file",
                        "model_description",
                        "label_file",
                        "device"]
        for _key in default_keys:
            if _key not in self.param.keys():
                logger.error("%s missing in self.param" % _key)
                self.terminate_flag = True
            else:
                continue
        logger.debug("keys of self.param are checked")

    def process_input(self, tensor, next_tensor=None):
        frame = tensor
        next_frame = next_tensor

        # original input shape will be used in process_output
        self.img_w = tensor.shape[1]
        self.img_h = tensor.shape[0]

        # Main sync point:
        # in the truly Async mode we start the NEXT infer request,
        # while waiting for the CURRENT to complete
        # in the regular mode we start the CURRENT request and immediately
        # wait for it's completion
        if self.is_async_mode:
            in_frame = cv2.resize(next_frame, (self.w, self.h))
            # Change data layout from HWC to CHW
            in_frame = in_frame.transpose((2, 0, 1))
            in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))
        else:
            in_frame = cv2.resize(frame, (self.w, self.h))
            # Change data layout from HWC to CHW
            in_frame = in_frame.transpose((2, 0, 1))
            in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))
        return in_frame

    def inference(self, tensor):
        inf_start = time()
        if self.is_async_mode:
            self.exec_net.start_async(request_id=self.next_request_id,
                                      inputs={self.input_blob: tensor})
        else:
            self.exec_net.start_async(request_id=self.cur_request_id,
                                      inputs={self.input_blob: tensor})

        if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
            inf_end = time()
            det_time = inf_end - inf_start
            if self.is_async_mode:
                logger.debug(r'Inference time: N\A for async mode')
            else:
                logger.debug("Inference time: {:.3f} ms".format(
                    det_time * 1000))

            # Parse detection results of the current request
            res = self.exec_net.requests[
                self.cur_request_id].outputs[self.out_blob]
        else:
            res = None

        return res

        # FIXME: async mode does not work currently.
        #        process_input needs to provide two input tensors for async.
        if self.is_async_mode:
            self.cur_request_id, self.next_request_id = \
                self.next_request_id, self.cur_request_id
            frame = next_frame

    def process_output(self, output):

        logger.debug("Processing output blob")
        logger.debug("Threshold: {} ".format(self.threshold))

        annotations = []

        for obj in output[0][0]:
            # Collect objects when probability more than specified threshold
            if obj[2] > self.threshold:
                xmin = int(obj[3] * self.img_w)
                ymin = int(obj[4] * self.img_h)
                xmax = int(obj[5] * self.img_w)
                ymax = int(obj[6] * self.img_h)
                class_id = int(obj[1])
                if self.labels_map:
                    det_label = self.labels_map[class_id]
                else:
                    str(class_id)
                annotations.append({
                    'label': det_label,
                    'confidence': float(obj[2]),
                    'left': xmin,
                    'top': ymin,
                    'right': xmax,
                    'bottom': ymax
                })
        logger.debug('process_output return: {}'.format(annotations))
        return annotations

    def main_process(self):
        if len(self.input_data) == 0:
            logger.error('no input_data found')
            self.terminate_flag = True

        logger.debug('self.input_data len: {}'.format(len(self.input_data)))
        for img_array in self.input_data:
            orig_h, orig_w = img_array.shape[:-1]
            img_array = self.process_input(img_array)
            inf_results = self.inference(img_array)
            det_results = self.process_output(inf_results)

            # why this code looks so redundant here is beacause that
            # this script is modified from berrynet openvino_engine.py,
            # and I follow the principle that make least change of original
            # code. the annotations structure in process_output is original
            # defined in berrynet, and in order to use the lab_tools, we
            # must re-define the structure
            annotations = [[det_result['label'],
                            det_result['confidence'],
                            [det_result['top'],
                             det_result['bottom'],
                             det_result['left'],
                             det_result['right']
                             ]
                            ] for det_result in det_results]

            # orders, labinfo = self.create_labinfo(inf_results)
            res = lab_tools.output_pred_detection(
                # input_path=self.orig_input_path,
                input_path='',
                annotations=annotations,
                img_size=(orig_h, orig_w),
                # labinfo=labinfo
                labinfo={}
            )
            self.results.append(res)
        logger.debug('self.results: {}'.format(self.results))
