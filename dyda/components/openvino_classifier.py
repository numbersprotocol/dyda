"""OpenVINO Classifier Component

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

from berrynet import logger
from dt42lab.core import lab_tools
from openvino.inference_engine import IENetwork, IEPlugin
from dyda.core import classifier_base


class ClassifierOpenVINO(classifier_base.ClassifierBase):
    """ Modified from label_image.py example in TensorFlow """

    def __init__(self, dyda_config_path='', debug=False):
        """ __init__ of ClassifierOpenVINO

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
        super(ClassifierOpenVINO, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

        self.check_param_keys()

        # Setup DL model
        model_xml = self.param['model_description']
        model_bin = self.param['model_file']
        with open(self.param['label_file'], 'r') as f:
            # Allow label name with spaces. To use onlyh the 1st word,
            # uncomment another labels_map implementation below.
            self.labels_map = [l.strip() for l in f.readlines()]
            # self.labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip()
            #                   for x in f]

        # Setup OpenVINO
        #
        # Plugin initialization for specified device and
        # load extensions library if specified
        #
        # Note: MKLDNN CPU-targeted custom layer support is not included
        #       because we do not use it yet.
        self.plugin = IEPlugin(device=self.param['device'], plugin_dirs=None)
        logger.debug("Computation device: {}".format(self.param['device']))

        # Read IR
        logger.debug("Loading network files:\n\t{}\n\t{}".format(
            model_xml, model_bin))
        net = IENetwork.from_ir(model=model_xml, weights=model_bin)

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
                sys.exit(1)

        assert len(net.inputs.keys()) == 1, (
            'Sample supports only single input topologies')
        assert len(net.outputs) == 1, (
            'Sample supports only single output topologies')

        # input_blob and and out_blob are the layer names in string format.
        logger.debug("Preparing input blobs")
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        net.batch_size = 1

        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape

        # Loading model to the plugin
        logger.debug("Loading model to the plugin")
        self.exec_net = self.plugin.load(network=net)

        del net

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
        return tensor.transpose((2, 0, 1))

    def inference(self, tensor):
        logger.debug("Starting inference")
        res = self.exec_net.infer(inputs={self.input_blob: tensor})
        return res[self.out_blob]

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
            probs = np.squeeze(probs)
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
            logger.error('no input_data found')
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
