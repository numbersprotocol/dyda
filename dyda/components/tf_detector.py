import os
import cv2
import copy
import numpy as np
import tensorflow as tf
from dyda_utils import lab_tools
from dyda.core import tf_detector_base


def load_graph(model_file_path):
    """ Load TensorFlow graph from a model file """

    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_file_path, 'rb') as model_file:
            graph_def.ParseFromString(model_file.read())
            tf.import_graph_def(graph_def, name='')

    return graph


def load_labels(label_map_path, npack):
    """ Load model labels from a label file """

    label_map = lab_tools.tf_label_map_to_dict(
        label_map_path, nline_in_pack=npack
    )
    return label_map


class DetectorMobileNetSSD(tf_detector_base.TFDetectorBase):

    def __init__(self, dyda_config_path='', testing=False):
        """ __init__ of DetectorMobileNetSSD

        Working with 4.0-beta lab docker or newer version
        Working with TensorFlow 1.12.0
        Trainer Variables:
            input_data: a list of image array
            results: defined by lab_tools.output_pred_classification (dict)

        Arguments:
            testing -- True to call set_testing_params. Anything written in
                       dyda.config will be overwritten.
        """

        super(DetectorMobileNetSSD, self).__init__(
            dyda_config_path=dyda_config_path
        )

        # check parameters and set default values
        if testing:
            self.set_testing_params()
        else:
            self.set_param(self.class_name)

        self.check_param_keys()
        self.convert_to_rgb = True
        if "convert_to_rgb" in self.param.keys():
            self.convert_to_rgb = self.param["convert_to_rgb"]
        self.thre = 0.0
        if "threshold" in self.param.keys():
            self.thre = self.param["threshold"]
        # Set numbers of lines in one pack of label_map. Default is 5
        self.npack = 5
        if "label_map_npack" in self.param.keys():
            self.npack = int(self.param["label_map_npack"])
        self.label_key = "display_name"
        if "label_map_key" in self.param.keys():
            self.label_key = self.param["label_map_key"]

        param = self.param

        # load graph
        try:
            self.graph = load_graph(param["model_file"])

        except Exception as error:
            self.logger.error("loading graph located in %s fails"
                              % param["model_file"])
            print(error)
            raise

        # set tf config
        self.config = tf.ConfigProto()
        if "gpu_options" in self.param.keys():
            gpu_options = self.config.gpu_options
            for config_key, value in self.param["gpu_options"].items():
                setattr(gpu_options, config_key, value)

        self.labels = load_labels(param["label_map"], self.npack)
        # FIXME: the input_data is a list, but the path is a fixed string
        self.orig_input_path = ""

        try:
            self.sess = tf.Session(
                graph=self.graph, config=self.config
            )
        except Exception as error:
            self.logger.error("fail to init TF session")
            print(error)
            raise

    def check_param_keys(self):
        """
        Check if any default key is missing in the self.param.
        """

        default_keys = ["model_file", "label_map"]
        for _key in default_keys:
            if _key not in self.param.keys():
                self.logger.error("%s missing in self.param" % _key)
                self.terminate_flag = True
            else:
                continue
        self.logger.info("keys of self.param are checked")

    def set_testing_params(self):
        """
        By calling this function, the default self.param will be defined.
        It should only be used in the unit or integration test.
        """

        self.logger.warning("self.param is overwritten!")
        model_file = "/home/shared/model_zoo/tf_detection_model_resnet/" +\
                     "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_" +\
                     "coco14_sync_2018_07_03/frozen_inference_graph.pb"
        label_map = "/home/shared/model_zoo/tf_detection_model_resnet/" +\
                    "mscoco_label_map.pbtxt"
        self.param["model_file"] = model_file
        self.param["label_map"] = label_map
        self.param["convert_to_rgb"] = True
        parent_dir = "/home/shared/DT42/test_data/test_detector"
        self.input_data = [cv2.imread(os.path.join(parent_dir, "dog.jpg"))]

    def check_bgr_rgb(self, np_image):
        """
        If convert_to_rgb is True, convert image to RGB from BGR
        """
        inf_array = np_image
        if self.convert_to_rgb:
            inf_array = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        return inf_array

    def main_process(self):
        """
        Main process
        """

        if len(self.input_data) == 0:
            print('[tf_detector] ERROR: no input_data found')
            self.terminate_flag = True

        unpack = False
        if type(self.input_data) is not list:
            self.pack_input_as_list()
            unpack = True

        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        det_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        det_scores = self.graph.get_tensor_by_name('detection_scores:0')
        det_classes = self.graph.get_tensor_by_name('detection_classes:0')
        num_dets = self.graph.get_tensor_by_name('num_detections:0')

        for _img_array in self.input_data:
            img_array = self.check_bgr_rgb(_img_array)
            rows = img_array.shape[0]
            cols = img_array.shape[1]
            img_array = np.expand_dims(img_array, axis=0)

            (boxes, scores, classes, num) = self.sess.run(
                [det_boxes, det_scores, det_classes, num_dets],
                feed_dict={image_tensor: img_array})

            # results are packaged in the list, unpack it
            boxes = boxes[0]
            scores = scores[0]
            classes = classes[0]

            annos = []
            for i, score in enumerate(scores):
                if score <= 0:
                    continue
                if score < self.thre:
                    continue
                bb = [int(boxes[i][0]*rows), int(boxes[i][2]*rows),
                      int(boxes[i][1]*cols), int(boxes[i][3]*cols)]
                annos.append([self.labels[int(classes[i])][self.label_key],
                              float(score), bb])

            # output_pred_detection(input_path, annotations, img_size=[],
            #                       labinfo={}, save_json=False):
            # annotations: A list of annotations [[label, conf, bb]]
            #              where bb is [top, bottom, left, right]
            res = lab_tools.output_pred_detection(
                self.orig_input_path, annos, img_size=(cols, rows))
            self.results.append(res)

        if unpack:
            self.unpack_single_input()
            self.unpack_single_results()

    def close_tf_session(self):
        """ External functions can call this to close self.sess """

        self.sess.close()
