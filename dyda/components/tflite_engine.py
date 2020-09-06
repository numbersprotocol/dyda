import cv2
import copy
import numpy as np
import tensorflow as tf

from dyda_utils import lab_tools
from dyda.core import tf_detector_base
from dyda.core import classifier_base
from berrynet.engine import DLEngine


class DetectorTFLiteMobileNetSSD(tf_detector_base.TFDetectorBase):

    def __init__(self, dyda_config_path='', testing=False):
        """ __init__ of DetectorTFLiteMobileNetSSD

        This detector component relies on BerryNet library.

        Trainer Variables:
            input_data: a list of image array
            results: defined by lab_tools.output_pred_classification (dict)

        Arguments:
            testing -- True to call set_testing_params. Anything written in
                       dyda.config will be overwritten.
        """

        super(DetectorTFLiteMobileNetSSD, self).__init__(
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
        self.thre = 0.3
        if "threshold" in self.param.keys():
            self.thre = self.param["threshold"]

        self.num_threads = 1
        if "num_threads" in self.param.keys():
            self.num_threads = self.param["num_threads"]

        self.model_file = self.param["model_file"]
        self.label_map = self.param["label_map"]

        self.engine = TFLiteDetectorEngine(
                        self.model_file, self.label_map,
                        threshold=self.thre, num_threads=self.num_threads
                      )

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
        model_file = "/usr/share/dlmodels/mobilenet-ssd-coco-tflite-2.0.0/" +\
                     "model.tflite"
        label_map = "/usr/share/dlmodels/mobilenet-ssd-coco-tflite-2.0.0/" +\
                    "labels.txt"
        self.logger.info("Path of testing file: %s" % model_file)
        self.logger.info("Path of label_map: %s" % label_map)
        self.param["model_file"] = model_file
        self.param["label_map"] = label_map
        self.param["convert_to_rgb"] = True
        testing_img = "/usr/share/dlmodels/dog.jpg"
        self.logger.info("Path of testing file: %s" % testing_img)
        self.input_data = [cv2.imread(testing_img)]

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

        for _img in self.input_data:
            img_array = copy.deepcopy(_img)
            img_array = self.check_bgr_rgb(img_array)
            tensor = self.engine.process_input(img_array)
            inf_results = self.engine.inference(tensor)
            final_results = self.engine.process_output(inf_results)
            self.results.append(final_results)

        if unpack:
            self.unpack_single_results()


class TFLiteDetectorEngine(DLEngine):
    def __init__(self, model, labels, threshold=0.5, num_threads=1):
        """
        Builds Tensorflow graph, load model and labels
        """
        # Load labels
        self.labels = self._load_label(labels)
        self.classes = len(self.labels)

        # Define lite graph and Load Tensorflow Lite model into memory
        self.interpreter = tf.lite.Interpreter(
            model_path=model,
            num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_dtype = self.input_details[0]['dtype']
        self.threshold = threshold

    def __delete__(self, instance):
        # tf.reset_default_graph()
        # self.sess = tf.InteractiveSession()
        del self.interpreter

    def process_input(self, tensor):
        """Resize and normalize image for network input"""

        self.img_w = tensor.shape[1]
        self.img_h = tensor.shape[0]

        tensor = cv2.resize(tensor, (300, 300))
        tensor = np.expand_dims(tensor, axis=0)
        if self.input_dtype == np.float32:
            tensor = (2.0 / 255.0) * tensor - 1.0
            tensor = tensor.astype('float32')
        else:
            # default data type returned by cv2.imread is np.unit8
            pass
        return tensor

    def inference(self, tensor):
        self.interpreter.set_tensor(self.input_details[0]['index'], tensor)
        self.interpreter.invoke()

        # get results
        boxes = self.interpreter.get_tensor(
            self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(
            self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(
            self.output_details[2]['index'])
        num = self.interpreter.get_tensor(
            self.output_details[3]['index'])
        return {
            'boxes': boxes,
            'classes': classes,
            'scores': scores,
            'num': num
        }

    def process_output(self, output):
        # get results
        boxes = np.squeeze(output['boxes'][0])
        classes = np.squeeze(output['classes'][0] + 1).astype(np.int32)
        scores = np.squeeze(output['scores'][0])
        num = output['num'][0]

        annotations = []
        number_boxes = boxes.shape[0]
        for i in range(number_boxes):
            box = tuple(boxes[i].tolist())
            ymin, xmin, ymax, xmax = box

            if scores[i] < self.threshold:
                continue
            lab_dict = lab_tools._lab_annotation_dic()
            lab_dict['type'] = 'detection'
            lab_dict['label'] = self.labels[classes[i]]
            lab_dict['confidence'] = float(scores[i])
            lab_dict['left'] = int(xmin * self.img_w)
            lab_dict['top'] = int(ymin * self.img_h)
            lab_dict['right'] = int(xmax * self.img_w)
            lab_dict['bottom'] = int(ymax * self.img_h)
            annotations.append(lab_dict)
        results = lab_tools.output_pred_detection(
                    "", annotations, anno_in_lab_format=True)
        results["width"] = self.img_w
        results["height"] = self.img_h
        return results

    def _load_label(self, path):
        with open(path, 'r') as f:
            labels = list(map(str.strip, f.readlines()))
        return labels


class ClassifierTFLiteMobileNet(classifier_base.ClassifierBase):

    def __init__(self, dyda_config_path='', testing=False):
        """ __init__ of ClassifierTFLiteMobileNet

        This classifier component relies on BerryNet library.

        Trainer Variables:
            input_data: a list of image array
            results: defined by lab_tools.output_pred_classification (dict)

        Arguments:
            testing -- True to call set_testing_params. Anything written in
                       dyda.config will be overwritten.
        """

        super(ClassifierTFLiteMobileNet, self).__init__(
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

        self.top_k = 3
        if "top_k" in self.param.keys():
            self.top_k = self.param["top_k"]

        self.num_threads = 1
        if "num_threads" in self.param.keys():
            self.num_threads = self.param["num_threads"]

        self.model_file = self.param["model_file"]
        self.label_map = self.param["label_map"]

        self.engine = TFLiteClassifierEngine(
                        self.model_file, self.label_map, top_k=self.top_k,
                        num_threads=self.num_threads
                      )

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
        model_file = "/usr/share/dlmodels/elder-detection-tflite-2.0.0/" +\
                     "output_graph.tflite"
        label_map = "/usr/share/dlmodels/elder-detection-tflite-2.0.0/" +\
                    "output_labels.txt"
        self.logger.info("Path of testing file: %s" % model_file)
        self.logger.info("Path of label_map: %s" % label_map)
        self.param["model_file"] = model_file
        self.param["label_map"] = label_map
        self.param["convert_to_rgb"] = True
        testing_img = "/usr/share/dlmodels/dog.jpg"
        self.logger.info("Path of testing file: %s" % testing_img)
        self.input_data = [cv2.imread(testing_img)]

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

        for _img in self.input_data:
            img_array = copy.deepcopy(_img)
            img_array = self.check_bgr_rgb(img_array)
            tensor = self.engine.process_input(img_array)
            inf_results = self.engine.inference(tensor)
            final_results = self.engine.process_output(inf_results)
            self.results.append(final_results)

        if unpack:
            self.unpack_single_results()


class TFLiteClassifierEngine(DLEngine):
    def __init__(self, model, labels, top_k=3, num_threads=1,
                 input_mean=127.5, input_std=127.5):
        """
        Builds Tensorflow graph, load model and labels
        """
        # Load labels
        self.labels = self._load_label(labels)
        self.classes = len(self.labels)

        # Define lite graph and Load Tensorflow Lite model into memory
        self.interpreter = tf.lite.Interpreter(
            model_path=model,
            num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.floating_model = False
        if self.input_details[0]['dtype'] == np.float32:
            self.floating_model = True
        self.input_mean = input_mean
        self.input_std = input_std
        self.top_k = int(top_k)

    def __delete__(self, instance):
        # tf.reset_default_graph()
        # self.sess = tf.InteractiveSession()
        del self.interpreter

    def process_input(self, tensor):
        """Resize and normalize image for network input"""

        self.img_w = tensor.shape[1]
        self.img_h = tensor.shape[0]

        frame = cv2.resize(tensor, (self.input_details[0]['shape'][2],
                           self.input_details[0]['shape'][1]))
        frame = np.expand_dims(frame, axis=0)
        if self.floating_model:
            frame = (np.float32(frame) - self.input_mean) / self.input_std
        return frame

    def inference(self, tensor):
        self.interpreter.set_tensor(self.input_details[0]['index'], tensor)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(
                        self.output_details[0]['index'])
        results = np.squeeze(output_data)
        return {
            'scores': results,
        }

    def process_output(self, output):
        # get results
        scores = output['scores']
        top_k_results = scores.argsort()[-self.top_k:][::-1]

        annotations = []

        for i in top_k_results:
            human_string = self.labels[i]
            if self.floating_model:
                score = scores[i]
            else:
                score = scores[i]/255.0
            anno = {
                'type': 'classification',
                'label': human_string,
                'confidence': score,
                'top': 0,
                'left': 0,
                'bottom': self.img_h,
                'right': self.img_w
            }
            annotations.append(anno)

        results = lab_tools.output_pred_detection(
                    "", annotations, anno_in_lab_format=True)
        results["width"] = self.img_w
        results["height"] = self.img_h

        return results

    def _load_label(self, path):
        with open(path, 'r') as f:
            labels = list(map(str.strip, f.readlines()))
        return labels
