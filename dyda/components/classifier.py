from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import copy
import pickle
import numpy as np
import tensorflow as tf
from sklearn import svm
from skimage import measure
from dt42lab.core import image
from dt42lab.core import tools
from dt42lab.core import lab_tools
from dt42lab.core import tinycv
from dyda.core import classifier_base


def load_graph(model_file_path):
    """ Load TensorFlow graph from a model file """

    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file_path, "rb") as model_file:
        graph_def.ParseFromString(model_file.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(input_height=192, input_width=192,
                                input_mean=0, input_std=255, ftype="jpg"):
    """ Define tensor based on the image type """

    input_name = "file_reader"
    # output_name = "normalized"

    file_name = tf.placeholder("string", name="fname")

    file_reader = tf.read_file(file_name, input_name)
    if ftype == "png":
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                           name='png_reader')
    elif ftype == "gif":
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif ftype == "bmp":
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name='jpeg_reader')

    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander,
                                       [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

    return normalized


def read_tensor_from_nparray(input_height=192, input_width=192,
                             input_mean=0, input_std=255):
    """ Create normalized tensor based on input numpy array """
    image_reader = tf.placeholder(tf.uint8, name='inarray')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander,
                                       [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    return normalized


def load_labels(label_file):
    """ Load model labels from a label file """

    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for line in proto_as_ascii_lines:
        label.append(line.rstrip())
    return label


class ClassifierInceptionv3(classifier_base.ClassifierBase):
    """ Modified from label_image.py example in TensorFlow """

    def __init__(self, dyda_config_path='', testing=False):
        """ __init__ of ClassifierInceptionv3

        Trainer Variables:
            input_data: a list of image array
            results: defined by lab_tools.output_pred_classification (dict)

        Arguments:
            testing -- True to call set_testing_params. Anything written in
                       dyda.config will be overwritten.
        """

        super(ClassifierInceptionv3, self).__init__(
            dyda_config_path=dyda_config_path
        )
        if testing:
            self.set_testing_params()
        else:
            self.set_param(self.class_name)

        self.check_param_keys()
        param = self.param
        try:
            self.graph = load_graph(param["model_file"])

        except Exception as error:
            print("[classifier] ERROR: loading graph located in %s fails"
                  % param["model_file"])
            print(error)
            raise

        self.config = tf.ConfigProto()

        if "gpu_options" in self.param.keys():
            gpu_options = self.config.gpu_options
            for config_key, value in self.param["gpu_options"].items():
                setattr(gpu_options, config_key, value)

        self.labels = load_labels(param["label_file"])
        # FIXME: the input_data is a list, but the path is a fixed string
        self.orig_input_path = ""

        try:
            # diff of Session and InteractiveSession see https://goo.gl/wVBGpH
            self.sess = tf.InteractiveSession(
                graph=self.graph, config=self.config
            )
            self.tensor_op = read_tensor_from_nparray(
                input_height=param["input_height"],
                input_width=param["input_width"],
                input_mean=param["input_mean"],
                input_std=param["input_std"]
            )
        except Exception as error:
            print("[classifier] ERROR: fail to init TF session and tensor_op")
            print(error)
            raise

    def create_labinfo(self, results):
        """ Create DT42 labinfo based on def in spec """
        orders = results.argsort()[::-1]
        labinfo = {'classifier': {}}
        for i in range(0, len(orders)):
            index = orders[i]
            labinfo['classifier'][self.labels[index]] = results[index]
        return orders, labinfo

    def check_param_keys(self):
        """
        Check if any default key is missing in the self.param.
        """

        default_keys = ["model_file", "label_file", "input_layer",
                        "output_layer", "input_height", "input_width",
                        "input_mean", "input_std", "ftype"]
        for _key in default_keys:
            if _key not in self.param.keys():
                print("[classifier] ERROR: %s missing in self.param" % _key)
                self.terminate_flag = True
            else:
                continue
        print("[classifier] INFO: keys of self.param are checked")

    def set_testing_params(self):
        """
        By calling this function, the default self.param will be defined.
        It should only be used in the unit or integration test.
        """

        print("[classifier] WARNING: self.param is overwritten!")
        model_file = "/home/shared/model_zoo/inception-v3/" +\
                     "dyda_test_model/output_graph.pb"
        label_file = "/home/shared/model_zoo/inception-v3/" +\
                     "dyda_test_model/output_labels.txt"
        self.param = {
            "model_file": model_file, "label_file": label_file,
            "ftype": "jpg"
        }
        self.set_classifier_default_param()
        parent_folder = "/home/shared/lab/dt42-dyda/classifier/"
        self.input_data = cv2.cvtColor(
            cv2.imread(os.path.join(parent_folder, "ayoung.jpg")),
            cv2.COLOR_BGR2RGB)

    def set_classifier_default_param(self):
        """
        This function provide the default parameters used by the
        Inception v3 model. It overwrites the values in self.param.
        """

        if not isinstance(self.param, dict):
            print("[classifier] ERROR: self.param is not a dictionary object")
            self.terminate_flag = True

        print("[classifier] WARNING: Reset inception v3 parameters "
              "to the default values in self.param!")
        self.param["input_height"] = 299
        self.param["input_width"] = 299
        self.param["input_mean"] = 0
        self.param["input_std"] = 255
        self.param["input_layer"] = "Mul"
        self.param["output_layer"] = "final_result"

    def main_process(self):
        """
        """

        input_name = "import/" + self.param["input_layer"]
        output_name = "import/" + self.param["output_layer"]

        if len(self.input_data) == 0:
            print('[classifier] ERROR: no input_data found')
            self.terminate_flag = True

        for img_array in self.input_data:
            if self.param["convert_to_rgb"]:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            tensor = self.sess.run(self.tensor_op,
                                   feed_dict={'inarray:0': img_array})

            input_operation = self.graph.get_operation_by_name(input_name)
            output_operation = self.graph.get_operation_by_name(output_name)

            inf_results = self.sess.run(output_operation.outputs[0],
                                        {input_operation.outputs[0]: tensor})

            inf_results = np.squeeze(inf_results)
            orders, labinfo = self.create_labinfo(inf_results)
            res = lab_tools.output_pred_classification(
                input_path=self.orig_input_path,
                conf=inf_results[orders[0]],
                label=self.labels[orders[0]],
                img_size=(img_array.shape[1], img_array.shape[0]),
                labinfo=labinfo
            )

            self.results.append(res)

    def close_tf_session(self):
        """ External functions can call this to close self.sess """

        self.sess.close()


class ClassifierMobileNet(ClassifierInceptionv3):
    """ Modified from label_image.py example in TensorFlow """

    def __init__(self, dyda_config_path='', testing=False):
        """ __init__ of ClassifierMobileNet

        Trainer Variables:
            input_data: a list of image array
            results: defined by lab_tools.output_pred_classification (dict)

        Arguments:
            testing -- True to call set_testing_params. Anything written in
                       dyda.config will be overwritten.
        """

        super(ClassifierMobileNet, self).__init__(
            dyda_config_path=dyda_config_path, testing=testing
        )
        self.logger.info("Initialization of ClassifierMobileNet done.")


class ClassifierAoiCV(classifier_base.ClassifierBase):
    """ Use CV to extract feature for AOI classification.

    """

    def __init__(self, dyda_config_path="", param=None):
        """Initialization function of dyda component.
           This is called when an object is created from the class and
           it allows the class to initialize the attributes of a class.
        """

        super(ClassifierAoiCV, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.check_param_keys([
            "model_path",
            "resize_width",
            "resize_height",
            "kernel_ratio",
            "iter_num",
            "fragment_num"])

        # load svm model
        model_file = open(self.param['model_path'], 'rb')
        self.clf = pickle.load(model_file)

    def main_process(self):
        """ Main function of dyda component. """

        self.results = []
        self.output_data = []

        if len(self.input_data) == 0:
            print('[classifier] ERROR: no input_data found')
            self.terminate_flag = True

        _input_data = copy.deepcopy(self.input_data)
        for img_array in _input_data:

            # label initialization
            result = 'ok'

            # resize
            img_box = image.resize_img(
                img_array, (self.param['resize_width'],
                            self.param['resize_height']))

            # rgb to gray
            im_g = cv2.cvtColor(img_box, cv2.COLOR_RGB2GRAY)

            # image binarization global
            thre = int(np.mean(im_g.mean(axis=1)))
            ret, im_b = cv2.threshold(im_g, thre, 1, cv2.THRESH_BINARY)

            # morphological opening
            kernel_size = int(np.round(self.param['resize_width'] *
                                       self.param['kernel_ratio']))
            iter_num = self.param['iter_num']
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            im_o = cv2.erode(im_b.astype(np.uint8), kernel, iter_num)
            im_o = cv2.dilate(im_o, kernel, iter_num)

            # connected components labeling
            cc_label = measure.label(im_o, background=0)
            cc_label = cc_label.astype(np.uint8)

            # fragment detection
            label_idx = self.param['fragment_num']
            label = self.extract_label(label_idx, cc_label)
            pixel_num = sum(sum(label))
            score = [[], []]
            if pixel_num > 0:
                result = 'ng'
                score[1].append(1)

            # projection feature
            for i in range(self.param['fragment_num']):
                feature = self.projection_feature(i, cc_label)
                class_idx = self.clf.predict([feature])[0]
                if class_idx == 1:
                    result = 'ng'
                score[class_idx].append(
                    self.clf.predict_proba([feature])[0][class_idx])

            # generate final results
            if result == 'ng':
                final_score = max(score[1])
            else:
                final_score = min(score[0])
            res = lab_tools.output_pred_classification(
                input_path="",
                conf=final_score,
                label=result,
                img_size=(img_array.shape[1], img_array.shape[0]),
                labinfo=[]
            )

            width = res["size"]["width"]
            height = res["size"]["height"]
            # Default space is 40 in tinycv, allow it to be adjusted based
            # on the size of the images
            space = max([40, int(width / 20), int(height / 20)])

            output_data = tinycv.patch_bb_dyda(
                img_array, res, color=[0, 128, 0], space=space
            )
            self.results.append(res)
            self.output_data.append(output_data)

    def extract_label(self, label_idx, cc_label):
        """Extract binary label.
        """

        height = self.param['resize_height']
        width = self.param['resize_width']
        label = np.zeros((height, width), np.bool_)
        label = np.where(cc_label == label_idx, 1, 0)
        return label

    def projection_feature(self, label_idx, cc_label):
        """Extract feature from horizontal projection and vertical projection.
        """
        width = self.param['resize_width']
        label = self.extract_label(label_idx, cc_label)
        column_sum = label.sum(0)
        column_idx = np.where(column_sum > 0)
        if column_idx[0][0] > width / 2:
            column_sum = reversed(column_sum)
        column_sum = list(column_sum)
        row_sum = list(label.sum(1))
        feature = row_sum + column_sum[:int(width / 4)]
        return(feature)


class ClassifierFrameDiff(classifier_base.ClassifierBase):
    """ Use image diff to determine the results """

    def __init__(self, dyda_config_path="", param=None):
        """Initialization function of dyda component.
           This is called when an object is created from the class and
           it allows the class to initialize the attributes of a class.
        """

        super(ClassifierFrameDiff, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.check_param_keys([
            "bkg_path",
            "thre"])
        self.bkg_path = self.param["bkg_path"]
        self.thre = self.param["thre"]

        self.sel_min = 0
        if "sel_min" in self.param.keys():
            self.sel_min = self.param["sel_min"]
        self.sel_max = -1
        if "sel_max" in self.param.keys():
            self.sel_max = self.param["sel_max"]
        self.mean_axis = 1
        if "mean_axis" in self.param.keys():
            self.mean_axis = self.param["mean_axis"]

        try:
            self.bkg_img = image.read_img(self.bkg_path)
        except:
            self.logger.error(
                "Fail to read background image %s" % self.bkg_path
            )
            sys.exit(0)

    def main_process(self):
        """ Main function of dyda component. """

        self.results = []
        self.output_data = []

        if len(self.input_data) == 0:
            self.logger.error("no input_data found")
            self.terminate_flag = True

        _input_data = copy.deepcopy(self.input_data)
        for img_array in _input_data:
            label = 'ok'
            if image.is_rgb(img_array):
                (h, w, ch) = img_array.shape
            else:
                (h, w) = img_array.shape
            img_array = image.resize_img(
                img_array, size=(self.bkg_img.shape[1], self.bkg_img.shape[0])
            )
            l1_diff = tinycv.l1_norm_diff_cv2(img_array, self.bkg_img)
            l1_diff_mean = np.mean(l1_diff, axis=self.mean_axis)
            max_diff = np.max(l1_diff_mean[self.sel_min:self.sel_max])

            label = 'ng'
            conf = 1.0
            if max_diff <= self.thre:
                label = 'ok'

            res = lab_tools.output_pred_classification(
                input_path="",
                conf=conf,
                label=label,
                img_size=(w, h),
                labinfo=[]
            )

            # this classifier only select ok samples
            if label == 'ng':
                res['annotations'] = []
            self.results.append(res)


class ClassifierAoiCornerAvg(classifier_base.ClassifierBase):
    """ Simple average to select corner defect samples.
        This should be run after ClassifierAoiCV

    """

    def __init__(self, dyda_config_path="", param=None):
        """Initialization function of dyda component.
           This is called when an object is created from the class and
           it allows the class to initialize the attributes of a class.
        """

        super(ClassifierAoiCornerAvg, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.check_param_keys([
            "gray_avg_thre",
            "corner_size_h",
            "corner_size_w"])

        self.dw = self.param["corner_size_w"]
        self.dh = self.param["corner_size_h"]
        self.thre = self.param["gray_avg_thre"]
        self.space_w = 0
        self.space_h = 0
        if "space_w" in self.param.keys():
            self.space_w = self.param["space_w"]
        if "space_h" in self.param.keys():
            self.space_h = self.param["space_h"]
        # lt will select event larger than the threshold
        # st will select event small than the threshold
        self.thre_type = "lt"
        if "thre_type" in self.param.keys():
            self.thre_type = self.param["thre_type"]

    def main_process(self):
        """ Main function of dyda component. """

        self.results = []
        self.output_data = []

        if len(self.input_data) == 0:
            self.logger.error("no input_data found")
            self.terminate_flag = True

        _input_data = copy.deepcopy(self.input_data)
        for img_array in _input_data:
            label = 'ok'
            if image.is_rgb(img_array):
                (h, w, ch) = img_array.shape
                gray = image.conv_gray(img_array)
            else:
                (h, w) = img_array.shape
                gray = img_array
            corners_avg = [
                np.average(gray[self.space_h:self.space_h+self.dh,
                                self.space_w:self.space_w+self.dw]),
                np.average(gray[self.space_h:self.space_h+self.dh,
                                w-(self.space_w+self.dw):w-self.space_w]),
                np.average(gray[h-(self.space_h+self.dh):h-self.space_h,
                                self.space_w:self.space_w+self.dw]),
                np.average(gray[h-(self.space_h+self.dh):h-self.space_h,
                                w-(self.space_w+self.dw):w-self.space_w])
            ]

            if self.thre_type == "lt":
                min_avg = min(corners_avg)
                conf = 1.0
                if min_avg < self.thre:
                    label = 'ng'
                    # linear scale pixel average range between 50~thre to 1~0.6
                    conf = (self.thre - 0.4*min_avg - 35)/(self.thre - 50)
            else:
                max_avg = max(corners_avg)
                conf = 1.0
                if max_avg > self.thre:
                    label = 'ng'
                    # linear scale pixel average range from thre~230 to 0.6~1
                    conf = (0.4*max_avg - self.thre + 138)/(230 - self.thre)

            res = lab_tools.output_pred_classification(
                input_path="",
                conf=conf,
                label=label,
                img_size=(w, h),
                labinfo=[]
            )

            # this classifier only select one type of ng
            if label == 'ok':
                res['annotations'] = []
            self.results.append(res)


# FIXME: this is a workaround for issue 93, need to remove this
#        once issue 93 is fixed
class ClassifierMobileNet2(ClassifierInceptionv3):
    """ Modified from label_image.py example in TensorFlow """

    def __init__(self, dyda_config_path='', testing=False):
        """ __init__ of ClassifierMobileNet2

        Trainer Variables:
            input_data: a list of image array
            results: defined by lab_tools.output_pred_classification (dict)

        Arguments:
            testing -- True to call set_testing_params. Anything written in
                       dyda.config will be overwritten.
        """

        super(ClassifierMobileNet2, self).__init__(
            dyda_config_path=dyda_config_path, testing=testing
        )
        self.logger.info("Initialization of ClassifierMobileNet2 done.")


# FIXME: this is a workaround for issue 93, need to remove this
#        once issue 93 is fixed
class ClassifierAoiCornerAvg2(ClassifierAoiCornerAvg):
    """ Simple average to select corner defect samples.
        This should be run after ClassifierAoiCV

    """

    def __init__(self, dyda_config_path="", param=None):
        """Initialization function of dyda component.
           This is called when an object is created from the class and
           it allows the class to initialize the attributes of a class.
        """

        super(ClassifierAoiCornerAvg2, self).__init__(
            dyda_config_path=dyda_config_path
        )
