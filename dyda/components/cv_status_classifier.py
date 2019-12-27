import random
import pickle
import numpy as np
from dyda_utils import lab_tools
from dyda_utils import image
from dyda_utils import tinycv
from dyda.core import classifier_base


class ClassifierSimpleCV(classifier_base.ClassifierBase):
    """ Use simple CV to classify event """

    def __init__(self, dyda_config_path=''):
        """ __init__ of ClassifierSimpleCV

        Variables:
            input_data: a list of image array
            results: defined by lab_tools.output_pred_classification (dict)

        """

        super(ClassifierSimpleCV, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

        self.check_param_keys([
            "bkg_ref_path",
            "re_width",
            "diff_thre",
            "pixel_num_min",
            "kernel_size",
            "iter_num",
            "seg_margin"])

        self.orig_input_path = "BINARY_INPUT"
        self.bkg_ref = image.read_img(self.param["bkg_ref_path"])

    def get_testing_result(self, img_array):
        """ main CV algorithm """
        bounding_box = tinycv.foreground_extraction_by_ccl(
            img_array,
            self.bkg_ref,
            re_width=self.param["re_width"],
            diff_thre=self.param["diff_thre"],
            pixel_num_min=self.param["pixel_num_min"],
            kernel_size=self.param["kernel_size"],
            iter_num=self.param["iter_num"],
            seg_margin=self.param["seg_margin"])

        fake_conf = random.uniform(0.7, 0.95)

        if bounding_box == []:
            res = {"conf": fake_conf, "label": "normal"}
        else:
            res = {"conf": fake_conf, "label": "anomaly"}

        return res

    def main_process(self):
        """ define main_process of dyda component """

        for img_array in self.input_data:
            algo_output = self.get_testing_result(img_array)

            res = lab_tools.output_pred_classification(
                input_path=self.orig_input_path,
                conf=algo_output["conf"],
                label=algo_output["label"],
                img_size=(img_array.shape[1], img_array.shape[0])
            )

            self.results.append(res)
            self.output_data.append(img_array)


class ClassifierGaussianMixtureModel(classifier_base.ClassifierBase):
    """ Use Gaussian Mixture Model to classify anomaly event """

    def __init__(self, dyda_config_path=''):
        """ __init__ of ClassifierGaussianMixtureModel

        Variables:
            input_data: a list of image array
            results: defined by lab_tools.output_pred_classification (dict)

        """

        super(ClassifierGaussianMixtureModel, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

        self.orig_input_path = "BINARY_INPUT"
        self.check_param_keys([
            "model_path"])
        with open(self.param["model_path"], 'rb') as handle:
            self.gmms = pickle.load(handle)
        self.accumulated_score = []

    def gmms_func(self, x, i):
        return abs(self.gmms[i].score([x]))

    def get_diff_img(self, img):
        """ main CV algorithm, it took 3s to use CPU"""

        ori_shape = img.shape
        img = img.reshape(-1, self.param['channel_num'])
        score = [self.gmms_func(x, i) for i, x in enumerate(img[:])]
        score = np.asarray(score)
        score = score.reshape(ori_shape[0], ori_shape[1])
        return score

    def get_output_result(self, diff_img):
        if self.accumulated_score == []:
            self.accumulated_score = np.zeros(diff_img.shape)
        self.accumulated_score = np.where(
            diff_img > self.param['diff_thre'],
            self.accumulated_score + 1, 0)
        temporal_score = np.zeros(diff_img.shape)
        temporal_score = np.where(
            self.accumulated_score >= self.param['temporal_thre'], 1, 0)
        self.output_data.append(temporal_score * 255)
        if sum(sum(temporal_score)) >= self.param['pixel_thre']:
            label = 'anomaly'
        else:
            label = 'normal'
        return label

    def main_process(self):
        """ define main_process of dyda component """

        self.output_data = []
        self.results = []
        for img in self.input_data:
            diff_img = self.get_diff_img(img)
            res = lab_tools.output_pred_classification(
                input_path=self.orig_input_path,
                label=self.get_output_result(diff_img),
                # FIXME: fake conf
                conf=random.uniform(0.7, 0.95)
            )
            self.results.append(res)
