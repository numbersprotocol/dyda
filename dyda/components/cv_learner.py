import os
import cv2
import pickle
from sklearn.mixture import GaussianMixture
from dyda_utils import tools
from dyda_utils import image
from dyda.core import learner_base


class LearnerSimpleCV(learner_base.LearnerBase):
    """ Use simple CV to classify event """

    def __init__(self, dyda_config_path=''):
        """ __init__ of LearnerSimpleCV """

        super(LearnerSimpleCV, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

        # initialize output normal image
        self.normal_image = []

    def main_process(self):
        """ define main_process of dyda component """

        for img_array in self.input_data:
            self.normal_learner_first(img_array)
            self.output_data = self.normal_image

    def post_process(self, out_folder_base=""):
        """ define post_process of dyda component """

        if not tools.check_exist(out_folder_base, log=False):
            out_folder_base = os.path.join(os.getcwd(), 'post_process')
        self.base_logger.info('post_process results saved to %s'
                              % out_folder_base)
        tools.dir_check(out_folder_base)
        self.results["output_folder"] = out_folder_base
        self.results["bkg_ref_basename"] = "ref_bkg.png"

        out_filename = os.path.join(
            self.results["output_folder"],
            self.results["bkg_ref_basename"])
        image.save_img(self.normal_image, fname=out_filename)

    def normal_learner_first(self, input_image):
        """ output the first image as reference background image(normal) """
        if self.normal_image == []:
            self.normal_image = input_image



class LearnerGaussianMixtureModel(learner_base.LearnerBase):
    """ Build Gaussian mixture model of nomal status """

    def __init__(self, dyda_config_path=''):
        """ __init__ of LearnerNormalModel """

        super(LearnerNormalModel, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

        self.results = {
            'model_path': self.param['model_path']}

    def main_process(self):
        """ define main_process of dyda component """

        ori_shape = self.input_data[0].shape
        imgs = []
        for img in self.input_data:
            imgs.append(img.reshape(-1,3))
        imgs = np.asarray(imgs)

        gmms=[]
        for k in range(imgs.shape[1]):
            X = imgs[:,k,:]
            gmm = GaussianMixture(
                self.param['covariance_type'],
                self.param['n_components'])
            gmm.fit(X)
            gmms.append(gmm)
        self.output_data = gmms


    def post_process(self, out_folder_base=""):
        """ define post_process of dyda component """

        model_path = self.param["model_path"]
        f = open(model_path, 'wb')
        pickle.dump(self.output_data, f)
        f.close()
        print('[LearnerNormalModel] post_process results saved to %s'
            % model_path)


