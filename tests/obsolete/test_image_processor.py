import cv2
import unittest
import numpy as np
from dyda_utils import tools
from dyda_utils import tinycv
from dyda_utils import lab_tools
from dt42lab.utility import dict_comparator
from dyda.components.image_processor import BGR2RGBImageProcessor
from dyda.components.face_recognizer import FaceEncodingImageProcessor


# pull test data from gitlab
input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            'a73f92e9b271420d48f70481f56944c4/2019-03-15-111527.jpg'
input_data = lab_tools.pull_img_from_gitlab(input_url)


class TestBGR2RGBImageProcessor(unittest.TestCase):

    def test_main_process(self):
        """ Main process of unit test. """

        # Test simple case
        dyda_config = {"BGR2RGBImageProcessor": {}}

        # initialization
        comp = BGR2RGBImageProcessor(
            dyda_config_path=dyda_config)

        # run component
        comp.reset()
        comp.input_data = input_data.copy()
        comp.run()

        # compare output_data with reference
        ref_data = cv2.cvtColor(input_data.copy(), cv2.COLOR_BGR2RGB)
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)

        # Test list input case
        dyda_config = {"BGR2RGBImageProcessor": {}}

        # initialization
        comp = BGR2RGBImageProcessor(
            dyda_config_path=dyda_config)

        # run component
        comp.reset()
        comp.input_data = [input_data.copy()]
        comp.run()

        # compare output_data with reference
        ref_datas = [cv2.cvtColor(input_data.copy(), cv2.COLOR_BGR2RGB)]
        tar_datas = comp.output_data

        for ref_data, tar_data in zip(ref_datas, tar_datas):
            img_diff = lab_tools.img_comparator(tar_data, ref_data)
            self.assertEqual(img_diff, 0.0)


face_detect_results = {'filename': '', 'size': {'width': 960, 'height': 544},
                       'annotations': [{'left': 414, 'right': 637, 'top': 192,
                                        'labinfo': {}, 'bottom': 415,
                                        'type': 'detection', 'id': 0,
                                        'label': 'face', 'confidence': -1.0}],
                       'folder': ''}

face_encoding_output = \
    [np.array(
        [-1.15552954e-01, 9.72168297e-02, 1.25082694e-02, -3.45896855e-02,
         -8.94738585e-02, -1.45118400e-01, -1.46271419e-02, -2.04383448e-01,
         1.47198588e-01, -4.26624641e-02, 2.27250263e-01, -6.78653345e-02,
         -2.08296269e-01, -1.15748361e-01, -3.84995714e-03, 1.58455670e-01,
         -1.85800105e-01, -1.39850110e-01, -5.89484796e-02, -4.87370975e-03,
         7.84208179e-02, -2.78235376e-02, 3.56304906e-02, 8.77563208e-02,
         -1.51798412e-01, -3.31332028e-01, -1.01870939e-01, -5.77603616e-02,
         -3.03587671e-02, -4.92708646e-02, -1.64172389e-02, 8.35793018e-02,
         -2.29068846e-01, -9.47869793e-02, 1.46344379e-02, 6.46413565e-02,
         7.12211709e-03, -5.56275882e-02, 1.98764384e-01, 1.91945657e-02,
         -2.69295812e-01, 1.08501986e-02, 5.09751365e-02, 2.67260075e-01,
         2.10388273e-01, 5.82690910e-03, 2.54405960e-02, -1.25058517e-01,
         1.45375192e-01, -1.68090999e-01, 7.35666370e-03, 1.45047739e-01,
         2.71528475e-02, -3.54558229e-04, 1.77285019e-02, -1.10368013e-01,
         4.74006385e-02, 1.55690312e-01, -2.07276717e-01, 1.95026807e-02,
         1.34549990e-01, -1.59524113e-01, -4.69095856e-02, -9.56463888e-02,
         2.43623167e-01, 6.78730011e-02, -1.19833902e-01, -1.40978172e-01,
         6.00612573e-02, -1.41941816e-01, -7.37254322e-02, 8.80655795e-02,
         -1.49219260e-01, -2.03093350e-01, -4.09248441e-01, 3.76143940e-02,
         4.11778331e-01, 9.38788205e-02, -1.89899385e-01, 7.51832277e-02,
         -5.03525808e-02, 2.34285556e-02, 1.66729033e-01, 1.70459494e-01,
         -1.16258524e-02, 6.03590012e-02, -1.47511557e-01, 4.81127203e-03,
         2.01997429e-01, -6.53343555e-03, -3.21282744e-02, 2.69905508e-01,
         -4.18500118e-02, 5.54613695e-02, 1.93587914e-02, 4.03433964e-02,
         -1.13657624e-01, 1.58958882e-03, -8.00858289e-02, -2.44916379e-02,
         1.01078369e-01, -2.36223973e-02, 2.44647600e-02, 1.20564178e-01,
         -1.49256796e-01, 8.39020088e-02, -2.67281160e-02, 7.26455599e-02,
         9.90013592e-03, -7.77477697e-02, -3.91532145e-02, -8.65772963e-02,
         7.21973851e-02, -2.61380255e-01, 1.74670324e-01, 1.32593185e-01,
         -3.73279257e-03, 9.82683823e-02, 5.90415411e-02, 6.06970675e-02,
         -8.13767537e-02, 2.76792645e-02, -1.68079942e-01, 2.91468669e-02,
         1.31535456e-01, -2.85522491e-02, 1.61660716e-01, -2.03275196e-02]
    )
    ]


class TestFaceEncodingImageProcessor(unittest.TestCase):

    def test_main_process(self):
        """ Main process of unit test. """

        # Test simple case.
        dyda_config = {"FaceEncodingImageProcessor": {}}

        # initialization
        comp = FaceEncodingImageProcessor(
            dyda_config_path=dyda_config)

        # run component
        comp.reset()
        comp.input_data = [input_data.copy(), face_detect_results]
        comp.run()

        # compare output_data with reference
        ref_data = face_encoding_output[0].round(5)
        tar_data = comp.output_data[0].round(5)

        self.assertEqual((ref_data == tar_data).all(), True)

        # Test list input case.
        dyda_config = {"FaceEncodingImageProcessor": {}}

        # initialization
        comp = FaceEncodingImageProcessor(
            dyda_config_path=dyda_config)

        # run component
        comp.reset()
        comp.input_data = [[input_data.copy()], [face_detect_results]]
        comp.run()

        # compare output_data with reference
        ref_datas = [face_encoding_output]
        tar_datas = comp.output_data
        for ref_data, tar_data in zip(ref_datas, tar_datas):
            ref_data = ref_data[0].round(5)
            tar_data = tar_data[0].round(5)

        self.assertEqual((ref_data == tar_data).all(), True)


if __name__ == '__main__':
    unittest.main()
