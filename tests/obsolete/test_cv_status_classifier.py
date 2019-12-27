import os
import unittest
from dyda_utils import image
from dyda_utils import data
from dyda_utils import tools
from dyda_utils import lab_tools
from dyda.components.cv_status_classifier import ClassifierSimpleCV
from dyda_utils import dict_comparator


class TestClassifierSimpleCV(unittest.TestCase):
    def test_main_process(self):

        # pull test data from gitlab
        input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            'b7abe78697075dc2e47de5327c9f2de3/cv_classifier_input_list.json'
        input_list = lab_tools.pull_json_from_gitlab(input_url)
        output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            'c9c8c98992abdf43036eed69d0e5fc3f/cv_output_list.json'
        output_list = lab_tools.pull_json_from_gitlab(output_url)

        # initialization
        classifier_ = ClassifierSimpleCV()
        classifier_.param["bkg_ref_path"] = '/home/shared/DT42/test_data/'\
            'test_demo_app_with_calibration/ref_bkg.png'

        image_list = []
        for i in range(len(input_list)):
            image_list.append(image.read_img(input_list[i]))

        # run classifier
        classifier_.reset()
        classifier_.input_data = image_list
        classifier_.run()

        # compare results with reference
        for j in range(len(classifier_.results)):
            ref_data = output_list[j]['annotations'][0]['label']
            tar_data = classifier_.results[j]['annotations'][0]['label']
            self.assertEqual(ref_data, tar_data)


if __name__ == '__main__':
    unittest.main()
