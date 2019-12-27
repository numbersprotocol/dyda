import os
import unittest
from dt42lab.core import image
from dt42lab.core import data
from dt42lab.core import tools
from dt42lab.core import lab_tools
from dt42lab.core import tinycv
from dyda.components.cv_learner import LearnerSimpleCV
from dt42lab.utility import dict_comparator


class TestLearnerSimpleCV(unittest.TestCase):
    def test_main_process(self):

        # pull test data from gitlab
        input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            'b7abe78697075dc2e47de5327c9f2de3/cv_classifier_input_list.json'
        input_list = lab_tools.pull_json_from_gitlab(input_url)
        output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '01fd55515c6373ced2541f0aec287f44/ref_bkg.jpg'
        ref_output = lab_tools.pull_img_from_gitlab(output_url)

        # initialization
        learner_ = LearnerSimpleCV()

        image_list = []
        for i in range(len(input_list)):
            image_list.append(image.read_img(input_list[i]))

        # run classifier
        learner_.reset()
        learner_.input_data = image_list
        learner_.run()
        diff = tinycv.l1_norm_diff_cv2(learner_.output_data, ref_output)
        self.assertEqual(sum(sum(diff)), 0.0)


if __name__ == '__main__':
    unittest.main()
