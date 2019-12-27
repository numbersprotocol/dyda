import cv2
import unittest
from dyda_utils import tools
from dyda_utils import tinycv
from dyda_utils import lab_tools
from dt42lab.utility import dict_comparator
from dyda.components.image_processor import CalibrateImageProcessor


# pull test data from gitlab
print('[Test_CalibrateImageProcessor] INFO: Pull 189 KB files from gitlab. ')
config_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '6ad6cb384843092a563a6f3b20a86672/dyda.config.CalibrateImageProcessor'
dyda_config_CalibrateImageProcessor = lab_tools.pull_json_from_gitlab(
    config_url)
input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '63ee0fac22eb43882dcefa3ec154df54/frame_781.jpg.0'
input_data_calibrate = lab_tools.pull_img_from_gitlab(input_url)
output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    'f3dc2a06bef9759fb92549f1d0491a5e/TestCalibrateImageProcessor_simple.bmp.0'
output_CalibrateImageProcessor = lab_tools.pull_img_from_gitlab(output_url)


class TestCalibrateImageProcessor_simple(unittest.TestCase):
    """ Test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = CalibrateImageProcessor(
            dyda_config_path=dyda_config_CalibrateImageProcessor)

        # run component
        comp.reset()
        comp.input_data = input_data_calibrate
        comp.run()

        # compare output_data with reference
        ref_data = output_CalibrateImageProcessor
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)


class TestCalibrateImageProcessor_double(unittest.TestCase):
    """ Double test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = CalibrateImageProcessor(
            dyda_config_path=dyda_config_CalibrateImageProcessor)

        # run component
        comp.reset()
        comp.input_data = input_data_calibrate
        comp.run()
        comp.input_data = input_data_calibrate
        comp.run()

        # compare output_data with reference
        ref_data = output_CalibrateImageProcessor
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)


class TestCalibrateImageProcessor_list(unittest.TestCase):
    """ Test list of input. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = CalibrateImageProcessor(
            dyda_config_path=dyda_config_CalibrateImageProcessor)

        # run component
        comp.reset()
        comp.input_data = [input_data_calibrate]
        comp.run()

        # compare output_data with reference
        ref_data = output_CalibrateImageProcessor
        tar_data = comp.output_data[0]
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)


if __name__ == '__main__':
    unittest.main()
