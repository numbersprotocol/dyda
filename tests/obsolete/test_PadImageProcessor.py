import cv2
import unittest
from dyda_utils import tools
from dyda_utils import tinycv
from dyda_utils import lab_tools
from dt42lab.utility import dict_comparator
from dyda.components.image_processor import PadImageProcessor


# pull test data from gitlab
print('[Test_PadImageProcessor] INFO: Pull 11 KB files from gitlab. ')
input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '05e15e0a9133b30f5bfbe6c02d6847a4/input_img.png.0'
input_data = lab_tools.pull_img_from_gitlab(input_url)
config_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '8db80481bbf1b8d35b141ddd48858f37/PadImageProcessor.config'
dyda_config_PadImageProcessor = lab_tools.pull_json_from_gitlab(
    config_url)
output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '25d9f94e6f47ec5571ed3c77c258fdfc/TestPadImageProcessor_simple.bmp.0'
output_PadImageProcessor = lab_tools.pull_img_from_gitlab(output_url)
results_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    'aed3023ac8e4ac89a0e53e7b0e07aec7/TestPadImageProcessor_simple.json'
results_PadImageProcessor = lab_tools.pull_json_from_gitlab(results_url)


class TestPadImageProcessor_simple(unittest.TestCase):
    """ Test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = PadImageProcessor(
            dyda_config_path=dyda_config_PadImageProcessor)

        # run component
        comp.reset()
        comp.input_data = input_data
        comp.run()

        # compare output_data with reference
        ref_data = output_PadImageProcessor
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)

        # compare results with reference
        ref_data = results_PadImageProcessor
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestPadImageProcessor_double(unittest.TestCase):
    """ Double test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = PadImageProcessor(
            dyda_config_path=dyda_config_PadImageProcessor)

        # run component
        comp.reset()
        comp.input_data = input_data
        comp.run()
        comp.input_data = input_data
        comp.run()

        # compare output_data with reference
        ref_data = output_PadImageProcessor
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)

        # compare results with reference
        ref_data = results_PadImageProcessor
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestPadImageProcessor_list(unittest.TestCase):
    """ Test list of input. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = PadImageProcessor(
            dyda_config_path=dyda_config_PadImageProcessor)

        # run component
        comp.reset()
        comp.input_data.append(input_data)
        comp.run()

        # compare output_data with reference
        ref_data = output_PadImageProcessor
        tar_data = comp.output_data[0]
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)

        # compare results with reference
        ref_data = results_PadImageProcessor
        tar_data = comp.results[0]
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
