import cv2
import unittest
from dt42lab.core import tools
from dt42lab.core import tinycv
from dt42lab.core import lab_tools
from dt42lab.utility import dict_comparator
from dyda.components.image_processor import ResizeImageProcessor


# pull test data from gitlab
print('[Test_ResizeImageProcessor] INFO: Pull 18 KB files from gitlab. ')
input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '05e15e0a9133b30f5bfbe6c02d6847a4/input_img.png.0'
input_data = lab_tools.pull_img_from_gitlab(input_url)
config_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '89e5f0abb7fc8a8f1591a45a105e6fca/dyda.config.ResizeImageProcessor'
dyda_config_ResizeImageProcessor = lab_tools.pull_json_from_gitlab(
    config_url)
output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '42a4b5bc1dc5f0ec8aad0ac58abdb699/TestResizeImageProcessor_simple.bmp.0'
output_ResizeImageProcessor = lab_tools.pull_img_from_gitlab(output_url)
results_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    'd291d335efe85ca594207fb4ff301a7b/TestResizeImageProcessor_simple.json'
results_ResizeImageProcessor = lab_tools.pull_json_from_gitlab(results_url)
output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '42a4b5bc1dc5f0ec8aad0ac58abdb699/TestResizeImageProcessor_simple.bmp.0'
output_ResizeImageProcessor = lab_tools.pull_img_from_gitlab(output_url)


class TestResizeImageProcessor_simple(unittest.TestCase):
    """ Test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = ResizeImageProcessor(
            dyda_config_path=dyda_config_ResizeImageProcessor)

        # run component
        comp.reset()
        comp.input_data = input_data
        comp.run()

        # compare output_data with reference
        ref_data = output_ResizeImageProcessor
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)

        # compare results with reference
        ref_data = results_ResizeImageProcessor
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestResizeImageProcessor_double(unittest.TestCase):
    """ Double test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = ResizeImageProcessor(
            dyda_config_path=dyda_config_ResizeImageProcessor)

        # run component
        comp.reset()
        comp.input_data = input_data
        comp.run()
        comp.input_data = input_data
        comp.run()

        # compare output_data with reference
        ref_data = output_ResizeImageProcessor
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)

        # compare results with reference
        ref_data = results_ResizeImageProcessor
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestResizeImageProcessor_list(unittest.TestCase):
    """ Test list of input. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = ResizeImageProcessor(
            dyda_config_path=dyda_config_ResizeImageProcessor)

        # run component
        comp.reset()
        comp.input_data = [input_data]
        comp.run()

        # compare output_data with reference
        ref_data = output_ResizeImageProcessor
        tar_data = comp.output_data[0]
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)

        # compare results with reference
        ref_data = results_ResizeImageProcessor
        tar_data = comp.results[0]
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
