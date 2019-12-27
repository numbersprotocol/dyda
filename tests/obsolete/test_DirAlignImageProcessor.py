import cv2
import unittest
from dyda_utils import tools
from dyda_utils import tinycv
from dyda_utils import lab_tools
from dyda_utils import dict_comparator
from dyda.components.image_processor import DirAlignImageProcessor
import numpy as np


input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '05e15e0a9133b30f5bfbe6c02d6847a4/input_img.png.0'
input_data = lab_tools.pull_img_from_gitlab(input_url)
output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '8eb946c4ed8f59dd70d4e2a02fd99859/output_vertical_ccw.bmp'
output_vertical = lab_tools.pull_img_from_gitlab(output_url)
result_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '2b0ae9fc2932384682798dcb0fa2a6db/results_vertical_ccw.json'
result_vertical_ccw = lab_tools.pull_json_from_gitlab(result_url)
result_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    'd895390c85d521ae62283843c8b11b96/results_vertical_cw.json'
result_vertical_cw = lab_tools.pull_json_from_gitlab(result_url)
output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '841881cf718f8d89c41d7e4e6ef8ebe0/output_horizontal_ccw.bmp'
output_horizontal_ccw = lab_tools.pull_img_from_gitlab(output_url)
result_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '12967f1b99a2a6fc4e3ccf8c65aed046/results_horizontal_ccw.json'
result_horizontal_ccw = lab_tools.pull_json_from_gitlab(result_url)
output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    'dbaf2039dfb868ef3852b0394fdc433a/output_horizontal_cw.bmp'
output_horizontal_cw = lab_tools.pull_img_from_gitlab(output_url)
result_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '1c2db558b2920891a2a6bf3e0c179b16/results_horizontal_cw.json'
result_horizontal_cw = lab_tools.pull_json_from_gitlab(result_url)
output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '81d79b48910e830f5ed8bd4406196910/output_horizontal_ccw_grey.bmp'
output_grey = lab_tools.pull_img_from_gitlab(output_url)
result_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '81c89609ee0fb407ef8f4ab15b1c5979/results_horizontal_ccw_grey.json'
result_grey = lab_tools.pull_json_from_gitlab(result_url)


class TestDirAlignImageProcessor_vertical_ccw(unittest.TestCase):
    """ Test case of vertical and counterclcokwise. """

    def test_main_process(self):
        """ Main process of unit test. """

        dyda_config_DirAlignImageProcessor = {
            "DirAlignImageProcessor": {"chosen_direction": "vertical",
                                       "rotate_direction": "ccw"}}

        # initialization
        comp = DirAlignImageProcessor(
            dyda_config_path=dyda_config_DirAlignImageProcessor)

        # run component
        comp.reset()
        comp.input_data = [input_data]
        comp.run()

        # compare output_data with reference
        ref_data = output_vertical
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(ref_data, tar_data)
        self.assertEqual(img_diff, 0.0)

        # compare results with reference
        ref_data = result_vertical_ccw
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestDirAlignImageProcessor_vertical_cw(unittest.TestCase):
    """ Test case of vertical and clockwise. """

    def test_main_process(self):
        """ Main process of unit test. """

        dyda_config_DirAlignImageProcessor = {
            "DirAlignImageProcessor": {"chosen_direction": "vertical",
                                       "rotate_direction": "cw"}}

        # initialization
        comp = DirAlignImageProcessor(
            dyda_config_path=dyda_config_DirAlignImageProcessor)

        # run component
        comp.reset()
        comp.input_data = [input_data]
        comp.run()

        # compare output_data with reference

        ref_data = output_vertical
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(ref_data, tar_data)
        self.assertEqual(img_diff, 0.0)

        # compare results with reference
        ref_data = result_vertical_cw
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestDirAlignImageProcessor_horizontal_ccw(unittest.TestCase):
    """ Test case of horizontal and counterclockwise. """

    def test_main_process(self):
        """ Main process of unit test. """

        dyda_config_DirAlignImageProcessor = {
            "DirAlignImageProcessor": {"chosen_direction": "horizontal",
                                       "rotate_direction": "ccw"}}

        # initialization
        comp = DirAlignImageProcessor(
            dyda_config_path=dyda_config_DirAlignImageProcessor)

        # run component
        comp.reset()
        comp.input_data = [input_data]
        comp.run()

        # compare output_data with reference

        ref_data = output_horizontal_ccw
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(ref_data, tar_data)
        self.assertEqual(img_diff, 0.0)

        # compare results with reference
        ref_data = result_horizontal_ccw
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestDirAlignImageProcessor_horizontal_cw(unittest.TestCase):
    """ Test case of horizontal and clockwise. """

    def test_main_process(self):
        """ Main process of unit test. """

        dyda_config_DirAlignImageProcessor = {
            "DirAlignImageProcessor": {"chosen_direction": "horizontal",
                                       "rotate_direction": "cw"}}

        # initialization
        comp = DirAlignImageProcessor(
            dyda_config_path=dyda_config_DirAlignImageProcessor)

        # run component
        comp.reset()
        comp.input_data = input_data
        comp.run()

        # compare output_data with reference
        ref_data = output_horizontal_cw
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(ref_data, tar_data)
        self.assertEqual(img_diff, 0.0)

        # compare results with reference
        ref_data = result_horizontal_cw
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestDirAlignImageProcessor_horizontal_ccw_grey(unittest.TestCase):
    """ Test case of grey picture(one channel). """

    def test_main_process(self):
        """ Main process of unit test. """

        dyda_config_DirAlignImageProcessor = {
            "DirAlignImageProcessor": {"chosen_direction": "horizontal",
                                       "rotate_direction": "ccw"}}

        # initialization
        comp = DirAlignImageProcessor(
            dyda_config_path=dyda_config_DirAlignImageProcessor)

        # run component
        comp.reset()
        comp.input_data = input_data[:, :, 0]
        comp.run()

        # compare output_data with reference
        ref_data = output_grey
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)

        # compare results with reference
        ref_data = result_grey
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
