import cv2
import unittest
from dyda_utils import tools
from dyda_utils import tinycv
from dyda_utils import lab_tools
from dyda_utils import dict_comparator
from dyda.components.determinator import DeterminatorSizeThreshold


input_data = {
    "folder": None,
    "filename": None,
    "size": {
        "width": 100,
        "height": 50
    },
    "annotations": [
        {"top": 0, "bottom": 5, "left": 0, "right": 15, "id": 0},
        {"top": 20, "bottom": 30, "left": 0, "right": 20, "id": 1},
        {"top": 30, "bottom": 50, "left": 20, "right": 50, "id": 2}
    ]
}

output_data = {
    "folder": None,
    "filename": None,
    "size": {
        "width": 100,
        "height": 50
    },
    "annotations": [
        {"top": 20, "bottom": 30, "left": 0, "right": 20, "id": 0},
        {"top": 30, "bottom": 50, "left": 20, "right": 50, "id": 1}
    ]
}

dyda_config_digit = {
    "DeterminatorSizeThreshold": {"threshold": 0.1}}

dyda_config_pixel = {
    "DeterminatorSizeThreshold": {"threshold": 10}}


class TestDeterminatorSizeThreshold_dict_thres_digit(unittest.TestCase):
    """ Test case of inputting a dictionary with treshold less than one. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = DeterminatorSizeThreshold(
            dyda_config_path=dyda_config_digit)

        # run component
        comp.reset()
        comp.input_data = input_data
        comp.run()

        # compare output_data with reference
        ref_data = output_data
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestDeterminatorSizeThreshold_dict_thres_pixel(unittest.TestCase):
    """ Test case of inputting a dictionary with threshold greater than one. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = DeterminatorSizeThreshold(
            dyda_config_path=dyda_config_pixel)

        # run component
        comp.reset()
        comp.input_data = input_data
        comp.run()

        # compare output_data with reference
        ref_data = output_data
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestDeterminatorSizeThreshold_list_thres_digit(unittest.TestCase):
    """ Test case of inputting a list of dictionaries with threshold less than one. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = DeterminatorSizeThreshold(
            dyda_config_path=dyda_config_digit)

        # run component
        comp.reset()
        comp.input_data = [input_data]
        comp.run()

        # compare output_data with reference
        ref_data = [output_data]
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestDeterminatorSizeThreshold_list_thres_pixel(unittest.TestCase):
    """ Test case of inputting a list of dictionaries with threshold greater than one. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = DeterminatorSizeThreshold(
            dyda_config_path=dyda_config_pixel)

        # run component
        comp.reset()
        comp.input_data = [input_data]
        comp.run()

        # compare output_data with reference
        ref_data = [output_data]
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
