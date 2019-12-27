import cv2
import unittest
from dyda_utils import tools
from dyda_utils import tinycv
from dyda_utils import lab_tools
from dyda_utils import dict_comparator
from dyda.components.determinator import DeterminatorByRoi


class TestDeterminatorByRoi_OneRoi(unittest.TestCase):
    """ Test case of one roi. """

    def test_main_process(self):
        """ Main process of unit test. """

        dyda_config_DeterminatorByRoi = {
            "DeterminatorByRoi": {"use_external_meta": True}}
        input_data = {
            "folder": None,
            "filename": None,
            "size": None,
            "annotations":[
                {"top":0, "bottom":10, "left":0, "right":10},
                {"top":20, "bottom":30, "left":0, "right":10}
            ]
        }
        external_metadata = {
            "roi": [
                {"top":3, "bottom":12, "left":0, "right":10, "overlap_threshold":0.5}
            ]
        }
        output_data = {
            "folder": None,
            "filename": None,
            "size": None,
            "annotations":[
                {"top":0, "bottom":10, "left":0, "right":10, "id":0}
            ]
        }

        # initialization
        comp = DeterminatorByRoi(
            dyda_config_path=dyda_config_DeterminatorByRoi)

        # run component
        comp.reset()
        comp.input_data = input_data
        comp.external_metadata = external_metadata
        comp.run()

        ## compare output_data with reference
        ref_data = output_data
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestDeterminatorByRoi_TwoRoi(unittest.TestCase):
    """ Test case of one roi. """

    def test_main_process(self):
        """ Main process of unit test. """

        dyda_config_DeterminatorByRoi = {
            "DeterminatorByRoi": {"use_external_meta": True}}
        input_data = {
            "folder": None,
            "filename": None,
            "size": None,
            "annotations":[
                {"top":0, "bottom":10, "left":0, "right":10},
                {"top":20, "bottom":30, "left":0, "right":10},
                {"top":40, "bottom":50, "left":0, "right":10}
            ]
        }
        external_metadata = {
            "roi": [
                {"top":3, "bottom":12, "left":0, "right":10, "overlap_threshold":0.5},
                {"top":23, "bottom":32, "left":0, "right":10, "overlap_threshold":0.5}
            ]
        }
        output_data = {
            "folder": None,
            "filename": None,
            "size": None,
            "annotations":[
                {"top":0, "bottom":10, "left":0, "right":10, "id":0},
                {"top":20, "bottom":30, "left":0, "right":10, "id":1}
            ]
        }

        # initialization
        comp = DeterminatorByRoi(
            dyda_config_path=dyda_config_DeterminatorByRoi)

        # run component
        comp.reset()
        comp.input_data = input_data
        comp.external_metadata = external_metadata
        comp.run()

        ## compare output_data with reference
        ref_data = output_data
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])



if __name__ == '__main__':
    unittest.main()
