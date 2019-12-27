import cv2
import unittest
from dyda_utils import tools
from dyda_utils import tinycv
from dyda_utils import lab_tools
from dt42lab.utility import dict_comparator
from dyda.components.box_processor import ResizeBoxProcessor

input_data = {
    "size": {"width": 1920, "height": 1080},
    "annotations": [
        {
            "type": "detection",
            "id": 1,
            "label": "person",
            "track_id": 11,
            "top": 306,
            "bottom": 391,
            "left": 635,
            "right": 673,
            "confidence": -1
        }
    ]
}

resize_info = {
    'ori_size': {
        'height': 1080,
        'width': 1920
    },
    'new_size': {
        'height': 360,
        'width': 640
    }
}

ref_data = {
    "size": {"width": 640, "height": 360},
    "annotations": [
        {
            "type": "detection",
            "id": 1,
            "label": "person",
            "track_id": 11,
            "top": 102,
            "bottom": 130,
            "left": 211,
            "right": 224,
            "confidence": -1
        }
    ]
}


class TestResizeBoxProcessor(unittest.TestCase):
    """ Test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = ResizeBoxProcessor()

        # run component
        comp.reset()
        comp.input_data = [resize_info, input_data]
        comp.run()

        ## compare output_data with reference
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])

if __name__ == '__main__':
    unittest.main()
