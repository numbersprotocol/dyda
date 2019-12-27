import cv2
import unittest
from dyda_utils import tools
from dyda_utils import tinycv
from dyda_utils import lab_tools
from dyda_utils import dict_comparator
from dyda.components.image_processor import ExtractNonBlackImageProcessor

input_data = cv2.imread('/home/shared/DT42/test_data/test_ExtractNonBlackImageProcessor/input.jpg')
ref_output = cv2.imread('/home/shared/DT42/test_data/test_ExtractNonBlackImageProcessor/ref_output.bmp')
ref_results = tools.parse_json('/home/shared/DT42/test_data/test_ExtractNonBlackImageProcessor/ref_results.json')

class TestExtractNonBlackImageProcessor_simple(unittest.TestCase):
    """ Test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = ExtractNonBlackImageProcessor()

        # run component
        comp.reset()
        comp.input_data = input_data
        comp.run()

        # compare output_data with reference
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(tar_data, ref_output)
        self.assertEqual(img_diff, 0.0)

        # compare results with reference
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_results, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])



if __name__ == '__main__':
    unittest.main()
