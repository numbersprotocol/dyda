import unittest
from dt42lab.core import data
from dyda.components.determinator import DeterminatorConfidenceThreshold
from dt42lab.utility import dict_comparator


class TestDeterminatorConfidenceThreshold(unittest.TestCase):
    def test_main_process(self):

        in_file = '/home/shared/DT42/dyda_test/frame/00000015.json'
        out_file = ('/home/shared/DT42/dyda_test/ref_results/'
                    'determinator_output/output_metadata/00000015.json')

        determinator_ = DeterminatorConfidenceThreshold()
        determinator_.input_metadata = data.parse_json(in_file)
        determinator_.main_process()

        ref_path = out_file
        tar_path = determinator_.output_metadata
        report = dict_comparator.get_diff(ref_path, tar_path)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
