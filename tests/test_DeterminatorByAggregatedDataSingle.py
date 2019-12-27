import unittest
from dyda.components.determinator import DeterminatorByAggregatedDataSingle
from dyda_utils import dict_comparator
from dyda_utils import lab_tools


class TestDeterminatorByAggregatedDataSingle(unittest.TestCase):
    def test_main_process(self):
        ref_data = [
            ['others', '1.00'], ['others', '1.00'],
            ['others', '0.90'], ['person', '0.90'],
            ['others', '0.83'], ['others', '0.80']
        ]
        r1 = lab_tools.output_pred_classification("", 1.0, "others")
        r2 = lab_tools.output_pred_classification("", 0.9, "person")
        r3 = lab_tools.output_pred_classification("", 0.8, "others")
        r4 = lab_tools.output_pred_classification("", 0.7, "others")
        r5 = lab_tools.output_pred_classification("", 0.2, "person")
        r6 = lab_tools.output_pred_classification("", 0.9, "others")
        results = []
        d = DeterminatorByAggregatedDataSingle()
        for r in [r1, r2, r3, r4, r5, r6]:
            d.input_data = r
            d.run()
            results.append([
                d.results["annotations"][0]["label"],
                "%0.2f" % d.results["annotations"][0]["confidence"]
            ])
        report = dict_comparator.get_diff(ref_data, results)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
