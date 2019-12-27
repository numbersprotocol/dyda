import unittest
from dt42lab.core import lab_tools
from dt42lab.core import tools
from dyda.components.validator import ClassificationValidator
from dyda.components.validator import DetectionValidator
from dt42lab.utility import dict_comparator


class TestClassificationValidator(unittest.TestCase):
    def test_main_process(self):

        config_url = ('https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'
                      '99e8dae78033b647562b72835aa84f5e/'
                      'dyda.config.validator')
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)
        output_url = ('https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'
                      'e9d7eea7eff43a67ac4c8f9bae3ec3ee/'
                      'ref_results_validator.json')
        label_txt_url = ('https://gitlab.com/DT42/galaxy42/dt42-dyda/'
                         'uploads/f1cf0a5cacb0e6dca35f0b400f9ce9e6/labels.txt')
        labels = lab_tools.pull_json_from_gitlab(label_txt_url)

        results_url = ('https://gitlab.com/DT42/galaxy42/dt42-dyda/'
                       'uploads/6350c824e163ead49f4849145a3d116d/'
                       'classification.txt')
        classification_results = lab_tools.pull_json_from_gitlab(results_url)

        # initialization
        validator = ClassificationValidator(dyda_config)
        validator.reset()
        validator.input_data = [classification_results, labels]
        validator.run()

        ref_data = lab_tools.pull_json_from_gitlab(output_url)
        tar_data = validator.results
        if not ref_data == [] and not tar_data == []:
            report = dict_comparator.get_diff(ref_data, tar_data)
            self.assertEqual(report['extra_field'], [])
            self.assertEqual(report['missing_field'], [])
            self.assertEqual(report['mismatch_val'], [])


class TestDetectionValidator(unittest.TestCase):
    def test_main_process(self):


        config_url = ('https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'
                      'a38ca41418635e32dcbc82a46a25937f/'
                      'DetectionValidator.config')
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)
        ref_url = ('https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'
                      'b5b1692ae296653840758710304fb17f/ref_data.json')
        ref_data = lab_tools.pull_json_from_gitlab(ref_url)
        gt_url = ('https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'
                  '66fbbb03bdc4be60e78a3385a457dfc5/ground_truth_data_all.json')
        gt = lab_tools.pull_json_from_gitlab(gt_url)
        det_url = ('https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'
                   'bd96dfe182f45acc7c1034d1fdc11c9c/prediction_data_all.json')
        det = lab_tools.pull_json_from_gitlab(det_url)

        # initialization
        validator = DetectionValidator(dyda_config)
        validator.reset()
        validator.input_data = [det, gt]
        validator.run()

        tar_data = validator.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
