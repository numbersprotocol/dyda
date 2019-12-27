import unittest
from dyda_utils import image
from dyda_utils import tools
from dyda_utils import lab_tools
from dyda.components.classifier import ClassifierInceptionv3
from dyda.components.classifier import ClassifierMobileNet
from dyda.components.classifier import ClassifierAoiCV
from dyda.components.classifier import ClassifierAoiCornerAvg
from dyda_utils import dict_comparator


class TestClassifierInceptionv3(unittest.TestCase):
    """ Unit test of ClassifierInceptionv3 component """

    def test_main_process(self):

        # pull test data from gitlab
        config_url = ("https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads"
                      "/b6fec0ae2bc4c0913e3e765c2a87b135/"
                      "dyda.config.inceptionv3_unittest")
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)

        img_url = ("https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/"
                   "4fa6b8a454f9b4041bdde316bac85a27/IMAG1450.jpg")
        ref_url = ("https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/"
                   "1ca61414f75fa6b93e09944a3bf1b66b/"
                   "ref_json.json.inceptionv3_unittest")
        img = lab_tools.pull_img_from_gitlab(img_url)
        ref_data = lab_tools.pull_json_from_gitlab(ref_url)

        classifier = ClassifierInceptionv3(dyda_config_path=dyda_config)
        classifier.input_data = [img]
        classifier.run()

        tar_data = classifier.results[0]
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestClassifierMobileNet(unittest.TestCase):
    """ Unit test of ClassifierMobileNet component """

    def test_main_process(self):

        # pull test data from gitlab
        config_url = ("https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/"
                      "abf5cf6f40a4b31216b0dabbe81a6863/"
                      "dyda.config.mobilenet_unittest")
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)

        img_url = ("https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/"
                   "4fa6b8a454f9b4041bdde316bac85a27/IMAG1450.jpg")
        ref_url = ("https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/"
                   "8ea9ac61e3403981eca23c658b2cdac7/"
                   "ref_results.json.mobilenet")
        img = lab_tools.pull_img_from_gitlab(img_url)
        ref_data = lab_tools.pull_json_from_gitlab(ref_url)

        classifier = ClassifierMobileNet(dyda_config_path=dyda_config)
        classifier.input_data = [img]
        classifier.run()

        tar_data = classifier.results[0]
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestClassifierAoiCV(unittest.TestCase):
    def test_main_process(self):

        # pull test data from gitlab
        config_url = ("https://gitlab.com/DT42/galaxy42/dt42-dyda/"
                      "uploads/6bd42ac664cecf4fdc29a4ed894874d2/"
                      "dyda.config.ClassifierAoiCornerAvg")
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)
        img_url = ("https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/"
                   "697de0fe3c7d5a530674ddd996a0129a/NG_TYPE2_0063.png")
        ref_url = ("https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/"
                   "316639f6278bfd28f04b605eb16aef57/"
                   "test_ClassifierAoiCV.json")
        img = lab_tools.pull_img_from_gitlab(img_url)
        ref_data = lab_tools.pull_json_from_gitlab(ref_url)

        classifier = ClassifierAoiCV(dyda_config_path=dyda_config)
        classifier.input_data = [img]
        classifier.run()

        tar_data = classifier.results[0]
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestClassifierAoiCornerAvg(unittest.TestCase):
    """ Unit test of ClassifierAoiCornerAvg component """

    def test_main_process(self):

        # pull test data from gitlab
        config_url = ("https://gitlab.com/DT42/galaxy42/dt42-dyda/"
                      "uploads/6bd42ac664cecf4fdc29a4ed894874d2/"
                      "dyda.config.ClassifierAoiCornerAvg")
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)

        img_url = ("https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/"
                   "697de0fe3c7d5a530674ddd996a0129a/NG_TYPE2_0063.png")
        ref_url = ("https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/"
                   "1cd293379a63b3a843bfe47e754e495f/"
                   "test_ClassifierAoiCornerAvg.json")
        img = lab_tools.pull_img_from_gitlab(img_url)
        ref_data = lab_tools.pull_json_from_gitlab(ref_url)[2]["results"][0]

        classifier = ClassifierAoiCornerAvg(
            dyda_config_path=dyda_config
        )
        classifier.input_data = [img]
        classifier.run()

        tar_data = classifier.results[0]
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
