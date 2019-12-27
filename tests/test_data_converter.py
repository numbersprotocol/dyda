import unittest
import numpy as np
from dt42lab.core import tools
from dt42lab.core import lab_tools
from dyda.components.data_converter import IrConverter
from dyda.components.data_converter import TimeScaleShiftConverter
from dyda.components.data_converter import PathLabelConverter
from dt42lab.utility import dict_comparator


class TestPathLabelConverter(unittest.TestCase):
    def test_main_process(self):

        config_url = ('https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'
                      'bb1bfe0785760aa7308f139a16bc4e76/'
                      'path_label_converter_dyda.config')
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)
        input_url = ('https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'
                     '7ad69fd8e0674843c74382f4d2ef6a16/'
                     'path_label_converter_input.json')
        input_list = lab_tools.pull_json_from_gitlab(input_url)
        output_url = ('https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'
                      'c2317edcab9d460e193827f1eb213568/'
                      'path_label_converter_results.json')

        # initialization
        converter_ = PathLabelConverter(dyda_config)
        converter_.reset()
        converter_.input_data = {'data_path': input_list}
        converter_.run()

        ref_data = lab_tools.pull_json_from_gitlab(output_url)
        tar_data = converter_.results
        if not ref_data == [] and not tar_data == []:
            report = dict_comparator.get_diff(ref_data, tar_data)
            self.assertEqual(report['extra_field'], [])
            self.assertEqual(report['missing_field'], [])
            self.assertEqual(report['mismatch_val'], [])


class TestTimeScaleShiftConverter(unittest.TestCase):
    def test_main_process(self):

        config_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '2f4b1e105fad5d935e83fa8e608c395e/'\
            'dyda.config.TimeScaleShiftConverter'
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)
        input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            'a25e50ea3b063d3e57aed402de9d0d25/input_list.json'
        input_list = lab_tools.pull_json_from_gitlab(input_url)
        output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '4e8e4849534e68152507bdf3640d5bd1/output.json'
        output_list = lab_tools.pull_json_from_gitlab(output_url)

        # initialization
        converter_ = TimeScaleShiftConverter(dyda_config)

        for i in range(len(input_list)):

            # run converter
            converter_.reset()
            converter_.input_data.append(
                tools.parse_json(input_list[i]))
            converter_.run()

            # compare results with reference
            ref_data = output_list[i]
            tar_data = converter_.results
            if not ref_data == [] and not tar_data == []:
                report = dict_comparator.get_diff(ref_data, tar_data)
                self.assertEqual(report['extra_field'], [])
                self.assertEqual(report['missing_field'], [])
                self.assertEqual(report['mismatch_val'], [])


class TestIrConverter(unittest.TestCase):
    def test_main_process(self):

        config_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '065d9f508fafe96da77d4e02fbd4fc12/dyda.config.IrConverter'
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)

        input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '25ddbb2fc6e6dfc0d6375b6d568f7c56/input_data.temp'
        input_data = lab_tools.pull_json_from_gitlab(input_url)

        output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '473956c9bdadd7847795d423b172e78b/output_data.json'
        output_data = lab_tools.pull_json_from_gitlab(output_url)

        # initialization
        converter_ = IrConverter(dyda_config_path=dyda_config)

        # run converter
        converter_.reset()
        converter_.input_data.append(input_data)
        converter_.run()

        # compare results with reference
        ref_data = output_data
        tar_data = list(converter_.output_data)
        diff = [(ref_data[i] - tar_data[i]) for i in range(
            len(ref_data))]
        diff_sum = sum(sum(diff))
        self.assertEqual(diff_sum, 0.0)


if __name__ == '__main__':
    unittest.main()
