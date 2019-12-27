"""
unit session in pipeline config:
{
    "name": "merge_detection_converter",
    "component": "data_converter",
    "class": "MergeDetectionConverter",
    "type": "normal",
    "input_type": "extend",
    "additional_info": {"input_data": [
        ["image_merger", "metadata"],
        ["label_determinator", "metadata"]
    ]}
}
which means:
1. $COMPONENT=data_converter
2. $CLASS=MergeDetectionConverter
3. input_data[0] is metadata(results) from image_merger
4. input_data[1] is metadata(results) from label_determinator
(other input types please reference dyda/pipeline/pipeline.py)
"""

import unittest
from dyda_utils import image
from dyda_utils import data
from dyda_utils import tools
from dyda_utils import lab_tools
# step1: from dyda.components.$COMPONENT import $CLASS
from dyda.components.data_converter import MergeDetectionConverter
from dt42lab.utility import dict_comparator

# step2: class Test$CLASS(unittest.TestCase):


class TestMergeDetectionConverter(unittest.TestCase):
    def test_main_process(self):

        # step3: upload config, input and output data to
        #     https://gitlab.com/DT42/galaxy42/dt42-dyda/issues/24
        #     and modify the urls
        config_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            'cde8519d76f5c7942d2712544eae6c6c/'\
            'dyda.config.MergeDetectionConverter'
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)
        input_url_0 = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            'e6f61815bac541233331f8b260e1ec5f/'\
            'MergeDetectionConverter_input_0.json'
        input_data_0 = lab_tools.pull_json_from_gitlab(input_url_0)
        input_url_1 = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '89ad7d554c42ec99fe28ea69f81397b5/'\
            'MergeDetectionConverter_input_1.json'
        input_data_1 = lab_tools.pull_json_from_gitlab(input_url_1)
        output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '998d86b29265f4005cb87e2b799bad56/'\
            'MergeDetectionConverter_output.json'
        output_data = lab_tools.pull_json_from_gitlab(output_url)

        # step4: unit = $CLASS(
        unit = MergeDetectionConverter(
            dyda_config_path=dyda_config)

        unit.reset()
        # step5: make sure feed correct input_data as pipeline config described
        unit.input_data.extend(input_data_0)
        unit.input_data.extend(input_data_1)
        unit.run()

        ref_data = output_data
        tar_data = unit.results
        report = dict_comparator.get_diff(
            ref_data, tar_data, ignore_keys=['filename'])
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
