import os
import unittest
from dyda_utils import data
from dyda_utils import tools
from dyda_utils import lab_tools
from dyda.components.frame_selector import FrameSelectorDownsampleFirst
from dyda_utils import dict_comparator


class TestFrameSelectorDownsampleFirst(unittest.TestCase):
    def test_main_process(self):

        # pull test data from gitlab
        input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '06e852b64cd24cdb017a56c540e195db/FrameSelectorDownsampleFirst_input_list.json'
        input_list = lab_tools.pull_json_from_gitlab(input_url)
        output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '6ce81dd346759aef1d8f4cc81675662e/FrameSelectorDownsampleFirst_output_list.json'
        output_list = lab_tools.pull_json_from_gitlab(output_url)

        # initialization
        frame_selector_ = FrameSelectorDownsampleFirst()

        for i in range(len(input_list)):

            # run frame_selector
            frame_selector_.reset()
            frame_selector_.metadata[0] = tools.remove_extension(
                input_list[i]['filename'],
                'base-only')
            frame_selector_.run()

            # compare results with reference
            ref_data = output_list[i]
            tar_data = frame_selector_.results
            report = dict_comparator.get_diff(ref_data, tar_data)
            self.assertEqual(report['extra_field'], [])
            self.assertEqual(report['missing_field'], [])
            self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
