import os
import sys
import unittest
import tarfile
import requests
from dt42lab.core import image
from dt42lab.core import tools
from dt42lab.core import lab_tools
from dt42lab.utility import dict_comparator
from dyda.components import relabeler


class TestRelabeler(unittest.TestCase):
    def test_main_process(self):
        # pull test data from gitlab

        label_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
                    '4a0011c0182b4f4a10514d04b9cd6c80/label_data.json'

        config_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
                     'a02929bcfae180946acec0637c000357/config.json'

        change_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
                     '959f91b50077b97476f3db55521bd53c/o_change_relabel.json'

        remove_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
                     '58a6bc8dba68de782930a1e37105a33c/o_rm_relabel.json'

        label_data = lab_tools.pull_json_from_gitlab(label_url)
        config_data = lab_tools.pull_json_from_gitlab(config_url)
        ref_change_data = lab_tools.pull_json_from_gitlab(change_url)
        ref_remove_data = lab_tools.pull_json_from_gitlab(remove_url)

        relabeler_ = relabeler.Relabeler()

        relabeler_.set_config(label_data, config_data, True)
        relabeler_.set_change_flag()
        relabeler_.run()
        chg_result = relabeler_.output_data.copy()

        relabeler_.set_config(label_data, config_data, True)
        relabeler_.set_remove_flag()
        relabeler_.run()
        rm_result = relabeler_.output_data.copy()

        change_report = dict_comparator.get_diff(ref_change_data, chg_result)
        remove_report = dict_comparator.get_diff(ref_remove_data, rm_result)

        self.assertEqual(change_report['extra_field'], [])
        self.assertEqual(change_report['missing_field'], [])
        self.assertEqual(change_report['mismatch_val'], [])

        self.assertEqual(remove_report['extra_field'], [])
        self.assertEqual(remove_report['missing_field'], [])
        self.assertEqual(remove_report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
