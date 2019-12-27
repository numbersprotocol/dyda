import os
import sys
import copy
import unittest
import requests
from dt42lab.core import tools
from dt42lab.core import lab_tools
from dt42lab.utility import dict_comparator
from dyda.components import batch_relabeler


class TestRelabeler(unittest.TestCase):
    def test_main_process(self):
        # pull test data from gitlab

        input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
                    '2b480d6f274c139f53a8baedd430c73a/input.json'

        config_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
                     '44fd1a6559c633e5d590fa8a43a0b1ca/config.json'

        chg0_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
                   '92f79b095a92af655513a933f00f8255/tar_chg0.json'

        chg1_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
                   '6bc8007cf68e262c9dcadc2d10adad49/tar_chg1.json'

        rm0_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
                  '645593b73feb5dbe48bac07defdfb56d/tar_rm0.json'

        input_data = lab_tools.pull_json_from_gitlab(input_url)
        config_data = lab_tools.pull_json_from_gitlab(config_url)
        ref_chg0 = lab_tools.pull_json_from_gitlab(chg0_url)
        ref_chg1 = lab_tools.pull_json_from_gitlab(chg1_url)
        ref_rm0 = lab_tools.pull_json_from_gitlab(rm0_url)
        #input_data['folder'] = input_data['folder'].replace('/home/shared/customer_data/acti/201711-ACTi-A', 
        #    '/home/shared/lab/wd-temp-external-4t/customer_data/acti/201711-ACTi-B')
        ref_chg0['folder'] = ref_chg0['folder'].replace('/home/shared/customer_data/acti/201711-ACTi-A',
            '/home/shared/lab/wd-temp-external-4t/customer_data/acti/201711-ACTi-B')
        ref_chg1['folder'] = ref_chg1['folder'].replace('/home/shared/customer_data/acti/201711-ACTi-A',
            '/home/shared/lab/wd-temp-external-4t/customer_data/acti/201711-ACTi-B')
        ref_rm0['folder'] = ref_rm0['folder'].replace('/home/shared/customer_data/acti/201711-ACTi-A',
            '/home/shared/lab/wd-temp-external-4t/customer_data/acti/201711-ACTi-B')

        relabel_tool_ = batch_relabeler.BatchRelabeler()
        relabel_tool_.window_name = 'detection_relabel'
        relabel_tool_.config_data = config_data

        # user defined tool to pick which annotation to relabel
        # idx: index of annotations
        relabel_tool_.init_deposit_list(input_data)

        # set_change_flag(idx): change label of idx-th annotation
        relabel_tool_.set_change_flag(10)
        relabel_tool_.run()
        relabel_tool_.deposit2output()
        tar_chg0 = copy.deepcopy(relabel_tool_.output_data)[0]
        tools.write_json(tar_chg0, 'tar_chg0.json')
        dict_comparator.restore_ordered_json('tar_chg0.json')

        relabel_tool_.set_change_flag(2)
        relabel_tool_.run()
        relabel_tool_.deposit2output()
        tar_chg1 = copy.deepcopy(relabel_tool_.output_data)[0]
        tools.write_json(tar_chg1, 'tar_chg1.json')
        dict_comparator.restore_ordered_json('tar_chg1.json')

        # set_remove_flag(idx): remove the label of idx-th annotation
        relabel_tool_.set_remove_flag(8)
        relabel_tool_.run()
        relabel_tool_.deposit2output()
        tar_rm0 = copy.deepcopy(relabel_tool_.output_data)[0]

        chg0_report = dict_comparator.get_diff(ref_chg0, tar_chg0)
        chg1_report = dict_comparator.get_diff(ref_chg1, tar_chg1)
        rm0_report = dict_comparator.get_diff(ref_rm0, tar_rm0)

        self.assertEqual(chg0_report['extra_field'], [])
        self.assertEqual(chg0_report['missing_field'], [])
        self.assertEqual(chg0_report['mismatch_val'], [])

        self.assertEqual(chg1_report['extra_field'], [])
        self.assertEqual(chg1_report['missing_field'], [])
        self.assertEqual(chg1_report['mismatch_val'], [])

        self.assertEqual(rm0_report['extra_field'], [])
        self.assertEqual(rm0_report['missing_field'], [])
        self.assertEqual(rm0_report['mismatch_val'], [])

if __name__ == '__main__':
    unittest.main()
