import os
import sys
import unittest
import tarfile
import requests
from dyda_utils import image
from dyda_utils import tools
from dyda_utils import lab_tools
from dyda_utils import dict_comparator
from dyda.components import labelme_data_preprocessor as preprocessor


class TestLabelmeDataPreProcessor(unittest.TestCase):
    def test_main_process(self):
        # pull test data from gitlab
        img_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
                  'd2ebc6170f05578273b934cfb6ed7344/pict000301.jpg'

        json_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
                   'c8457db3f0d2b1ddb1fd7f6730168b9c/metadata_pic301.json'

        result_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
                     '952964ae5dbdb06eec74d716d7636576/pict000301_result.json'

        img = lab_tools.pull_img_from_gitlab(img_url)
        meta_data = lab_tools.pull_json_from_gitlab(json_url)
        ref_data = lab_tools.pull_json_from_gitlab(result_url)

        # initialization
        data = [img, meta_data]
        data_preprocessor_ = preprocessor.LabelMeDataPreProcessor()
        data_preprocessor_.meta_data = meta_data

        data_preprocessor_.input_data = data
        data_preprocessor_.run()

        tar_data = data_preprocessor_.results
        ref_data['data_path'] = []
        report = dict_comparator.get_diff(ref_data, tar_data)

        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
