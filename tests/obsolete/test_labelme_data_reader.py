import os
import sys
import unittest
import tarfile
import requests
from dyda_utils import image
from dyda_utils import tools
from dyda_utils import lab_tools
from dyda_utils import dict_comparator
from dyda.components import labelme_data_reader as reader


class TestLabelmeDataReader(unittest.TestCase):
    def test_main_process(self):
        # pull test data from gitlab
        tar_url = "https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/"\
                  "3111a2fd3690b775e7e088aa97d26330/labelme_example.tar.gz"

        golden1_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
                      '572998d818f6bceb5bf89346427243b7/golden_pic303.json'

        golden2_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
                      '64d34d53518c44a68b7904ed4c1b8265/golden_pic301.json'

        output_list = list()
        output_list.append(lab_tools.pull_json_from_gitlab(golden1_url))
        output_list.append(lab_tools.pull_json_from_gitlab(golden2_url))

        # download tar.gz file
        self.pull_gz_from_gitlab(tar_url)

        # unzip the tar.gz file
        targz_path = os.path.basename(tar_url)
        tar = tarfile.open(targz_path, "r")
        tar.extractall()

        # get the all the image path in the folder
        unzip_folder = os.path.basename(tar_url)[:-7]
        img_paths = image.find_images(unzip_folder)

        # initialization
        data_reader_ = reader.LabelMeDataReader()

        for idx, img_path in enumerate(img_paths):
            data_reader_.reset_results()
            data_reader_.img_path = img_path
            data_reader_.run()

            # compare results with reference
            ref_data = output_list[idx]
            tar_data = data_reader_.results
            ref_data['folder'] = ref_data['folder'].replace('/home/hki541/dt42-dyda/tests', os.getcwd())
            report = dict_comparator.get_diff(ref_data, tar_data)
            self.assertEqual(report['extra_field'], [])
            self.assertEqual(report['missing_field'], [])
            self.assertEqual(report['mismatch_val'], [])

    def pull_gz_from_gitlab(self, url):
        unzip_folder = os.path.basename(url)[:-7]
        targz_path = os.path.basename(url)

        token_path = "./gitlab_token.json"
        token = lab_tools.get_gitlab_token(token_path)
        headers = {'PRIVATE-TOKEN': token}
        response = requests.get(url, headers=headers)
        status = response.status_code

        if status == 200:
            try:
                with open(targz_path, "wb") as f:
                    r = requests.get(url, headers=headers)
                    f.write(r.content)

            except BaseException:
                print('[dt42lab] ERROR: Fail to get json from gitlab.'
                      'Check if token is set correctly or if the url is right')
                sys.exit(1)
        else:
            print('[dt42lab] ERROR: Fail with status_code %i.' % status)
            sys.exit(1)


if __name__ == '__main__':
    unittest.main()
