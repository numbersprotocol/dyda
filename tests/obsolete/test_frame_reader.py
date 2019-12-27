import unittest
import os
from dyda_utils import lab_tools
from dyda.components.frame_reader import FrameReader
from dyda_utils import dict_comparator


class TestFrameReader(unittest.TestCase):
    def test_main_process(self):

        # pull test data from gitlab
        config_url = ("https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads"
                      "/418f17cdcdb4c9ee7555ef467e042a8c/dyda.config")
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)

        cwd = os.getcwd()
        filename = ".testing_img.jpg"
        local_path = os.path.join(cwd, filename)

        img_url = ("https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/"
                   "4fa6b8a454f9b4041bdde316bac85a27/IMAG1450.jpg")
        img = lab_tools.pull_img_from_gitlab(img_url, save_to=local_path)

        ref_url = ("https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/"
                   "4d57dd72be420cda1acea39f7665d76c/"
                   "ref_data.json.framereader_unittest")
        ref_data = lab_tools.pull_json_from_gitlab(ref_url)

        reader = FrameReader(dyda_config_path=dyda_config)
        reader.input_data = [local_path]
        reader.run()

        tar_data = reader.results

        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])

        # Since the filepath depending on cwd, the folder wont match anyway
        for mis_match in report['mismatch_val']:
            for tar_key in mis_match['tar_key']:
                if isinstance(tar_key, list) and tar_key[0] == 'data_path':
                    _tar_val = os.path.basename(mis_match['tar_val'])
                    _ref_val = os.path.basename(mis_match['ref_val'])
                    self.assertEqual(_tar_val, _ref_val)
                else:
                    self.assertEqual(report['mismatch_val'], [])
        os.remove(local_path)

if __name__ == '__main__':
    unittest.main()
