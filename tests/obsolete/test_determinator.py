import unittest
from dt42lab.core import image
from dt42lab.core import data
from dt42lab.core import tools
from dt42lab.core import lab_tools
from dyda.components.determinator import DeterminatorByRoi
from dyda.components.determinator import DeterminatorParkingLotStatus
from dyda.components.determinator import DeterminatorConfidenceThreshold
from dyda.components.determinator import DeterminatorTargetLabel
from dyda.components.determinator import DeterminatorCharacter
from dt42lab.utility import dict_comparator


class TestDeterminatorByRoi(unittest.TestCase):
    def test_main_process(self):

        # pull test data from gitlab
        config_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '8710dfd057501758190778cb056e4449/dyda.config'
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)
        input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            'a9324bdf32ed5a1e237be94fbc74c5d5/input.json'
        input_data = lab_tools.pull_json_from_gitlab(input_url)
        output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '0f801ae6692bcc168ba1c5045b806c78/res.json'
        output_data = lab_tools.pull_json_from_gitlab(output_url)

        # initialization
        determinator_ = DeterminatorByRoi(
            dyda_config_path=dyda_config)

        # run determinator
        determinator_.reset()
        determinator_.input_data.append(input_data)
        determinator_.run()

        # compare results with reference
        ref_data = output_data
        tar_data = determinator_.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestDeterminatorCharacter(unittest.TestCase):
    def test_main_process(self):

        # pull test data from gitlab
        config_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '36356e92cad4a608d4c84bba769c0d53/'\
            'dyda.config.DeterminatorCharacter'
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)
        input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '521feb8def6e63d6e187622b171b7233/input_list.json'
        input_list = lab_tools.pull_json_from_gitlab(input_url)
        output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '9f25865b93a133bdec459ace16de8ccd/output_list.json'
        output_list = lab_tools.pull_json_from_gitlab(output_url)

        # initialization
        determinator_ = DeterminatorCharacter(
            dyda_config_path=dyda_config)
        # run determinator
        for i in range(len(input_list)):

            # run determinator
            determinator_.reset()
            determinator_.input_data.append(image.read_img(input_list[i]))
            determinator_.input_data.append(
                tools.parse_json(tools.replace_extension(
                    input_list[i], 'json')))
            determinator_.run()

            # compare results with reference
            if not determinator_.results == []:
                ref_data = output_list[i]
                tar_data = determinator_.results
                for j in range(len(ref_data)):
                    for k in range(len(ref_data[j])):
                        report = dict_comparator.get_diff(
                            ref_data[j][k], tar_data[j][k])
                        self.assertEqual(report['extra_field'], [])
                        self.assertEqual(report['missing_field'], [])
                        self.assertEqual(report['mismatch_val'], [])


class TestDeterminatorParkingLotStatus(unittest.TestCase):
    def test_main_process(self):

        # pull test data from gitlab
        config_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            'abc67841cfb173c22cc13087fdd4d7b2/dyda.config'
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)
        input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '83db1facaef1bc83efc2b9ed25256ad1/park_input_list.json'
        input_list = lab_tools.pull_json_from_gitlab(input_url)
        output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '775252d8456257f3364f788556bb553c/park_output_list.json'
        output_list = lab_tools.pull_json_from_gitlab(output_url)
        frame_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            'aa4675a8057f3ae79bf5c0153cdfc69e/'\
            'DeterminatorParkingLotStatus_frame_list.json'
        frame_list = lab_tools.pull_json_from_gitlab(frame_url)

        # initialization
        determinator_ = DeterminatorParkingLotStatus(
            dyda_config_path=dyda_config)

        # run determinator
        for i in range(len(output_list)):

            # run determinator
            determinator_.reset()
            determinator_.input_data.append([image.read_img(frame_list[i])])
            determinator_.input_data.append(input_list[i][0])
            determinator_.input_data.append(input_list[i][1])
            determinator_.input_data.append(input_list[i][2])
            determinator_.run()

            # compare results with reference
            if not determinator_.results == []:
                ref_data = output_list[i][0]
                tar_data = determinator_.results[0]
                report = dict_comparator.get_diff(ref_data, tar_data)
                self.assertEqual(report['extra_field'], [])
                self.assertEqual(report['missing_field'], [])
                self.assertEqual(report['mismatch_val'], [])


class TestDeterminatorTargetLabel(unittest.TestCase):
    def test_main_process(self):

        # pull test data from gitlab
        config_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '79187835280b7adc8149e5e9a8e6ec9c/DeterminatorTargetLabel.config'
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)
        input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '4cdb639a0f0862f504e3223713da1c49/'\
            'DeterminatorTargetLabel_input.json'
        input_data = lab_tools.pull_json_from_gitlab(input_url)
        output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            'fd45424cee6c68c7de28fc16f6af734d/'\
            'DeterminatorTargetLabel_output.json'
        output_data = lab_tools.pull_json_from_gitlab(output_url)

        # initialization
        determinator_ = DeterminatorTargetLabel(
            dyda_config_path=dyda_config)

        # run determinator
        determinator_.reset()
        determinator_.metadata[0] = tools.remove_extension(
            input_data["filename"],
            'base-only')
        determinator_.input_data.append(input_data)
        determinator_.run()

        # compare results with reference
        ref_path = output_data
        tar_path = determinator_.results[0]
        report = dict_comparator.get_diff(ref_path, tar_path)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestDeterminatorConfidenceThreshold(unittest.TestCase):
    def test_main_process(self):

        # pull test data from gitlab
        config_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '2b588582c1ad5b901f6b4cc1c8f81e15/dyda.config'
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)

        input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '41ed033f30b132a0a48d7ba214f75c41/00000015.json'
        input_data = lab_tools.pull_json_from_gitlab(input_url)
        output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '73b6ab545229971f1149ade93dc1ca8f/00000015.json'
        output_data = lab_tools.pull_json_from_gitlab(output_url)

        # initialization
        determinator_ = DeterminatorConfidenceThreshold(dyda_config)

        # run determinator
        determinator_.reset()
        determinator_.metadata[0] = tools.remove_extension(
            input_data["filename"],
            'base-only')
        determinator_.input_data.append(input_data)
        determinator_.run()

        # compare results with reference
        ref_path = output_data
        tar_path = determinator_.results
        report = dict_comparator.get_diff(ref_path, tar_path)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
