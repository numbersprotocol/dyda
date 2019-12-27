import cv2
import unittest
from dyda_utils import tools
from dyda_utils import tinycv
from dyda_utils import lab_tools
from dyda_utils import dict_comparator
from dyda.components.image_processor import MergeImageProcessor


# pull test data from gitlab
print('[Test_MergeImageProcessor] INFO: Pull 44 KB files from gitlab. ')
input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '93cae1cc3aa4d6e1dcf9b9c052b1bc3b/00000001.png.0'
input_data_0 = lab_tools.pull_img_from_gitlab(input_url)
input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '1bcd5ac6691702dbbe445e2712f0f59f/00000370.png.0'
input_data_1 = lab_tools.pull_img_from_gitlab(input_url)
input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '4d471f2dd9750d9ebedfb90f0a8d7d3e/00000290.png.0'
input_data_2 = lab_tools.pull_img_from_gitlab(input_url)
input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '10a38fd6449a7838689a30f2fcea0d88/00000030.png.0'
input_data_3 = lab_tools.pull_img_from_gitlab(input_url)
config_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '7b0fb06a326646bd696fee48722b3f12/MergeImageProcessor.config'
dyda_config_MergeImageProcessor = lab_tools.pull_json_from_gitlab(
    config_url)
output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '11c4450c29fbc111daa115e55c48100f/TestMergeImageProcessor_simple.bmp.0'
output_MergeImageProcessor = lab_tools.pull_img_from_gitlab(output_url)


class TestMergeImageProcessor_simple(unittest.TestCase):
    """ Test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = MergeImageProcessor(
            dyda_config_path=dyda_config_MergeImageProcessor)

        # run component
        comp.reset()
        comp.input_data = [input_data_0, input_data_1,
                           input_data_2, input_data_3]
        comp.run()

        # compare output_data with reference
        ref_data = output_MergeImageProcessor
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)


class TestMergeImageProcessor_double(unittest.TestCase):
    """ Double test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = MergeImageProcessor(
            dyda_config_path=dyda_config_MergeImageProcessor)

        # run component
        comp.reset()
        comp.input_data = [input_data_0, input_data_1,
                           input_data_2, input_data_3]
        comp.run()
        comp.input_data = [input_data_0, input_data_1,
                           input_data_2, input_data_3]
        comp.run()

        # compare output_data with reference
        ref_data = output_MergeImageProcessor
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)


class TestMergeImageProcessor_list(unittest.TestCase):
    """ Test list of input. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = MergeImageProcessor(
            dyda_config_path=dyda_config_MergeImageProcessor)

        # run component
        comp.reset()
        comp.input_data = [[input_data_0, input_data_1,
                            input_data_2, input_data_3]]
        comp.run()

        # compare output_data with reference
        ref_data = output_MergeImageProcessor
        tar_data = comp.output_data[0]
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)


if __name__ == '__main__':
    unittest.main()
