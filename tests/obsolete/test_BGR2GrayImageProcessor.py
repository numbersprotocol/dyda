import cv2
import unittest
from dt42lab.core import tools
from dt42lab.core import tinycv
from dt42lab.core import lab_tools
from dt42lab.utility import dict_comparator
from dyda.components.image_processor import BGR2GrayImageProcessor


# pull test data from gitlab
print('[Test_BGR2GrayImageProcessor] INFO: Pull 9 KB files from gitlab. ')
input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '05e15e0a9133b30f5bfbe6c02d6847a4/input_img.png.0'
input_data = lab_tools.pull_img_from_gitlab(input_url)
config_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '886e16adbfb8d630ead9ab74e88d113c/BGR2GrayImageProcessor.config'
dyda_config_BGR2GrayImageProcessor = lab_tools.pull_json_from_gitlab(
    config_url)
output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    'baf7b36e1aff5a1abfb623af7fbf09a7/TestBGR2GrayImageProcessor_simple.bmp.0'
output_BGR2GrayImageProcessor = lab_tools.pull_img_from_gitlab(output_url)


class TestBGR2GrayImageProcessor_simple(unittest.TestCase):
    """ Test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = BGR2GrayImageProcessor(
            dyda_config_path=dyda_config_BGR2GrayImageProcessor)

        # run component
        comp.reset()
        comp.input_data = input_data
        comp.run()

        # compare output_data with reference
        ref_data = output_BGR2GrayImageProcessor
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)


class TestBGR2GrayImageProcessor_double(unittest.TestCase):
    """ Double test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = BGR2GrayImageProcessor(
            dyda_config_path=dyda_config_BGR2GrayImageProcessor)

        # run component
        comp.reset()
        comp.input_data = input_data
        comp.run()
        comp.input_data = input_data
        comp.run()

        # compare output_data with reference
        ref_data = output_BGR2GrayImageProcessor
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)


class TestBGR2GrayImageProcessor_list(unittest.TestCase):
    """ Test list of input. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = BGR2GrayImageProcessor(
            dyda_config_path=dyda_config_BGR2GrayImageProcessor)

        # run component
        comp.reset()
        comp.input_data.append(input_data)
        comp.run()

        # compare output_data with reference
        ref_data = output_BGR2GrayImageProcessor
        tar_data = comp.output_data[0]
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)


if __name__ == '__main__':
    unittest.main()
