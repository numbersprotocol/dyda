import cv2
import unittest
from dt42lab.core import tools
from dt42lab.core import tinycv
from dt42lab.core import lab_tools
from dt42lab.utility import dict_comparator
from dyda.components.image_processor import CropRoiImageProcessor


# pull test data from gitlab
print('[Test_CropRoiImageProcessor] INFO: Pull 34 KB files from gitlab. ')
input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '05e15e0a9133b30f5bfbe6c02d6847a4/input_img.png.0'
input_data = lab_tools.pull_img_from_gitlab(input_url)
input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '63ee0fac22eb43882dcefa3ec154df54/frame_781.jpg.0'
input_data_calibrate = lab_tools.pull_img_from_gitlab(input_url)
config_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '78a5e35b2802ddbee8614aefbceb2fdb/dyda.config.CropRoiImageProcessor'
dyda_config_CropRoiImageProcessor = lab_tools.pull_json_from_gitlab(
    config_url)
metadata_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    'ddc0b4970202e99c3a1248c5af66480e/TestResizeImageProcessor_metadata.json'
metadata_CropRoiImageProcessor = lab_tools.pull_json_from_gitlab(metadata_url)
output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    'c8c952e97929e1e3f47c15740f2b7dc4/TestCropRoiImageProcessor_simple.bmp.0'
output_CropRoiImageProcessor = lab_tools.pull_img_from_gitlab(output_url)
results_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '473726f1c287475360e7a177dbb2b4aa/TestCropRoiImageProcessor_simple.json'
results_CropRoiImageProcessor = lab_tools.pull_json_from_gitlab(results_url)
output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    'a9bf448d4c9831559053bde0dea0c595/TestCropRoiImageProcessor_metadata.bmp.0'
output_CropRoiImageProcessor_m = lab_tools.pull_img_from_gitlab(output_url)
results_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    'eafb01647855d671bfa3e1c19596f9d7/TestCropRoiImageProcessor_metadata.json'
results_CropRoiImageProcessor_m = lab_tools.pull_json_from_gitlab(results_url)
output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '42a4b5bc1dc5f0ec8aad0ac58abdb699/TestResizeImageProcessor_simple.bmp.0'
output_ResizeImageProcessor = lab_tools.pull_img_from_gitlab(output_url)


class TestCropRoiImageProcessor_metadata(unittest.TestCase):
    """ Test if ResizeImageProcessor is applied before. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = CropRoiImageProcessor(
            dyda_config_path=dyda_config_CropRoiImageProcessor)

        # run component
        comp.reset()
        comp.metadata = metadata_CropRoiImageProcessor
        comp.input_data = output_ResizeImageProcessor
        comp.run()

        # compare output_data with reference
        ref_data = output_CropRoiImageProcessor_m
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)

        # compare results with reference
        ref_data = results_CropRoiImageProcessor_m
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestCropRoiImageProcessor_simple(unittest.TestCase):
    """ Test simple case. """
 
    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = CropRoiImageProcessor(
            dyda_config_path=dyda_config_CropRoiImageProcessor)

        # run component
        comp.reset()
        comp.input_data = input_data
        comp.run()

        # compare output_data with reference
        ref_data = output_CropRoiImageProcessor
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)

        # compare results with reference
        ref_data = results_CropRoiImageProcessor
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestCropRoiImageProcessor_double(unittest.TestCase):
    """ Double test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = CropRoiImageProcessor(
            dyda_config_path=dyda_config_CropRoiImageProcessor)

        # run component
        comp.reset()
        comp.input_data = input_data
        comp.run()
        comp.input_data = input_data
        comp.run()

        # compare output_data with reference
        ref_data = output_CropRoiImageProcessor
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)

        # compare results with reference
        ref_data = results_CropRoiImageProcessor
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestCropRoiImageProcessor_list(unittest.TestCase):
    """ Test list of input. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = CropRoiImageProcessor(
            dyda_config_path=dyda_config_CropRoiImageProcessor)

        # run component
        comp.reset()
        comp.input_data = [input_data]
        comp.run()

        # compare output_data with reference
        ref_data = output_CropRoiImageProcessor
        tar_data = comp.output_data[0]
        img_diff = lab_tools.img_comparator(tar_data, ref_data)
        self.assertEqual(img_diff, 0.0)

        # compare results with reference
        ref_data = results_CropRoiImageProcessor
        tar_data = comp.results[0]
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
