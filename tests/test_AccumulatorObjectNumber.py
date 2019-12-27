import cv2
import unittest
from dt42lab.core import tools
from dt42lab.core import tinycv
from dt42lab.core import lab_tools
from dt42lab.utility import dict_comparator
from dyda.components.accumulator import AccumulatorObjectNumber
import json


input_data = [{'folder': '',
               'annotations': [{'confidence': 0.5132837891578674,
                                'top': 6,
                                'type': 'detection',
                                'label': 'car',
                                'left': 208,
                                'id': 0,
                                'right': 434,
                                'track_id': 8,
                                'bottom': 151},
                               {'confidence': 0.7307100892066956,
                                'top': 78,
                                'type': 'detection',
                                'label': 'car',
                                'left': 0,
                                'id': 1,
                                'right': 134,
                                'track_id': 2,
                                'bottom': 678},
                               {'confidence': 0.8351224660873413,
                                'top': 11,
                                'type': 'detection',
                                'label': 'car',
                                'left': 366,
                                'id': 2,
                                'right': 935,
                                'track_id': 0,
                                'bottom': 301}],
               'filename': '00000352',
               'size': {'height': 720,
                        'width': 1280}},
              {'folder': '',
               'annotations': [{'confidence': 0.3328424096107483,
                                'top': 3,
                                'type': 'detection',
                                'label': 'bus',
                                'left': 56,
                                'id': 0,
                                'right': 475,
                                'track_id': 10,
                                'bottom': 155},
                               {'confidence': 0.4029218554496765,
                                'top': 11,
                                'type': 'detection',
                                'label': 'car',
                                'left': 222,
                                'id': 1,
                                'right': 441,
                                'track_id': 8,
                                'bottom': 152},
                               {'confidence': 0.4060383439064026,
                                'top': 112,
                                'type': 'detection',
                                'label': 'car',
                                'left': 0,
                                'id': 2,
                                'right': 61,
                                'track_id': 11,
                                'bottom': 604},
                               {'confidence': 0.5273750424385071,
                                'top': 12,
                                'type': 'detection',
                                'label': 'car',
                                'left': 4,
                                'id': 3,
                                'right': 98,
                                'track_id': 2,
                                'bottom': 719},
                               {'confidence': 0.8162580132484436,
                                'top': 14,
                                'type': 'detection',
                                'label': 'car',
                                'left': 363,
                                'id': 4,
                                'right': 937,
                                'track_id': 0,
                                'bottom': 300}],
               'filename': '00000353',
               'size': {'height': 720,
                        'width': 1280}},
              {'folder': '',
               'annotations': [{'confidence': 0.3305732011795044,
                                'top': 5,
                                'type': 'detection',
                                'label': 'bus',
                                'left': 64,
                                'id': 0,
                                'right': 464,
                                'track_id': 10,
                                'bottom': 151},
                               {'confidence': 0.39185482263565063,
                                'top': 11,
                                'type': 'detection',
                                'label': 'car',
                                'left': 226,
                                'id': 1,
                                'right': 444,
                                'track_id': 8,
                                'bottom': 149},
                               {'confidence': 0.791477382183075,
                                'top': 16,
                                'type': 'detection',
                                'label': 'car',
                                'left': 360,
                                'id': 2,
                                'right': 941,
                                'track_id': 0,
                                'bottom': 299}],
               'filename': '00000354',
               'size': {'height': 720,
                        'width': 1280}}]

dyda_config_AccumulatorObjectNumber = {
    'AccumulatorObjectNumber': {"reset_frame_num": 2}}

output_AccumulatorObjectNumber_N_equals_two = [
    {
        'object_counting': {
            'car': 3}}, {
                'object_counting': {
                    'car': 4, 'bus': 1}}, {
                        'object_counting': {
                            'car': 2, 'bus': 1}}]


class TestAccumulatorObjectNumber_N_equals_two(unittest.TestCase):
    """ Test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = AccumulatorObjectNumber(
            dyda_config_path=dyda_config_AccumulatorObjectNumber
        )

        # run component
        for idx, data in enumerate(input_data):
            comp.reset()
            comp.input_data = data
            comp.run()

            # compare output_data with reference
            ref_data = output_AccumulatorObjectNumber_N_equals_two[idx]
            tar_data = comp.results
            report = dict_comparator.get_diff(ref_data, tar_data)
            self.assertEqual(report['extra_field'], [])
            self.assertEqual(report['missing_field'], [])
            self.assertEqual(report['mismatch_val'], [])


dyda_config_AccumulatorObjectNumber_default = {
    'AccumulatorObjectNumber': {"reset_frame_num": -1}}

output_AccumulatorObjectNumber_default = [
    {
        'object_counting': {
            'car': 3}}, {
                'object_counting': {
                    'car': 4, 'bus': 1}}, {
                        'object_counting': {
                            'car': 4, 'bus': 1}}]


class TestAccumulatorObjectNumber_default(unittest.TestCase):
    """ Test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = AccumulatorObjectNumber(
            dyda_config_path=dyda_config_AccumulatorObjectNumber_default
        )

        # run component
        for idx, data in enumerate(input_data):
            comp.reset()
            comp.input_data = data
            comp.run()
            # compare output_data with reference
            ref_data = output_AccumulatorObjectNumber_default[idx]
            tar_data = comp.results
            report = dict_comparator.get_diff(ref_data, tar_data)
            self.assertEqual(report['extra_field'], [])
            self.assertEqual(report['missing_field'], [])
            self.assertEqual(report['mismatch_val'], [])


dyda_config_AccumulatorObjectNumber_dict_list = {
    'AccumulatorObjectNumber': {"reset_frame_num": -1}}

output_AccumulatorObjectNumber_dict_list = [
    {
        'object_counting': {
            'car': 3}}, {
                'object_counting': {
                    'car': 4, 'bus': 1}}, {
                        'object_counting': {
                            'car': 2, 'bus': 1}}]


class TestAccumulatorObjectNumber_dict_list(unittest.TestCase):
    """ Test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = AccumulatorObjectNumber(
            dyda_config_path=dyda_config_AccumulatorObjectNumber_default
        )

        # run component

        comp.reset()
        comp.input_data = input_data
        comp.run()
        # compare output_data with reference
        ref_data = output_AccumulatorObjectNumber_dict_list
        tar_data = comp.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
