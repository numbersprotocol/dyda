import unittest
import numpy as np
from dt42lab.core import tools
from dt42lab.core import lab_tools
from dt42lab.utility import dict_comparator
from dyda.components.output_generator import OutputGeneratorCombineDetCla

combinedetcla_input = [{'size': {'height': 480, 'width': 640}, 'filename': '',
                        'annotations': [{'id': 0, 'confidence': -1.0,
                                         'left': 319, 'right': 587, 'top': 82,
                                         'label': 'face', 'type': 'detection',
                                         'labinfo': {}, 'bottom': 350}],
                        'folder': ''},
                       [{'size': {'height': -1, 'width': -1},
                         'filename': '',
                         'annotations': [{'id': 0, 'confidence': -1,
                                          'left': 0, 'right': -1, 'top': 0,
                                          'label': 'George',
                                          'type': 'classification',
                                          'labinfo': {}, 'bottom': -1}],
                         'folder': ''}]
                       ]


class TestOutputGeneratorCombineDetCla(unittest.TestCase):
    def test_main_process(self):
        """ Test simple case. """

        dyda_config = {"OutputGeneratorCombineDetCla": {}}
        # initialization
        comp = OutputGeneratorCombineDetCla(dyda_config_path=dyda_config)
        # run component
        comp.reset()
        comp.input_data = combinedetcla_input
        comp.run()

        # compare results with reference
        ref_data = {'size': {'height': 480, 'width': 640},
                    'folder': '', 'filename': '',
                    'annotations': [{'type': 'classification', 'bottom': 350,
                                     'top': 82, 'left': 319, 'right': 587,
                                     'labinfo': {}, 'label': 'George',
                                     'confidence': -1, 'id': 0}]
                    }
        tar_data = comp.results

        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])

        """ Test append case. """
        dyda_config_append = {
            "OutputGeneratorCombineDetCla": {
                "behavior": "append",
                "cla_key_name": {
                    "label": "person"}}}
        # initialization
        comp = OutputGeneratorCombineDetCla(
            dyda_config_path=dyda_config_append)
        # run component
        comp.reset()
        comp.input_data = combinedetcla_input
        comp.run()

        # compare results with reference
        ref_data = {'size': {'height': 480,
                             'width': 640},
                    'folder': '',
                    'filename': '',
                    'annotations': [{'type': 'detection',
                                     'person': 'George',
                                     'top': 82,
                                     'right': 587,
                                     'labinfo': {},
                                     'label': 'face',
                                     'cla_labinfo': {},
                                     'cla_type': 'classification',
                                     'bottom': 350,
                                     'left': 319,
                                     'confidence': -1.0,
                                     'id': 0,
                                     'cla_confidence': -1}]}
        tar_data = comp.results

        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
