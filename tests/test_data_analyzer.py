import unittest
import os
import numpy as np
import pandas as pd
from dt42lab.core import lab_tools
from dt42lab.core import pandas_data
from dyda.components.data_analyzer import UncertaintyAnalyzerSimple
from dyda.components.data_analyzer import StatAnalyzer
from dyda.components.data_analyzer import FaceMatchAnalyzer
from dt42lab.utility import dict_comparator


class TestUncertaintyAnalyzerSimple(unittest.TestCase):
    def test_main_process(self):

        config_url = ('https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'
                      '07478873045e3731eb2336445f87e080/dyda.config')
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)
        output_url = ('https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'
                      'a9fb65dbb1684979b5b3f6f8bce1784b/'
                      'uncertainty_unit_test_results.json')

        # initialization
        analyzer_ = UncertaintyAnalyzerSimple(dyda_config)
        analyzer_.reset()
        analyzer_.run()

        ref_data = lab_tools.pull_json_from_gitlab(output_url)[0]
        tar_data = analyzer_.results[0]
        if not ref_data == [] and not tar_data == []:
            report = dict_comparator.get_diff(ref_data, tar_data)
            self.assertEqual(report['extra_field'], [])
            self.assertEqual(report['missing_field'], [])
            self.assertEqual(report['mismatch_val'], [])


# test data of TestAnalyzer
input_df = pd.DataFrame(index=np.arange(10),
                        columns=['virtual_id',
                                 'timestamps',
                                 'longitude',
                                 'latitude'])
input_df.loc[:, 'virtual_id'] = \
    '284a38ab19acfe79b5a88b02e1d82699829b0153656a099f48f4a4d77064fb92'
input_df.loc[:, 'timestamps'] = [1504079020649,
                                 1504107661539,
                                 1504108731950,
                                 1504109664983,
                                 1504144235032,
                                 1504145332035,
                                 1504146333094,
                                 1504152251840,
                                 1504153206049,
                                 1505180279414]
input_df.loc[:, 'longitude'] = [-122.031,
                                -122.073,
                                -122.19,
                                -122.332,
                                -122.217,
                                -122.065,
                                -122.061,
                                -122.185,
                                -122.185,
                                -122.33]
input_df.loc[:, 'latitude'] = [47.549,
                               47.553,
                               47.58,
                               47.6,
                               47.582,
                               47.544,
                               47.546,
                               47.614,
                               47.618,
                               47.601]


class TestStatAnalyzer(unittest.TestCase):
    def test_main_process(self):
        """ Test simple case. """

        ref_df = pd.DataFrame(index=['count',
                                     'mean',
                                     'std',
                                     'min',
                                     '25%',
                                     '50%',
                                     '75%',
                                     'max'],
                              columns=['timestamps',
                                       'longitude',
                                       'latitude'])
        ref_df.loc[:, 'timestamps'] = [10,
                                       1504232671658.5,
                                       333895199.67643,
                                       1504079020649,
                                       1504108965208.25,
                                       1504144783533.5,
                                       1504150772153.5,
                                       1505180279414]
        ref_df.loc[:, 'longitude'] = [10,
                                      -122.1669,
                                      0.108751960391013,
                                      -122.332,
                                      -122.21025,
                                      -122.185,
                                      -122.067,
                                      -122.031]
        ref_df.loc[:, 'latitude'] = [10,
                                     47.5787,
                                     0.029002107203137,
                                     47.544,
                                     47.55,
                                     47.581,
                                     47.60075,
                                     47.618]

        dyda_config_StatAnalyzer_simple = {'StatAnalyzer': {}}

        # initialization
        comp = StatAnalyzer(
            dyda_config_path=dyda_config_StatAnalyzer_simple
        )
        # run component
        comp.reset()
        comp.input_data = input_df
        comp.run()

        # compare output_data with reference
        ref_data = ref_df

        # why I do not use pandas_data.df_to_lab_anno.
        # the reason is pandas_data.df_to_lab_anno using
        # df.to_dict("records"), and it will discard index
        # in the next test case I need to compare index
        ref_data = ref_data.to_dict('index')
        tar_data = comp.output_data
        tar_data = tar_data.to_dict('index')

        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])

        """ Test case with specified object_col. """

        dyda_config_StatAnalyzer_object = {
            'StatAnalyzer': {"object_col": 'virtual_id'}
        }

        # initialization
        comp = StatAnalyzer(
            dyda_config_path=dyda_config_StatAnalyzer_object
        )
        # run component
        comp.reset()
        comp.input_data = input_df
        comp.run()

        # compare output_data with reference
        ref_data = ref_df.copy()
        uniques = \
            str(['284a38ab19acfe79b5a88b02e1d8'
                 '2699829b0153656a099f48f4a4d77064fb92'])
        ref_data.index = pd.MultiIndex.from_product(
            [uniques, ref_data.index],
            names=["stat of", None]
        )

        ref_data = ref_data.to_dict('index')
        tar_data = comp.output_data
        tar_data = tar_data.to_dict('index')

        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


# test data of FaceMatchAnalyzer
facematch_input = [np.array(
    [-1.25909820e-01, 6.19858466e-02, -1.92859992e-02, -8.77254531e-02,
     -8.18075910e-02, -9.61264819e-02, -3.27285752e-02, -1.61052763e-01,
     1.55191675e-01, -8.14432055e-02, 2.36405909e-01, -5.02960756e-02,
     -2.13651672e-01, -1.34547353e-01, -4.42342572e-02, 1.79538816e-01,
     -1.72321677e-01, -1.54231846e-01, -1.15180418e-01, -6.69394583e-02,
     7.16613606e-02, -2.18024850e-02, 7.73137659e-02, 5.52590340e-02,
     -1.08245209e-01, -3.83317739e-01, -9.79347453e-02, -7.16100186e-02,
     2.10064687e-02, -3.74283344e-02, -2.59360820e-02, 5.87740690e-02,
     -2.05119342e-01, -9.38989297e-02, 5.45350686e-02, 9.42733586e-02,
     1.77727677e-02, -7.32651353e-02, 2.13006124e-01, 1.85950659e-03,
     -2.16924682e-01, 2.31894664e-04, 4.53974195e-02, 2.17527717e-01,
     2.07627580e-01, 5.56880012e-02, 3.56101841e-02, -1.66266292e-01,
     1.33905590e-01, -1.21491335e-01, 4.36513200e-02, 1.55165642e-01,
     2.28043981e-02, 3.03663984e-02, 1.31373573e-02, -9.70270783e-02,
     8.28148723e-02, 1.43253505e-01, -1.47262797e-01, -2.20880099e-02,
     7.35357404e-02, -1.42083198e-01, -4.10361476e-02, -5.17379567e-02,
     2.15900987e-01, 5.54233976e-02, -9.09745842e-02, -1.54041752e-01,
     1.32771209e-01, -1.39044210e-01, -5.68357147e-02, 4.19111699e-02,
     -1.81505978e-01, -1.56515360e-01, -3.61369759e-01, -9.39996913e-04,
     4.22779292e-01, 9.82117727e-02, -1.81269825e-01, 4.32736874e-02,
     -5.30622602e-02, 1.80250127e-02, 1.82049334e-01, 1.54655471e-01,
     -5.79310618e-02, -5.40678576e-03, -1.33926541e-01, 7.20694661e-04,
     2.30842933e-01, -2.06722170e-02, -7.59157389e-02, 2.27963820e-01,
     -9.77697596e-03, 1.04884028e-01, 5.86406924e-02, 4.99408245e-02,
     -8.95563513e-02, 7.92465918e-03, -8.92178863e-02, -2.06362680e-02,
     9.66302603e-02, -1.85956359e-02, 2.56986171e-03, 1.14254534e-01,
     -1.21284872e-01, 4.16897424e-02, -2.89634187e-02, 6.18447736e-02,
     1.70928203e-02, -3.14667225e-02, -4.19338159e-02, -1.36174858e-01,
     8.52623656e-02, -1.76090956e-01, 1.85550734e-01, 2.22863287e-01,
     4.56090271e-02, 1.17231622e-01, 1.12823151e-01, 7.32913017e-02,
     -3.04848514e-02, 6.10425696e-03, -1.83370322e-01, 1.35350432e-02,
     8.88953358e-02, 2.08062753e-02, 1.24552161e-01, 9.97809693e-04])]


class TestFaceMatchAnalyzer(unittest.TestCase):
    def test_main_process(self):
        """ Test simple case. """

        dyda_config = {
            'FaceMatchAnalyzer': {
                "pickle_path": '/home/shared/DT42/test_data/'
                               'test_face_recognition/'
                               'George/encodings.pickle'}}

        # initialization
        comp = FaceMatchAnalyzer(
            dyda_config_path=dyda_config
        )
        # run component
        comp.reset()
        comp.input_data = facematch_input
        comp.run()

        # compare output_data with reference

        ref_data = [{'size': {'height': -1, 'width': -1},
                     'folder': '', 'filename': '',
                     'annotations': [{'labinfo': {}, 'bottom': -1,
                                      'confidence': -1, 'id': 0,
                                      'type': 'classification', 'right': -1,
                                      'left': 0, 'label': 'George', 'top': 0}]
                     }]
        tar_data = comp.results

        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])

        """ Test input is np.ndarray(not [np.ndarray]) case. """

        # initialization
        comp = FaceMatchAnalyzer(
            dyda_config_path=dyda_config
        )
        # run component
        comp.reset()
        comp.input_data = facematch_input[0]
        comp.run()

        # compare output_data with reference

        ref_data = {'size': {'height': -1, 'width': -1},
                    'folder': '', 'filename': '',
                    'annotations': [{'labinfo': {}, 'bottom': -1,
                                     'confidence': -1, 'id': 0,
                                     'type': 'classification', 'right': -1,
                                     'left': 0, 'label': 'George', 'top': 0}]
                    }
        tar_data = comp.results

        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])

        """ Test input is [[np.ndarray, np.ndarray], [np.ndarray]] case. """

        # initialization
        comp = FaceMatchAnalyzer(
            dyda_config_path=dyda_config
        )
        # run component
        comp.reset()
        comp.input_data = [[facematch_input[0], facematch_input[0]],
                           [facematch_input[0]]]
        comp.run()

        # compare output_data with reference

        ref_data = {'size': {'height': -1, 'width': -1},
                    'folder': '', 'filename': '',
                    'annotations': [{'labinfo': {}, 'bottom': -1,
                                     'confidence': -1, 'id': 0,
                                     'type': 'classification', 'right': -1,
                                     'left': 0, 'label': 'George', 'top': 0}]
                    }
        ref_datas = [[ref_data, ref_data], [ref_data]]
        tar_datas = comp.results

        for ref, tar in zip(ref_datas, tar_datas):
            report = dict_comparator.get_diff(ref, tar)
            self.assertEqual(report['extra_field'], [])
            self.assertEqual(report['missing_field'], [])
            self.assertEqual(report['mismatch_val'], [])

        """ Test case with specify tolerance. """

        dyda_config = {
            'FaceMatchAnalyzer': {
                "pickle_path": '/home/shared/DT42/test_data/'
                               'test_face_recognition/'
                               'George/encodings.pickle',
                "tolerance": 0}
        }

        # initialization
        comp_t = FaceMatchAnalyzer(
            dyda_config_path=dyda_config
        )
        # run component
        comp_t.reset()
        comp_t.input_data = facematch_input
        comp_t.run()

        # compare output_data with reference

        ref_data = [{'size': {'height': -1, 'width': -1},
                     'folder': '', 'filename': '',
                     'annotations': [{'labinfo': {}, 'bottom': -1,
                                      'confidence': -1, 'id': 0,
                                      'type': 'classification', 'right': -1,
                                      'left': 0, 'label': 'unknown', 'top': 0}]
                     }]
        tar_data = comp_t.results

        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
