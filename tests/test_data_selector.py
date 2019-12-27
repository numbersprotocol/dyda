import unittest
import os
import numpy as np
import pandas as pd
from dt42lab.core import tools
from dt42lab.core import lab_tools
from dt42lab.core import pandas_data
from dt42lab.utility import dict_comparator
from dyda.components.data_selector import RandomDataSelector
import json


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


class TestRandomDataSelector(unittest.TestCase):
    def test_main_process(self):

        np.random.seed(555)

        dyda_config_RandomDataSelector_no_split = {
            'RandomDataSelector': {"random_by": 'timestamps',
                                   "how_many": 5,
                                   "split": False}}

        """ Test no split case. """

        # initialization
        comp = RandomDataSelector(
            dyda_config_path=dyda_config_RandomDataSelector_no_split
        )
        comp.reset()
        comp.input_data = input_df
        comp.run()

        for df in comp.output_data:
            ref_data = input_df.loc[[0, 2, 3, 6, 7], :]
            ref_data = pandas_data.df_to_lab_anno(ref_data)
            tar_data = df
            tar_data = pandas_data.df_to_lab_anno(tar_data)

            report = dict_comparator.get_diff(ref_data, tar_data)
            self.assertEqual(report['extra_field'], [])
            self.assertEqual(report['missing_field'], [])
            self.assertEqual(report['mismatch_val'], [])

        """ Test split case. """

        dyda_config_RandomDataSelector_split = {
            'RandomDataSelector': {"random_by": 'timestamps',
                                   "how_many": 5,
                                   "split": True}}

        # initialization
        comp = RandomDataSelector(
            dyda_config_path=dyda_config_RandomDataSelector_split
        )
        comp.reset()
        comp.input_data = input_df
        comp.run()

        index_list = [1, 4, 8, 0, 3]

        for i, df in zip(index_list, comp.output_data):
            ref_data = input_df.loc[[i]]
            ref_data = pandas_data.df_to_lab_anno(ref_data)
            tar_data = df
            tar_data = pandas_data.df_to_lab_anno(tar_data)

            report = dict_comparator.get_diff(ref_data, tar_data)
            self.assertEqual(report['extra_field'], [])
            self.assertEqual(report['missing_field'], [])
            self.assertEqual(report['mismatch_val'], [])

        """ Test too many uniques request and no split case. """

        dyda_config_too_many_no_split = {
            'RandomDataSelector': {"random_by": 'timestamps',
                                   "how_many": 20,
                                   "split": False}}

        # initialization
        comp = RandomDataSelector(
            dyda_config_path=dyda_config_too_many_no_split)
        comp.reset()
        comp.input_data = input_df
        comp.run()

        for df in comp.output_data:
            ref_data = input_df.copy()
            ref_data = pandas_data.df_to_lab_anno(ref_data)
            tar_data = df
            tar_data = pandas_data.df_to_lab_anno(tar_data)

            report = dict_comparator.get_diff(ref_data, tar_data)
            self.assertEqual(report['extra_field'], [])
            self.assertEqual(report['missing_field'], [])
            self.assertEqual(report['mismatch_val'], [])

        """ Test too many uniques request and split case. """

        dyda_config_too_many_split = {
            'RandomDataSelector': {"random_by": 'timestamps',
                                   "how_many": 20,
                                   "split": True}}

        # initialization
        comp = RandomDataSelector(
            dyda_config_path=dyda_config_too_many_split)
        comp.reset()
        comp.input_data = input_df
        comp.run()

        index_list = np.arange(10)

        for i, df in zip(index_list, comp.output_data):
            ref_data = input_df.loc[[i]]
            ref_data = pandas_data.df_to_lab_anno(ref_data)
            tar_data = df
            tar_data = pandas_data.df_to_lab_anno(tar_data)

            report = dict_comparator.get_diff(ref_data, tar_data)
            self.assertEqual(report['extra_field'], [])
            self.assertEqual(report['missing_field'], [])
            self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
