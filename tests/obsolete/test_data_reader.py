import os
import unittest
import numpy as np
import pandas as pd
from dyda_utils import data
from dyda_utils import tools
from dyda_utils import lab_tools
from dyda_utils import pandas_data
from dyda.components.data_reader import JsonReaderFix
from dyda.components.data_reader import Video2FrameReader
from dyda.components.data_reader import CsvDataReader
from dt42lab.utility import dict_comparator


class TestJsonReaderFix(unittest.TestCase):
    def test_main_process(self):

        # pull test data from gitlab
        config_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '611776ae5881ecb301eb0df91cafd4ae/JsonReaderFix.config'
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)

        output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
            '25e0372cec8b562c10e098d3295dd0fa/JsonReaderFix_output.json'
        output_list = lab_tools.pull_json_from_gitlab(output_url)

        # initialization
        unit_tested_ = JsonReaderFix(dyda_config_path=dyda_config)

        # run frame_selector
        unit_tested_.reset()
        unit_tested_.run()

        # compare results with reference
        ref_data = output_list
        tar_data = unit_tested_.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestVideo2FrameReader(unittest.TestCase):
    def test_main_process(self):
        testing_video_path = (
            "/home/shared/DT42/test_data/test_video_to_frame/test.avi"
        )
        if os.path.exists(testing_video_path):
            pass
        else:
            print("TestVideo2FrameReader: No testing data found in: %s,"
                  "this test is skipped." % testing_video_path)

        # pull test data from gitlab
        config_url = ('https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'
                      '45c24c8d19e9512e9ba5df79f9aef16a/'
                      'video2frame.dyda.config')
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)
        output_url = ('https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'
                      'f959eac0b515a1d977d50ffb3dd26988/video2frame.results')
        output_list = lab_tools.pull_json_from_gitlab(output_url)

        unit_tested_ = Video2FrameReader(dyda_config_path=dyda_config)

        unit_tested_.reset()
        unit_tested_.input_data = testing_video_path
        unit_tested_.run()

        # compare results with reference
        ref_data = output_list
        tar_data = unit_tested_.results
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])


class TestCsvDataReader(unittest.TestCase):
    def test_main_process(self):

        output_CsvDataReader = pd.DataFrame(index=np.arange(10),
                                            columns=['virtual_id',
                                                     'timestamps',
                                                     'longitude',
                                                     'latitude'])
        output_CsvDataReader.loc[:, 'virtual_id'] = \
            '284a38ab19acfe79b5a88b02e1d82699829b0153656a099f48f4a4d77064fb92'
        output_CsvDataReader.loc[:, 'timestamps'] = [1504079020649,
                                                     1504107661539,
                                                     1504108731950,
                                                     1504109664983,
                                                     1504144235032,
                                                     1504145332035,
                                                     1504146333094,
                                                     1504152251840,
                                                     1504153206049,
                                                     1505180279414]
        output_CsvDataReader.loc[:, 'longitude'] = [-122.031,
                                                    -122.073,
                                                    -122.19,
                                                    -122.332,
                                                    -122.217,
                                                    -122.065,
                                                    -122.061,
                                                    -122.185,
                                                    -122.185,
                                                    -122.33]
        output_CsvDataReader.loc[:, 'latitude'] = [47.549,
                                                   47.553,
                                                   47.58,
                                                   47.6,
                                                   47.582,
                                                   47.544,
                                                   47.546,
                                                   47.614,
                                                   47.618,
                                                   47.601]

        # in the present folder create csv for check
        CsvData_input_path = './test_CsvDataReader_temp.csv'
        output_CsvDataReader.to_csv(CsvData_input_path)

        # test simple case
        dyda_config_CsvDataReader_simple = {
            'CsvDataReader': {"index_col": 0}
        }

        # initialization
        comp = CsvDataReader(
            dyda_config_path=dyda_config_CsvDataReader_simple
        )
        # run component
        comp.reset()
        comp.input_data = CsvData_input_path
        comp.run()

        # compare output_data with reference
        ref_data = output_CsvDataReader
        ref_data = pandas_data.df_to_lab_anno(ref_data)
        tar_data = comp.output_data
        tar_data = pandas_data.df_to_lab_anno(tar_data)
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])

        # test case with specified thousands
        dyda_config_CsvDataReader_thousands = {
            'CsvDataReader': {"thousands": ',',
                              "index_col": 0}
        }

        # initialization
        comp = CsvDataReader(
            dyda_config_path=dyda_config_CsvDataReader_thousands
        )
        # run component
        comp.reset()
        comp.input_data = CsvData_input_path
        comp.run()

        # compare output_data with reference
        ref_data = output_CsvDataReader
        ref_data = pandas_data.df_to_lab_anno(ref_data)
        tar_data = comp.output_data
        tar_data = pandas_data.df_to_lab_anno(tar_data)
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])
        self.assertEqual(report['mismatch_val'], [])

        # remove the file created for check
        os.remove(CsvData_input_path)


if __name__ == '__main__':
    unittest.main()
