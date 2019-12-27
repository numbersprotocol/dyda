import unittest
import warnings
import cv2
import numpy as np
import pandas as pd
from dyda_utils import tools
from dyda_utils import lab_tools
from dt42lab.utility import dict_comparator
from dyda.components.data_plotter import DataFrameHistPlotter
from dyda.components.data_plotter import LocationBubblePlotter
warnings.filterwarnings("ignore")


test_df = pd.DataFrame(index=np.arange(9),
                       columns=['virtual_id',
                                'timestamps',
                                'longitude',
                                'latitude'])
test_df.loc[:, 'virtual_id'] = \
    '284a38ab19acfe79b5a88b02e1d82699829b0153656a099f48f4a4d77064fb92'
test_df.loc[:, 'timestamps'] = [1504079020649,
                                1504107661539,
                                1504108731950,
                                1504109664983,
                                1504144235032,
                                1504145332035,
                                1504146333094,
                                1504152251840,
                                1504153206049]
test_df.loc[:, 'longitude'] = [-122.031,
                               -122.073,
                               -122.19,
                               -122.332,
                               -122.217,
                               -122.065,
                               -122.061,
                               -122.185,
                               -122.185]
test_df.loc[:, 'latitude'] = [47.549,
                              47.553,
                              47.58,
                              47.6,
                              47.582,
                              47.544,
                              47.546,
                              47.614,
                              47.618]


class TestDataFrameHistPlotter(unittest.TestCase):
    def test_main_process(self):
        """ Main process of unit test. """

        dyda_config_DataFrameHistPlotter = {
            'DataFrameHistPlotter': {
                'hist_feature': ['longitude', 'latitude']}
        }
        output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/'\
                     'uploads/23e072aff6360991f5c004b475aaf7f2/hist.bmp'
        # initialization
        comp = DataFrameHistPlotter(
            dyda_config_path=dyda_config_DataFrameHistPlotter
        )

        # run component
        comp.reset()
        comp.input_data = test_df
        comp.run()
        # compare output_data with reference
        ref_data = lab_tools.pull_img_from_gitlab(output_url)
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(ref_data, tar_data)
        self.assertEqual(img_diff, 0.0)

        # test the case with using all parameters
        dyda_config_DataFrameHistPlotter = {
            'DataFrameHistPlotter': {
                'hist_feature': ['longitude', 'latitude'],
                'range': [(-122.5, -121.5), (47.5, 47.7)],
                'bins': 20,
                'belongs_to': 'virtual_id'}
        }
        output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
                     'c7667cbeae10c6cc41defcae2c6bafe1/hist_complicated.bmp'
        # initialization
        comp = DataFrameHistPlotter(
            dyda_config_path=dyda_config_DataFrameHistPlotter
        )

        # run component
        comp.reset()
        comp.input_data = test_df
        comp.run()
        # compare output_data with reference
        ref_data = lab_tools.pull_img_from_gitlab(output_url)
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(ref_data, tar_data)
        self.assertEqual(img_diff, 0.0)


class TestLocationBubblePlotter(unittest.TestCase):
    def test_main_process(self):
        """ Main process of unit test. """

        # test case with plotting arrow
        dyda_config_LocationBubblePlotter = {
            'LocationBubblePlotter': {"belongs_to": "virtual_id",
                                      "plot_arrow": True}
        }

        output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/'\
                     'uploads/ee7e62a737ab364cf96bc29dfaf55425/bp_arrow.bmp'

        # initialization
        comp = LocationBubblePlotter(
            dyda_config_path=dyda_config_LocationBubblePlotter
        )
        comp.reset()
        comp.input_data = test_df
        comp.run()

        # compare output_data with reference
        ref_data = lab_tools.pull_img_from_gitlab(output_url)
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(ref_data, tar_data)
        self.assertEqual(img_diff, 0.0)

        # test case without plotting arrow
        dyda_config_LocationBubblePlotter = {
            'LocationBubblePlotter': {"belongs_to": "virtual_id",
                                      "plot_arrow": False}
        }

        output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/'\
                     'uploads/4b592ecb09944b7d9412db84a233f166/bp.bmp'

        # initialization
        comp = LocationBubblePlotter(
            dyda_config_path=dyda_config_LocationBubblePlotter
        )
        comp.reset()
        comp.input_data = test_df
        comp.run()

        # compare output_data with reference
        ref_data = lab_tools.pull_img_from_gitlab(output_url)
        tar_data = comp.output_data
        img_diff = lab_tools.img_comparator(ref_data, tar_data)
        self.assertEqual(img_diff, 0.0)


if __name__ == '__main__':
    unittest.main()
