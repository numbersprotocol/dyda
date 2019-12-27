import os
import unittest
from dyda_utils import data
from dyda_utils import tools
from dyda_utils import lab_tools
from dyda.components.tracker import TrackerSimple
from dyda_utils import dict_comparator


class TestTrackerSimple(unittest.TestCase):
    """ Test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        dyda_config_TrackerSimple = {"TrackerSimple":{
            'matching_thre': 1000,
            'max_missing_frame': 50
        }}
        input_data = tools.parse_json('/home/shared/DT42/test_data/test_object_detection_and_tracking/P2_ref_output.json')
        # initialization
        comp = TrackerSimple(
            dyda_config_path=dyda_config_TrackerSimple)


        # run component
        for data in input_data:
            comp.reset()
            comp.input_data = data
            comp.run()


if __name__ == '__main__':
    unittest.main()

