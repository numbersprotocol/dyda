import os
import unittest
from dt42lab.core import data
from dt42lab.core import tools
from dt42lab.core import lab_tools
from dyda.components.tracker import TrackerByOverlapRatio
from dyda.components.tracker import TrackerSimple
from dt42lab.utility import dict_comparator


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


#class TestTrackerByOverlapRatio(unittest.TestCase):
#    def test_main_process(self):
#
#        # pull test data from gitlab
#        input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
#            'a3240e1039983da353a7c6542bb8735f/input_list.json'
#        input_list = lab_tools.pull_json_from_gitlab(input_url)["data"]
#        output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
#            '9f118c36de077b56e2a05e55dc145800/tracker_output_list.json'
#        output_list = lab_tools.pull_json_from_gitlab(output_url)
#
#        # initialization
#        tracker_ = TrackerByOverlapRatio()
#
#        for i in range(len(input_list)):
#
#            # run tracker
#            tracker_.reset()
#            tracker_.metadata[0] = tools.remove_extension(
#                input_list[i]['filename'],
#                'base-only')
#            tracker_.input_data.append([input_list[i]])
#            tracker_.run()
#
#            # compare results with reference
#            if not tracker_.results == []:
#                ref_data = output_list[i]
#                tar_data = tracker_.results
#                report = dict_comparator.get_diff(ref_data, tar_data)
#                self.assertEqual(report['extra_field'], [])
#                self.assertEqual(report['missing_field'], [])
#                self.assertEqual(report['mismatch_val'], [])



if __name__ == '__main__':
    unittest.main()

