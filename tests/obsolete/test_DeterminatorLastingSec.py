import cv2
import unittest
from dt42lab.core import tools
from dt42lab.core import tinycv
from dt42lab.core import lab_tools
from dt42lab.utility import dict_comparator
from dyda.components.determinator import DeterminatorLastingSec


# pull test data from gitlab
input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    '987dd2321e3770967f452faefbb2e2fc/TestDeterminatorLastingSec_simple_input.json'
input_data = lab_tools.pull_json_from_gitlab(input_url)

output_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
    'e4bf8e156bd8804bf71021901e02951a/DeterminatorLastingSec_Simple_output.json'
output_DeterminatorLastingSec = lab_tools.pull_json_from_gitlab(output_url)

class TestDeterminatorLastingSec_simple(unittest.TestCase):
    """ Test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = DeterminatorLastingSec()

        # run component
        for i in range(len(input_data)):
            comp.reset()
            comp.input_data = input_data[i]
            comp.run()

            ## compare output_data with reference
            ref_data = output_DeterminatorLastingSec[i]
            tar_data = comp.results
            report = dict_comparator.get_diff(ref_data, tar_data)
            self.assertEqual(report['extra_field'], [])
            self.assertEqual(report['missing_field'], [])
            self.assertEqual(report['mismatch_val'], [])

if __name__ == '__main__':
    unittest.main()
