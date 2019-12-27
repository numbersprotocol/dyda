import unittest
from dyda.components.data_balancer import DataBalancerSimple


class TestUncertaintyAnalyzerSimple(unittest.TestCase):
    def test_main_process(self):

        in_list_2 = ['ok', 'ok', 'ok', 'ng', 'ng', 'ng', 'ng', 'ok', 'ok']
        in_list_1 = [i for i in range(0, len(in_list_2))]

        balancer = DataBalancerSimple()
        balancer.input_data = [in_list_1, in_list_2]
        balancer.run()
        for i in range(0, len(balancer.results[0])):
            index = balancer.results[0][i]
            label = balancer.results[1][i]
            original_label = in_list_2[index]
            self.assertEqual(original_label, label)
        len_ok = balancer.results[1].count('ok')
        len_ng = balancer.results[1].count('ng')
        self.assertEqual(len_ok, 4)
        self.assertEqual(len_ng, 4)


if __name__ == '__main__':
    unittest.main()
