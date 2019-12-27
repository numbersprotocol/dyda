import numpy as np
from dyda.core import data_preprocessor_base


class DataNormalizerSimple(data_preprocessor_base.DataPreProcessor):
    """ Simple data normalizer, divide every element by divide_fact """

    def __init__(self, dyda_config_path=''):
        """ __init__ of DataNormalizerSimple """

        super(DataNormalizerSimple, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)
        self.divide_fact = 1.0
        if "divide_fact" in self.param.keys():
            self.divide_fact = self.param["divide_fact"]
        self.use_previous_ncount = False
        if "use_previous_ncount" in self.param.keys():
            self.use_previous_ncount = self.param["use_previous_ncount"]
        if self.use_previous_ncount:
            self.logger.warning(
                "use_previous_ncount is True, divide_fact will be ignored"
            )

    def main_process(self):
        """ define main_process of dyda component """

        input_data = self.uniform_input()

        if self.use_previous_ncount:
            try:
                self.divide_fact = self.metadata[-1]["results"]["ncount"]
            except KeyError:
                self.logger.error(
                    "Cannot find ncount from the results of previous component"
                )
                self.terminate_flag = True

        for data_array in input_data:
            # avoid ZeroDivisionError
            # if self.divide_fact is zero, append the original matrix
            if self.divide_fact == 0:
                self.output_data.append(data_array)
            else:
                out_data = np.true_divide(data_array, self.divide_fact)
                self.output_data.append(out_data)

        self.uniform_output()
        return True
