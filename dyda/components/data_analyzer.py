import math
import copy
import pandas as pd
from dyda_utils import lab_tools
from dyda.core import data_analyzer_base


class UncertaintyAnalyzerSimple(data_analyzer_base.DataAnalyzerBase):
    """ Simple uncertainty analyzer """

    def __init__(self, dyda_config_path=''):
        """ __init__ of UncertaintyAnalyzerSimple """

        super(UncertaintyAnalyzerSimple, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)
        self.results = []
        self.input_data = {}

    def main_process(self):
        """ define main_process of dyda component """

        error_square = 0.0
        if "uncertainties" in self.param.keys():
            for k, v in self.param["uncertainties"].items():
                if not self.check_uncertainty_value(v):
                    return False
                self.logger.debug('Calculating uncertainty: %s' % k)
                error_square = error_square + v * v
        else:
            self.param["uncertainties"] = {}

        if isinstance(self.input_data, dict):
            self.results = copy.deepcopy(
                {**self.input_data, **self.param["uncertainties"]}
            )
            self.results["error"] = math.sqrt(error_square)
        elif isinstance(self.input_data, list):
            for data_dict in self.input_data:
                self.results.append(
                    {**data_dict, **self.param["uncertainties"]}
                )
                self.results[-1]["error"] = math.sqrt(error_square)

        return True

    def check_uncertainty_value(self, error):
        """ Check if uncertainty value is within expected range """
        if error >= 1.0 or error < 0.0:
            self.terminate_flag = True
            self.logger.error("Uncertainty value is not correct")
            return False
        return True

    def reset_results(self):
        self.results = []


class StatAnalyzer(data_analyzer_base.DataAnalyzerBase):
    """ Simple statistics analyzer

        @param object_col: if the DataFrame contains column recording
                           event of whom, like "id", feed column name
                           in this parameter, and output will append
                           one column, "stat of", records the statistic
                           if of whom.

    """

    def __init__(self, dyda_config_path=''):
        """ __init__ of StatAnalyzer """

        super(StatAnalyzer, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

        if 'object_col' in self.param.keys():
            self.object_col = self.param['object_col']
        else:
            self.object_col = None

    def main_process(self):

        self.pack_input_as_list()

        # let input_data will always be list of list
        if not any(isinstance(i, list) for i in self.input_data):
            self.input_data = [self.input_data]

        for dfs in self.input_data:
            output = pd.DataFrame()

            for df in dfs:

                stat = df.describe()

                if self.object_col is not None:
                    uniques = str(pd.unique(df[self.object_col]))
                    stat.index = pd.MultiIndex.from_product(
                        [uniques, stat.index],
                        names=["stat of", None]
                    )

                output = output.append(stat)
            self.output_data.append(output)
        self.unpack_single_output()
