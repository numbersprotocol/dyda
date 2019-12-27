""" Base class of data analyzer """
import copy
from dyda.core import dyda_base
from dt42lab.core import lab_tools


class BoxProcessorBase(dyda_base.TrainerBase):
    """
    BoxProcessorBase.input_data
        Dict in lab format or list of dicts in lab format.
    ConverterBase.results
        Dict in lab format or list of dicts in lab format.
    """

    def __init__(self, dyda_config_path=''):
        """ Init function of BoxProcessorBase

        """
        super(BoxProcessorBase, self).__init__(
            dyda_config_path=dyda_config_path
        )

    def reset_output_data(self):
        """ Reset output_data """
        self.output_data = []

    def reset_results(self):
        """ Reset results """
        self.results = []

    def uniform_input(self, input_data):
        """ Package input_data if it is not a list and lab_format check.

        """
        input_data = copy.deepcopy(input_data)
        if not isinstance(input_data, list):
            input_data = [input_data]
            self.package = True
        else:
            self.package = False
        for data in input_data:
            if isinstance(data, list) and len(data) == 1:
                data = data[0]
            if not lab_tools.if_result_match_lab_format(data):
                self.terminate_flag = True
                self.logger.error("Support dict in lab format only")
        return input_data

    def uniform_output(self):
        """ Un-package output_data and results if they are packaged before.

        """
        if self.package:
            self.results = self.results[0]
            if not self.output_data == []:
                self.output_data = self.output_data[0]
