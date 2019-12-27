import copy
from dt42lab.core import lab_tools
from dyda.core import dyda_base


class TrackerBase(dyda_base.TrainerBase):

    def __init__(self, dyda_config_path=''):
        """ __init__ of TrackerBase

        TrackerBase.input_metadata
            output from detector
        TrackerBase.output_metadata
            output from detector + track_id

        """

        super(TrackerBase, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.results = []

    def reset_results(self):
        """ reset TrackerBase.results to empty list """
        self.results = []

    def reset_output_data(self):
        """ Reset output_data """
        self.output_data = []

    def pre_process(self):
        """
        Pre-processing of data and results should happen here. This is
        called before the main_process.
        """

    def main_process(self):
        """
        main_process function will be called in the run function after
        pre_process and before post_process. The main logical computation
        of data should happen here.
        """
        pass

    def post_process(self):
        """
        post_process function will be called in the run function after
        main_process.
        """
        pass

    def uniform_input(self, input_data=[]):
        """ Package input_data if it is not a list and lab_format check.

        """
        if input_data == []:
            input_data = copy.deepcopy(self.input_data)
        else:
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
