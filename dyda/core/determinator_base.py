import copy
from dt42lab.core import lab_tools
from dyda.core import dyda_base


class DeterminatorBase(dyda_base.TrainerBase):
    """
    DeterminatorBase.input_data
        The first element of the list is a dictionary of a inferencer result.
    DeterminatorBase.results
        A dictionary of a inferencer result in which the detected objects
        are screened out by the determinator.
    """

    def __init__(self, dyda_config_path=''):
        """ Init function of DeterminatorBase

        """
        super(DeterminatorBase, self).__init__(
            dyda_config_path=dyda_config_path
        )

    def reset_output_data(self):
        """ Reset output_data, this should be defined in the base component."""
        self.output_data = []

    def reset_results(self):
        """ Reset results, this should be defined in the base component. """
        self.results = []

    def uniform_input(self, ori_input=None):
        """ Package input_data if it is not a list and lab_format check.

        """
        if ori_input is None:
            ori_input = self.input_data
        input_data = copy.deepcopy(ori_input)
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

    def uniform_results(self):
        """ Un-package results if they are packaged before.

        """
        if self.package:
            self.results = self.results[0]

    def set_re_assign_id(self):
        """ Set param re_assign_id.

        """
        if 're_assign_id' not in self.param.keys():
            self.re_assign_id = True
        else:
            self.re_assign_id = self.param['re_assign_id']

    def run_re_assign_id(self, annotations):
        """ Re-assign id of objects in annotations.

        """
        if self.re_assign_id:
            for i in range(len(annotations)):
                annotations[i]['id'] = i
        return annotations
