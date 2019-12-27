from dyda.core import dyda_base


class ClassifierBase(dyda_base.TrainerBase):
    """ Base class of classifier """

    def __init__(self, dyda_config_path=''):
        """ Init function of ClassifierBase """
        super(ClassifierBase, self).__init__(
            dyda_config_path=dyda_config_path
        )
        # Note: results is a list of output from
        #       lab_tools.output_pred_classification
        self.results = []

    def reset_results(self):
        self.results = []

    def reset_output_data(self):
        """ Reset output_data """
        self.output_data = []
