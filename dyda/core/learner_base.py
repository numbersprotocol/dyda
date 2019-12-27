from dyda.core import dyda_base


class LearnerBase(dyda_base.TrainerBase):
    """ Base class of learner """

    def __init__(self, dyda_config_path=''):
        """ Init function of LearnerBase """
        super(LearnerBase, self).__init__(
            dyda_config_path=dyda_config_path
        )
        # Note: results is a diction with information of the algorithm
        #       which is ready to be used for validation
        self.results = {}

    def reset_results(self):
        self.results = {}

    def reset_output_data(self):
        """ Reset output_data """
        self.output_data = []
