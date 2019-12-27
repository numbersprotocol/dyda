import math
from dyda.core import dyda_base


class ValidatorBase(dyda_base.TrainerBase):
    """ Base class of learner """

    def __init__(self, dyda_config_path=''):
        """ Init function of ValidatorBase """
        super(ValidatorBase, self).__init__(
            dyda_config_path=dyda_config_path
        )
        # Note: results is a diction with information of the algorithm
        #       which is ready to be used for validation
        self.reset_results()

    def reset_results(self):
        """ Reset results """
        self.results = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "nsamples": 0,
            "stat_error": 0.0,
            "lab_info": {}
        }

    def reset_output_data(self):
        """ Reset output_data """
        self.output_data = []

    def cal_stat_error(self, prob, N):
        """ Calculate error for Binomial distribution """

        if prob < 0 or prob > 1:
            self.logger.error("Probablity value %.2f is not valid" % prob)
            return -1
        return math.sqrt((prob*(1-prob))/N)
