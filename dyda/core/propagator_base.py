import os
from dyda_utils import tools
from dyda_utils import data
from dyda.core import dyda_base


class PropagatorBase(dyda_base.TrainerBase):
    """
    PropagatorBase.input_data
        A list of dictionary. The first element is a frame selector result.
        The second element is a inferencer result.
    Propagator.results
        A dictionary of a inferencer result propagated from reference data
        to base data.
    """

    def __init__(self, dyda_config_path=''):
        """ Init function of PropagatorBase

        """
        super(PropagatorBase, self).__init__(
            dyda_config_path=dyda_config_path
        )

    def reset_output_data(self):
        """ Reset output_data """
        self.output_data = []

    def reset_results(self):
        """ Reset results """
        pass
