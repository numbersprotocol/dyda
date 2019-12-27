from dyda.core import dyda_base
from dyda_utils import data


class OutputGeneratorBase(dyda_base.TrainerBase):
    """
    OutputGeneratorBase.input_data
        Metadata needed to generate results.
    OutputGeneratorBase.results
        Results which is meet output format defined in specification.
    """

    def __init__(self, dyda_config_path=''):
        """ Init function of OutputGeneratorBase

        """
        super(OutputGeneratorBase, self).__init__(
            dyda_config_path=dyda_config_path
        )

    def reset_results(self):
        """ reset_results """
        self.results = {}

    def reset_output_data(self):
        """ Reset output_data """
        self.output_data = []
