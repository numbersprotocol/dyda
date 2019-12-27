""" Base class of data analyzer """
from dyda.core import dyda_base


class DataAnalyzerBase(dyda_base.TrainerBase):
    """
    DataAnalyzerBase.input_data
        List of data arrays
    ConverterBase.results
        List of dictionaries of results
    """

    def __init__(self, dyda_config_path=''):
        """ Init function of DataAnalyzerBase

        """
        super(DataAnalyzerBase, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.results = []

    def reset_results(self):
        self.results = []

    def reset_output_data(self):
        """ Reset output_data """
        self.output_data = []
