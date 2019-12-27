from dyda.core import dyda_base


class ExtractorBase(dyda_base.TrainerBase):
    """
    ExtractorBase.input_data
        a list of input_data to be extracted
    ExtractorBase.results
        a list extracted results
    """

    def __init__(self, dyda_config_path=''):
        """ Init function of ExtractorBase

        """
        super(ExtractorBase, self).__init__(
            dyda_config_path=dyda_config_path
        )

    def reset_results(self):
        """ reset_results for ExtractorBase"""
        self.results = []

    def reset_output_data(self):
        """ Reset output_data """
        self.output_data = []
