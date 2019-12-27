from dyda.core import dyda_base
from dt42lab.core import data


class ConverterBase(dyda_base.TrainerBase):
    """
    ConverterBase.input_data
        The first element of the list could be a dictionary
        of a inferencer result or a numpy array.
    ConverterBase.results
        The same data type of the input_data after converted.
    """

    def __init__(self, dyda_config_path=''):
        """ Init function of ConverterBase

        """
        super(ConverterBase, self).__init__(
            dyda_config_path=dyda_config_path
        )

    def reset_output_data(self):
        """ Reset output_data """
        self.output_data = []

    def reset_results(self):
        """ Reset results, this should be defined in the base component. """
        pass
