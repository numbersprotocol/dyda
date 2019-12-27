from dyda.core import dyda_base


class AccumulatorBase(dyda_base.TrainerBase):
    """
    AccumulatorBase.input_data
        one component, one image, one dict or a list of components
    AccumulatorBase.output_data
        accumulated data
    AccumulatorBase.results
        {"ncount": $NUMBER_OF_ACCUMULATED_Results}
    """

    def __init__(self, dyda_config_path=''):
        """ Init function of AccumulatorBase

        """
        super(AccumulatorBase, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.results = {"ncount": 0}

    def reset_output_data(self):
        """ Reset output_data """
        pass

    def reset_results(self):
        """ Reset results, this should be defined in the base component. """
        pass
