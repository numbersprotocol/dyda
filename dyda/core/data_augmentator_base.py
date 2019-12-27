from dyda.core import dyda_base


class AugmentatorBase(dyda_base.TrainerBase):
    """
    AugmentatorBase.input_data
        an input item, an image, a dictionary, a list with only one item
    AugmentatorBase.results
        a list of augmented results
    """

    def __init__(self, dyda_config_path=''):
        """ Init function of AugmentatorBase

        """
        super(AugmentatorBase, self).__init__(
            dyda_config_path=dyda_config_path
        )

    def reset_results(self):
        """ reset_results for AugmentatorBase"""
        self.results = []

    def reset_output_data(self):
        """ Reset output_data """
        self.output_data = []

    def check_input_size(self):
        """ Check input size and unpack if it is packaged as a list """

        if isinstance(self.input_data, list):
            if len(self.input_data) > 1:
                if self.param["keep_first"]:
                    self.logger.warning(
                        "Length of input_data is larger than 1, only the"
                        " first component is proceeded to RT converter."
                    )
                    self.input_data = self.input_data[0]
                else:
                    self.logger.error(
                        "Augmentator does not allow input size > 1, exit"
                    )
                    self.terminate_flag = True
                    return False
            else:
                self.unpack_single_input()

        return True
