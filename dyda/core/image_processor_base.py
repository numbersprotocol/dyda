""" Base class of data analyzer """
import copy
import numpy as np
from dyda.core import dyda_base


class ImageProcessorBase(dyda_base.TrainerBase):
    """
    ImageProcessorBase.input_data
        A np.ndarray or list of np.ndarray.
    ImageProcessorBase.output_data
        A np.ndarray array or list of np.ndarray.
    """

    def __init__(self, dyda_config_path=''):
        """ Init function of ImageProcessorBase

        """
        super(ImageProcessorBase, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.results = []

    def reset_results(self):
        """ Reset results """
        self.results = []

    def reset_output_data(self):
        """ Reset output_data """
        self.output_data = []

    def uniform_input(self):
        """ Package input_data if it is not a list.

        """
        input_data = copy.deepcopy(self.input_data)
        if not isinstance(input_data, list):
            input_data = [input_data]
            self.package = True
        else:
            self.package = False
        for data in input_data:
            if not isinstance(data, np.ndarray):
                self.terminate_flag = True
                self.logger.error(" supports np.ndarray "\
                    "or list of np.ndarray as input_data only")
        return input_data

    def uniform_output(self):
        """ Un-package output_data and results if they are packaged before.

        """
        if self.package:
            self.unpack_single_output()
            self.unpack_single_results()
