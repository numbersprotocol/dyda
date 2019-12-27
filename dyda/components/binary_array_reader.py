import os
from dyda.core import data_reader_base


class BinaryDataReader(data_reader_base.DataReaderBase):

    def __init__(self, dyda_config_path=""):
        super(BinaryDataReader, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.input_data = []

    def main_process(self):
        """ Main function called by the external code """

        self.results['data_path'].append('DIRECT_BINARY_INPUT')
        self.results['data_type'] = 'array'
        #Do nothing, assign input_data to output_data
        self.output_data = self.input_data
