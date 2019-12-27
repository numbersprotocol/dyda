import os
from dyda.core import data_reader_base


class FrameReader(data_reader_base.DataReaderBase):

    def __init__(self, dyda_config_path=""):
        super(FrameReader, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.input_data = []

    def main_process(self):
        """ Main function called by the external code """

        if isinstance(self.input_data, list):
            for full_path in self.input_data:

                input_meta = {
                    'folder': os.path.dirname(full_path),
                    'filename': os.path.basename(full_path),
                    'data_type': 'image'
                }

                self.read_data_from_metadata(input_meta, "image")
        else:
            input_meta = {
                'folder': os.path.dirname(self.input_data),
                'filename': os.path.basename(self.input_data),
                'data_type': 'image'
            }

            self.read_data_from_metadata(input_meta, "image")


class FrameReaderFromDict(data_reader_base.DataReaderBase):

    def __init__(self, dyda_config_path="", param=None):
        """ Initialization function of dyda component. """

        super(FrameReaderFromDict, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        input_data = self.uniform_input(self.input_data, 'lab-format')

        for data in input_data:
            data['data_type'] = 'image'
            self.read_data_from_metadata(data, "image")

        self.uniform_output()
