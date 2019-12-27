import os
import sys
import numpy as np
import traceback

from dyda.core import dyda_base
from dyda_utils import tools
from dyda_utils import data
from dyda_utils import image


class DataReaderBase(dyda_base.TrainerBase):

    def __init__(self, dyda_config_path=''):
        """ Init function of DataReaderBase """
        super(DataReaderBase, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.reset_results()

    def reset_output_data(self):
        """ Reset output_data """
        self.output_data = []

    def reset_results(self):
        self.results['data_path'] = []
        self.results['data_type'] = ''

    def read_data_from_metadata(self, metadata_input, encoding=None):
        """
        Get data using the information provided in metadata_input.
        The metadata can be a dictionary or the file path a json file. The
        metadata is required to include the following two keys
            * filename: the file name of the data
            * folder: the folder path contains the input data
            * data_type:  the type of the input data
                * image: RGB or gray image file

        """

        metadata = self.get_metadata(metadata_input, encoding=encoding)
        meta_keys = metadata.keys()
        if 'filename' not in meta_keys or 'folder' not in meta_keys:
            print('[dyda] Error: Cannot find filename ',
                  'and folder in metadata.')
            self.terminate_flag = True
        if 'data_type' not in meta_keys:
            print('[dyda] Error: Cannot find data_type in metadata.')
            self.terminate_flag = True

        fname = os.path.join(metadata['folder'], metadata['filename'])
        dtype = metadata['data_type']

        if not os.path.isfile(fname):
            print('[dyda] Error: %s is not a file.' % fname)
            self.terminate_flag = True

        self.read_data(fname, dtype)

    def read_data(self, input_data, dtype):
        """
        Get data from given input_data and dtype and change the
        base_name (the first element of metadata) to the base filename without
        extension if it is still empty

        @param input_data: can be path of the data or a numpy array
        @dtype: type of the data
            -- image: use opencv to read the data file
            -- array: assign input_data directly to DataReaderBase.

        """

        self.results['data_type'] = dtype
        opened_data = np.empty(shape=(0, 0))
        if dtype == 'image':
            opened_data = self.read_image(input_data)
            self.results['data_path'].append(input_data)
            base_name = tools.remove_extension(input_data,
                                               return_type='base-only')
            if self.metadata[0] == "":
                self.metadata[0] = base_name

        elif dtype == 'array':
            opened_data = input_data
        else:
            print('[dyda] Error: Type %s is not supported' % dtype)
            self.terminate_flag = True
        self.output_data.append(opened_data)

    def read_image(self, input_data):
        try:
            img_data = image.read_img(input_data, log=False)
            if img_data is None or img_data.shape[0] == 0:
                print('[dyda] Error: Cannot read %s' % input_data)
                self.terminate_flag = True
            return img_data
        except Exception:
            traceback.print_exc(file=sys.stdout)
            self.terminate_flag = True

    def get_metadata(self, metadata_input, encoding=None):
        """
        Read from metadata_input

        @param metadata_input: filepath of the dict of metadata

        Keyword arguments:
        encoding -- encoding of the json file

        """

        metadata = {}
        if isinstance(metadata_input, dict):
            metadata = metadata_input
            return metadata
        elif os.path.isfile(metadata_input):
            try:
                metadata = data.parse_json(metadata_input, encoding=encoding)
                return metadata
            except IOError:
                print('[dyda] Error: cannot open %s' % metadata_input)
                self.terminate_flag = True
                return {}
            except Exception:
                traceback.print_exc(file=sys.stdout)
                self.terminate_flag = True
                return {}
        else:
            self.terminate_flag = True
            print("%s is not a valid file" % metadata_input)
            return {}
