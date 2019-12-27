import cv2
import copy
import numpy as np
from skimage import measure
from dyda_utils import image
from dyda.core import data_selector_base


class FrameSelectorDownsampleMedian(data_selector_base.DataSelectorBase):
    """The median data during an interval is selected and assigned as
       reference data to others not be selected.

       @param interval: during the interval only one data selected.
    """

    def __init__(self, dyda_config_path="", param={'interval': 10}):

        super(FrameSelectorDownsampleMedian, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.data = {
            'base_name': '',
            'is_key': False,
            'ref_data_name': ''}
        self.data_list = []
        self.counter = 0
        self.ref_index = int(self.param['interval'] / 2) - 1

    def main_process(self):

        # update data
        data = copy.deepcopy(self.data)
        data['base_name'] = self.metadata[0]
        if self.counter < self.ref_index:
            data['is_key'] = False
        elif self.counter == self.ref_index:
            data['is_key'] = True
            self.ref_data_name = self.metadata[0]
            data['ref_data_name'] = self.ref_data_name
            for i in range(self.ref_index):
                self.data_list[i * -1]['ref_data_name'] = self.ref_data_name
        else:
            data['is_key'] = False
            data['ref_data_name'] = self.ref_data_name

        # update data_list
        self.data_list.append(data)

        # update results
        if len(self.data_list) > self.ref_index:
            self.results = self.data_list[0]
            self.data_list.pop(0)
        else:
            self.results = self.data

        # update counter
        self.counter += 1
        if self.counter == self.param['interval']:
            self.counter = 0

        # update output_data
        self.output_data = self.data_list[-1]['is_key']


class FrameSelectorDownsampleFirst(data_selector_base.DataSelectorBase):
    """The first data during an interval is selected and assigned as
       reference data to others not be selected.

       @param interval: during the interval only one data selected.
    """

    def __init__(self, dyda_config_path="", param=None):

        super(FrameSelectorDownsampleFirst, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__

        self.set_param(class_name, param=param)

        self.counter = 0

    def main_process(self):

        # update results
        self.results['base_name'] = self.metadata[0]
        if self.counter == 0:
            self.results['is_key'] = True
            self.ref_data_name = self.metadata[0]
            self.results['ref_data_name'] = self.ref_data_name
        else:
            self.results['is_key'] = False
            self.results['ref_data_name'] = self.ref_data_name

        # update counter
        self.counter += 1
        if self.counter == self.param['interval']:
            self.counter = 0

        # update output_data
        self.output_data = self.results['is_key']


class FrameSelectorSsimFirst(data_selector_base.DataSelectorBase):
    """The data are grouped according to ssim. The first data in a group
       is selected and assigned as reference data to others not be selected.

       @param threshold: threshold for ssim, the higher the less frame grouped.
       @param length: frames are resized to [length, length] to reduce
                      computation time.
    """

    def __init__(self, dyda_config_path="", param={
            'threshold': 0.95, 'length': 100}):

        super(FrameSelectorSsimFirst, self).__init__(
            dyda_config_path=dyda_config_path
        )

        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.threshold = self.param['threshold']
        self.length = self.param['length']

        self.ref_data = np.empty(shape=(0, 0))

    def main_process(self):

        image = self.input_data[0]

        # update results
        if self.ref_data.size == 0:
            self.results['is_key'] = True
            self.ref_data_name = self.metadata[0]
            self.ref_data = image.resize_img(
                image,
                (self.length, self.length))
        else:
            resize_data = cv2.resize(
                image,
                (self.length, self.length))
            s = measure.compare_ssim(
                resize_data,
                self.ref_data,
                multichannel=True)
            if s > self.threshold:
                self.results['is_key'] = False
            else:
                self.results['is_key'] = True
                self.ref_data_name = self.metadata[0]
                self.ref_data = resize_data
        self.results['ref_data_name'] = self.ref_data_name


class FrameSelectorChannelCycling(data_selector_base.DataSelectorBase):
    """Same as FrameSelectorInRotation but also select metadata
       input_data[0]: list of frames from multiple channels
       input_data[1]: list of results or external_meta from multiple channels
       output_data: selected frame
       results: selected result/external_meta
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(FrameSelectorChannelCycling, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.counter = 0
        self.cycling_max = 0

        self.read_meta = True
        if "read_meta" in self.param.keys():
            self.read_meta = self.param["read_meta"]

    def main_process(self):
        """ Main function of dyda component. """

        if self.read_meta:
            if len(self.input_data[0]) != len(self.input_data[1]):
                self.terminate_flag = True
                self.logger.error(
                    "Different input_data and meta size detected"
                )
                return False

            self.cycling_max = max(self.cycling_max, len(self.input_data[0]))
            self.output_data = self.input_data[0][self.counter]
            self.results = self.input_data[1][self.counter]

        else:
            self.cycling_max = max(self.cycling_max, len(self.input_data))
            self.output_data = self.input_data[self.counter]
            self.results = {'channel_index': self.counter}

        self.counter += 1
        if self.counter == self.cycling_max:
            self.counter = 0


class FrameSelectorInRotation(data_selector_base.DataSelectorBase):
    """The images in self.input_data are output in rotation.
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(FrameSelectorInRotation, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.counter = 0

    def main_process(self):
        """ Main function of dyda component. """

        input_data = self.uniform_input(self.input_data, 'ndarray')
        if len(input_data) != self.param['rotation_num']:
            self.terminate_flag = True
            self.logger.error("Length of input_data should match "
                              "param rotation_num")

        self.output_data = input_data[self.counter]
        self.results = {'channel_index': self.counter}

        self.counter += 1
        if self.counter == self.param['rotation_num']:
            self.counter = 0
