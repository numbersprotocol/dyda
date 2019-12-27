import copy
from dyda.core import data_extractor_base


class JsonFieldExtractor(data_extractor_base.ExtractorBase):
    """ Get labels from data path """

    def __init__(self, dyda_config_path='', param=None):
        super(JsonFieldExtractor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.output_data = []
        self.results = []

    def main_process(self):
        """ main_process of JsonFieldExtractor """

        key_series = ["annotations", 0, "label"]
        if "extract_key_series" in self.param.keys():
            key_series = self.param["extract_key_series"]
        for ori_data in self.input_data:
            data = copy.deepcopy(ori_data)
            for key in key_series:
                try:
                    data = data[key]
                except KeyError:
                    self.terminate_flag = True
                    self.logger.error(
                        "cannot find key %s in input data" % str(key)
                    )
                    return False
            self.results.append(copy.deepcopy(data))

    def reset_results(self):
        """ reset_results for JsonFieldExtractor"""
        self.results = []


class TargetDataExtractor(data_extractor_base.ExtractorBase):
    """ Extract target data based on the input label list
        input_data[0]: a list of data to be extracted from
        input_data[1]: a list of reference label
        output_data: a dictionary of categoried data
    """

    def __init__(self, dyda_config_path='', param=None):
        """ init function of TargetDataExtractor """

        super(TargetDataExtractor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.reset_output_data()

    def reset_output_data(self):
        """ reset_output_data function called by pipeline """
        self.output_data = {}

    def main_process(self):
        """ main_process of TargetDataExtractor """

        if len(self.input_data[0]) != len(self.input_data[1]):
            self.terminate_flag = False
            self.logger.error("lengths of input_data do not match")
            return False

        data_list = self.input_data[0]
        label_list = self.input_data[1]
        for label in set(label_list):
            self.output_data[label] = []

        for i in range(0, len(label_list)):
            self.output_data[label_list[i]].append(data_list[i])

        if "sel_key" in self.param.keys():
            self.output_data = self.output_data[self.param["sel_key"]]
