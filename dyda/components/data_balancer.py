from random import shuffle
from dyda.core import data_preprocessor_base


class DataBalancerSimple(data_preprocessor_base.DataPreProcessor):
    """ Simple data balancer """

    def __init__(self, dyda_config_path=''):
        """ __init__ of DataBalancerSimple """

        super(DataBalancerSimple, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)
        self.results = []
        self.input_data = []

    def main_process(self):
        """ define main_process of dyda component """

        image_paths = self.input_data[0]
        labels = self.input_data[1]

        if len(image_paths) != len(labels):
            self.terminate_flag = True
            self.logger.error("Lengths of labels and data do not match")
            return False

        min_items = len(labels)
        counter = {}
        for label in set(labels):
            counter[label] = 0
            _min_items = labels.count(label)
            if _min_items > 0 and _min_items < min_items:
                min_items = _min_items
        if "multiplier" in self.param.keys():
            min_items = min_items * self.param["multiplier"]
        self.logger.warning(
            "Balancer will shuffle and select %i of each label." % min_items
        )
        index_list = [i for i in range(0, len(labels))]
        shuffle(index_list)
        output_index = []
        for i in index_list:
            label = labels[i]
            if counter[label] >= min_items:
                continue
            counter[label] += 1
            output_index.append(i)

        self.results = [[image_paths[i] for i in output_index],
                        [labels[i] for i in output_index]]

        return True
