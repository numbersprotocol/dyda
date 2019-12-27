import copy
from dt42lab.core import image
from dyda.components import determinator
from dyda.core import image_processor_base


class BlackImageSelector(image_processor_base.ImageProcessorBase):
    """Select none black image input"""

    def __init__(self, dyda_config_path=''):
        super(BlackImageSelector, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)
        self.select_black = False
        if "select_black" in self.param.keys():
            self.select_black = self.param["select_black"]

    def main_process(self):
        self.unpack_single_input()
        self.output_data = True
        try:
            if image.is_black(self.input_data):
                if self.select_black:
                    self.output_data = True
                else:
                    self.output_data = False
        except Exception as e:
            print(e)


class SelectorTargetLabel(determinator.DeterminatorTargetLabel):
    """The detected object in the input inferencer result is
       left if the label is target one.
       output_data will be True is the taget label cannot be found
       means when it is used as selector, the next process will not be skipped

       @param target: list of target label.
    """

    def __init__(self, dyda_config_path='', param=None):
        super(SelectorTargetLabel, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        if isinstance(self.param['target'], str):
            self.param['target'] = [self.param['target']]
        self.sel_none = False
        if "sel_none" in self.param.keys():
            self.sel_none = self.param["sel_none"]
        self.all_find_as_pass = False
        # For multiple input annotations
        if "all_find_as_pass" in self.param.keys():
            self.all_find_as_pass = self.param["all_find_as_pass"]

    def main_process(self):
        self.results = copy.deepcopy(self.input_data)
        if isinstance(self.results, dict):
            self.output_data = False
            self.results = self.extract_target(self.results)
            if self.results is None:
                if self.sel_none:
                    self.output_data = True
                else:
                    self.output_data = False
            elif len(self.results['annotations']) > 0:
                self.output_data = False
            return self.output_data

        elif isinstance(self.results, list):
            self.output_data = False
            for i in range(len(self.results)):
                self.results[i] = self.extract_target(self.results[i])
                if self.all_find_as_pass:
                    if self.results[i] is None:
                        self.output_data = False
                        return self.output_data
                    elif len(self.results[i]['annotations']) == 0:
                        self.output_data = False
                        return self.output_data
                elif self.sel_none and self.results[i] is None:
                    self.output_data = True
                elif len(self.results[i]['annotations']) > 0:
                    self.output_data = True
                    return self.output_data
            return self.output_data
