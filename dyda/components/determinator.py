import sys
import copy
import numpy as np
import traceback
from dyda_utils import tinycv
from dyda_utils import tools
from dyda_utils import lab_tools
from dyda_utils import pandas_data
from dyda.core import determinator_base


class DeterminatorByRoi(determinator_base.DeterminatorBase):
    """ The detected object in the input_data is kept if the overlap ratio
        with roi is larger than threshold. Overlap ratio is defined as
        interaction area / object area.
       @param top: top of roi.
       @param bottom: bottom of roi.
       @param left: left of roi.
       @param right: right of roi.
       @param threshold: overlap ratio threshold.

    """

    def __init__(self, dyda_config_path="", param=None):
        """Initialization function of dyda component.
           This is called when an object is created from the class and
           it allows the class to initialize the attributes of a class.
        """

        super(DeterminatorByRoi, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        if 'use_external_meta' not in self.param:
            self.use_external_meta = False
        else:
            self.use_external_meta = self.param['use_external_meta']
        self.set_re_assign_id()

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()
        skip = self.parse_param()

        if skip:
            self.results = input_data
        else:
            for data in input_data:
                self.results.append(self.roi_determine(data))

        self.uniform_results()

    def parse_param(self):
        """ Parse parameters to extract roi info. """

        if self.use_external_meta:
            if isinstance(self.external_metadata, list) \
                    and len(self.external_metadata) == 1:
                if 'roi' not in self.external_metadata[0].keys():
                    self.logger.warning(
                        "No roi given and skip roi determinator")
                    return True
                rois = self.external_metadata[0]['roi']
            elif isinstance(self.external_metadata, dict):
                if 'roi' not in self.external_metadata.keys():
                    self.logger.warning(
                        "No roi given and skip roi determinator")
                    return True
                rois = self.external_metadata['roi']
            else:
                self.logger.error("Wrong external_metadata type")
                self.terminate_flag = True
                return False
            self.rois = []
            for roi in rois:
                bb = tinycv.Rect([
                    roi['top'],
                    roi['bottom'],
                    roi['left'],
                    roi['right']])
                if bb.r == -1 or bb.l == -1 or \
                        bb.t == -1 or bb.b == -1:
                    return True
                bb.th = roi['overlap_threshold']
                self.rois.append(bb)
        else:
            self.rois = []
            bb = tinycv.Rect([
                self.param['top'],
                self.param['bottom'],
                self.param['left'],
                self.param['right']])
            if bb.r == -1 or bb.l == -1 or \
                    bb.t == -1 or bb.b == -1:
                return True
            bb.th = self.param['threshold']
            self.rois.append(bb)
        return False

    def roi_determine(self, data):
        """ Determine if objects detected are in the roi. """

        if not lab_tools.if_result_match_lab_format(data):
            self.terminate_flag = True
            self.logger.error("Input is not valid.")
            return False

        annotations = data['annotations']
        for i in range(len(annotations) - 1, -1, -1):
            top = annotations[i]['top']
            bottom = annotations[i]['bottom']
            left = annotations[i]['left']
            right = annotations[i]['right']
            kept = False
            for roi in self.rois:
                interaction_area = max(
                    min(roi.r, right) - max(roi.l, left),
                    0) * max(
                    min(roi.b, bottom) - max(roi.t, top),
                    0)
                object_area = (right - left) * (bottom - top)
                ratio = float(interaction_area) / float(object_area)
                if ratio >= roi.th:
                    kept = True
            if not kept:
                annotations.pop(i)

        annotations = self.run_re_assign_id(annotations)

        return data


class DeterminatorByAggregatedDataSingle(determinator_base.DeterminatorBase):
    """ Determine results by min/max/mean of aggregated data
        input_data: dict or list of lab results
        output_data: determined list of lab results
        WARNING: this component only support single result now!
    """

    def __init__(self, dyda_config_path="", param=None):
        """Initialization function of dyda component.
           This is called when an object is created from the class and
           it allows the class to initialize the attributes of a class.
        """

        super(DeterminatorByAggregatedDataSingle, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.filter_rule = "max"
        if "filter_rule" in self.param.keys():
            self.filter_rule = self.param["filter_rule"]
        if self.filter_rule not in ["max", "min"]:
            self.logger.error(
                "%s is not a supported filter rule" % self.filter_rule
            )
            sys.exit(0)

        self.comp_rule = "mean"
        if "comp_rule" in self.param.keys():
            self.comp_rule = self.param["comp_rule"]
        if self.comp_rule not in ["mean", "sum"]:
            self.logger.error(
                "%s is not a supported rule" % self.comp_rule
            )
            sys.exit(0)

        self.cal_base = "confidence"
        if "cal_base" in self.param.keys():
            self.cal_base = self.param["cal_base"]

        self.groups = ["label", "id"]
        if "groups" in self.param.keys():
            self.groups = self.param["groups"]
        if not isinstance(self.groups, list):
            self.logger.error(
                "groups should be a list."
            )
            sys.exit(0)

        self.selected = "label"
        if "selected" in self.param.keys():
            self.selected = self.param["selected"]
        if self.selected not in self.groups:
            self.logger.error(
                "selected parameter must be a member of groups"
            )
            sys.exit(0)

        self.agg_num = 5
        if "agg_num" in self.param.keys():
            self.agg_num = self.param["agg_num"]
        if not isinstance(self.agg_num, int):
            self.logger.error(
                "agg_num should be integer"
            )
            sys.exit(0)
        self.logger.info(
            "Will aggregate %i results and output" % self.agg_num
        )
        self.agg_results = []

    def main_process(self):
        """ Main function of dyda component. """

        input_data = self.pack_as_list(copy.deepcopy(self.input_data))
        # [190506] WARNING: this component only support single result now!
        self.agg_results.append(input_data[0])
        if len(self.agg_results) == self.agg_num + 1:
            self.agg_results = self.agg_results[1:self.agg_num + 1]

        df = pandas_data.create_anno_df_and_concat(self.agg_results)
        label, conf = pandas_data.select_item_from_target_values(
            df, self.groups, self.selected, self.cal_base,
            filter_rule=self.filter_rule, comp_rule=self.comp_rule
        )
        # [190506] INFO: Leverage the bb of latest result
        self.results = self.pack_as_list(copy.deepcopy(self.input_data))
        self.results[0]["annotations"][0][self.selected] = label
        self.results[0]["annotations"][0][self.cal_base] = conf
        if self.unpack_single_list:
            self.unpack_single_results()


class DeterminatorByDynamicRoi(DeterminatorByRoi):
    """ Same as DeterminatorByRoi but use dynamic ROI input
        input_data[0] ROI
        input_data[1] Detected results
        param threshold: overlap ratio threshold.
    """

    def __init__(self, dyda_config_path="", param=None):
        """Initialization function of dyda component.
           This is called when an object is created from the class and
           it allows the class to initialize the attributes of a class.
        """

        super(DeterminatorByDynamicRoi, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.set_re_assign_id()
        self.thre = 0.1
        if "overlap_threshold" in self.param.keys():
            self.thre = self.param["overlap_threshold"]

    def main_process(self):
        """ Main function of dyda component. """

        if not isinstance(self.input_data, list) or \
                len(self.input_data) != 2:
            self.terminate_flag = True
            self.logger.error(
                "Input data type should be a list of ROI and detected results"
            )
            return False

        roi_setting = self.input_data[0]
        roi_setting = self.unpack_list(roi_setting)
        if not isinstance(roi_setting, dict):
            self.terminate_flag = True
            self.logger.error(
                "ROI settings should be dict or a list of dict"
            )

        detections = self.input_data[1]
        detections = self.uniform_input(ori_input=detections)
        roi_found = self.parse_param(roi_setting)

        if roi_found:
            for data in detections:
                self.results.append(self.roi_determine(data))
        else:
            self.results = detections

        self.uniform_results()

    def parse_param(self, roi_settings):
        """ Parse parameters to extract roi info. """

        if lab_tools.is_lab_format(roi_settings):
            rois = roi_settings["annotations"]
        elif "roi" in roi_settings.keys():
            rois = roi_settings["roi"]
        else:
            self.logger.warning("No valid ROI found")
            return False

        self.rois = []
        for roi in rois:
            try:
                bb = tinycv.Rect([
                    max(0, roi['top']),
                    roi['bottom'],
                    max(0, roi['left']),
                    roi['right']])
                if bb.b == -1 or bb.r == -1:
                    self.rois.append(None)
                else:
                    self.rois.append(bb)

            except BaseException:
                return False
        return True

    def roi_determine(self, data):
        """ Determine if objects detected are in the roi. """

        if not lab_tools.if_result_match_lab_format(data):
            self.terminate_flag = True
            self.logger.error("Input is not valid.")
            return False

        annotations = data['annotations']
        for i in range(len(annotations) - 1, -1, -1):
            top = annotations[i]['top']
            bottom = annotations[i]['bottom']
            left = annotations[i]['left']
            right = annotations[i]['right']
            kept = True
            for roi in self.rois:
                if roi is None:
                    continue
                int_w = max(min(roi.r, right) - max(roi.l, left), 0)
                int_h = max(min(roi.b, bottom) - max(roi.t, top), 0)
                interaction_area = int_w * int_h
                object_area = (right - left) * (bottom - top)
                ratio = float(interaction_area) / float(object_area)
                if ratio < self.thre:
                    kept = False
            if not kept:
                annotations.pop(i)

        annotations = self.run_re_assign_id(annotations)
        return data


class DeterminatorSelAnnoInGivenInterval(determinator_base.DeterminatorBase):
    """
       Select annotations in the given interval and only output one.
       ::input_data::results
       ::output_data::results
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(DeterminatorSelAnnoInGivenInterval, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.interval = 5
        if 'interval' in self.param.keys():
            self.interval = int(self.param['interval'])
        self.counter = 0
        self.previous = -1

    def main_process(self):
        """ Main function of dyda component. """

        input_data = self.uniform_input()
        self.results = []

        for result in input_data:
            self.results.append(result)
            # only drop results it the format match
            if not lab_tools.is_lab_format(result):
                continue
            if result["annotations"]:
                diff = self.counter - self.previous
                if self.previous < 0 or diff >= self.interval:
                    self.previous = self.counter
                else:
                    # clear annotations if within specified interval
                    self.results[-1]["annotations"] = []
        self.counter += 1


class DeterminatorTargetLabel(determinator_base.DeterminatorBase):
    """The detected object in the input inferencer result is
       left if the label is in target list.
       output_data will be True is the taget label cannot be found
       means when it is used as selector, the next process will not be skipped

       @param target: list of target label.
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(DeterminatorTargetLabel, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        if isinstance(self.param['target'], str):
            self.param['target'] = [self.param['target']]
        self.set_re_assign_id()
        if 'output_when_target_exists' not in self.param.keys():
            self.output_flag = False
        else:
            self.output_flag = self.param['output_when_target_exists']

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        self.output_data = not self.output_flag
        for data in input_data:
            self.results.append(self.extract_target(data))
            if self.results[-1]['annotations']:
                # self.output_data is true if the label is found in
                # any of the list component
                self.output_data = self.output_flag

        self.uniform_results()

    def extract_target(self, data):
        """ Extract target label. """

        if not lab_tools.if_result_match_lab_format(data):
            self.terminate_flag = True
            self.logger.error("Input is not valid.")
            return False

        annotations = data['annotations']
        for i in range(len(annotations) - 1, -1, -1):
            if not annotations[i]['label'] in self.param['target']:
                annotations.pop(i)
        annotations = self.run_re_assign_id(annotations)

        return(data)


class DeterminatorThreshold(determinator_base.DeterminatorBase):
    """The detected object in the input inferencer result is
       left if the key value is higher/lower than the threshold.

       @param threshold: threshold of key value.
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(DeterminatorThreshold, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        for data in input_data:
            self.results.append(self.extract_by_threshold(data))

        self.uniform_results()

    def extract_by_threshold(self, data):
        """ Extract objects by given key and threshold. """

        if not lab_tools.if_result_match_lab_format(data):
            self.terminate_flag = True
            self.logger.error("Input is not valid.")
            return False

        anno = data['annotations']
        for di in range(len(anno) - 1, -1, -1):
            if not self.param['key'] in anno[di].keys():
                self.terminate_flag = True
                self.logger.error(
                    "Target key is not found")
                return False
            value = anno[di][self.param['key']]
            if self.param['type'] == 'larger':
                if not value > self.param['threshold']:
                    anno.pop(di)
            elif self.param['type'] == 'smaller':
                if not value < self.param['threshold']:
                    anno.pop(di)
            else:
                self.terminate_flag = True
                self.logger.error(
                    "Parameter type is not supported")
                return False
        for di in range(len(anno)):
            anno[di]['id'] = di

        return(data)


class DeterminatorBinaryConfThreshold(determinator_base.DeterminatorBase):
    """For one binary classification result, use conf threshold
       to determine the label.
    """

    def __init__(self, dyda_config_path='', param=None):
        """Initialization function of dyda component.
        """

        super(DeterminatorBinaryConfThreshold, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        label_path = self.param["label_file"]
        self.labels = []
        self.results = []
        try:
            self.labels = tools.txt_to_list(label_path)
        except BaseException:
            traceback.print_exc(file=sys.stdout)
            sys.exit(0)
        if len(self.labels) != 2:
            self.logger.error(
                "Wrong number of  labels, this is for binary classification"
                " only. Check %s." % label_path
            )
            sys.exit(0)
        if self.param["sel_label"] not in self.labels:
            self.logger.error(
                "sel_label %s is not in label set"
                % self.param["sel_label"]
            )
            sys.exit(0)

        self.label_1 = self.param["sel_label"]
        self.label_2 = self.labels[0] if self.labels[0] != self.label_1 \
            else self.labels[1]

        self.conf_thre = 0.0
        if "conf_thre" in self.param.keys():
            self.conf_thre = self.param["conf_thre"]
        self.logger.info("confidence threshold = %.2f" % self.conf_thre)

    def reset_results(self):
        """ reset_results function called by pipeline.py """
        self.results = []

    def main_process(self):
        """ main process of DeterminatorBinaryConfThreshold """

        self.reset_output()
        input_data = self.uniform_input()

        for result in input_data:
            try:
                pred_label = result["annotations"][0]["label"]
                conf = result["annotations"][0]["confidence"]
            except BaseException:
                self.logger.error("Fail to get annotations")
                self.terminate_flag = True
                return False

            if pred_label == self.label_2 and conf <= self.conf_thre:
                self.logger.debug(
                    "Changing label from %s to %s" % (pred_label, self.label_1)
                )
                result["annotations"][0]["label"] = self.label_1
                new_conf = 1 - result["annotations"][0]["confidence"]
                result["annotations"][0]["confidence"] = new_conf
            self.results.append(result)

        self.uniform_results()


class DeterminatorValidAnnoFirst(determinator_base.DeterminatorBase):
    """Pick the valid annotation from several input lists

    """

    def __init__(self, dyda_config_path='', param=None):
        super(DeterminatorValidAnnoFirst, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        self.results = []
        first_list = None
        for anno_list in self.input_data:
            if len(anno_list) == 0:
                continue
            if first_list is None:
                first_list = anno_list
            if len(anno_list) != len(first_list):
                self.terminate_flag = True
                self.logger.error("Length of input lists do not match")
                return False

        for i in range(0, len(first_list)):
            self.results.append({})
            for j in range(0, len(self.input_data)):
                if len(self.input_data[j]) == 0:
                    continue
                anno_list = self.input_data[j]
                if "annotations" in anno_list[i].keys() and \
                        len(anno_list[i]["annotations"]) > 0:
                    self.results[-1] = anno_list[i]
                    break


class DeterminatorConfidenceThreshold(determinator_base.DeterminatorBase):
    """The detected object in the input inferencer result is
       left if the confidence score is hight than the threshold.

       @param key_to_apply: key to apply the threshold in annotations.
       @param threshold: threshold of confidence score.
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(DeterminatorConfidenceThreshold, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.key_to_apply = 'confidence'
        if 'key_to_apply' in self.param.keys():
            self.key_to_apply = self.param['key_to_apply']
        self.thre = 0
        if 'threshold' in self.param.keys():
            self.thre = self.param['threshold']
        self.set_re_assign_id()

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        for data in input_data:
            if not lab_tools.if_result_match_lab_format(data):
                self.terminate_flag = True
                self.logger.error("Input is not valid.")
                return None

            self.results.append(self.confidence_threshold(data))
        self.uniform_results()

    def confidence_threshold(self, data):
        """ Threshold by confidence. """

        annotations = data['annotations']
        for di in range(len(annotations) - 1, -1, -1):
            if annotations[di][self.key_to_apply] < self.thre:
                annotations.pop(di)
        annotations = self.run_re_assign_id(annotations)

        return data


class DeterminatorSizeThreshold(determinator_base.DeterminatorBase):
    """The detected object in the input inferencer result is
       left if the size is larger than the threshold.

    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(DeterminatorSizeThreshold, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.set_re_assign_id()

    def get_threshold(self, data):
        """ Get size threshold. """
        self.thre = 0
        if 'threshold' in self.param.keys():
            if self.param['threshold'] > 1:
                self.thre = self.param['threshold']
            else:
                if data['size']['width'] is None or \
                        data['size']['height'] is None or \
                        data['size']['width'] < 0 or \
                        data['size']['height'] < 0:
                    self.logger.warning("Image size is not valid "
                                        "for adaptive threshold.")
                else:
                    base_length = max(
                        data['size']['width'],
                        data['size']['height'])
                    self.thre = self.param['threshold'] * base_length

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        for data in input_data:
            if not lab_tools.if_result_match_lab_format(data):
                self.terminate_flag = True
                self.logger.error("Input is not valid.")
                return None
            self.get_threshold(data)
            self.results.append(self.size_threshold(data))
        self.uniform_results()

    def size_threshold(self, data):
        """ Threshold by size. """

        annotations = data['annotations']
        for di in range(len(annotations) - 1, -1, -1):
            width = annotations[di]['right'] - annotations[di]['left']
            height = annotations[di]['bottom'] - annotations[di]['top']
            if min(width, height) < self.thre:
                annotations.pop(di)
        annotations = self.run_re_assign_id(annotations)

        return data


class DeterminatorGroup(determinator_base.DeterminatorBase):
    """The detected objects in the input inferencer result is
       grouped as one object when they are closed enough.

       @param threshold: be groupped when ratio of overlap
            is larger than the threshold
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(DeterminatorGroup, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        for data in input_data:
            if not lab_tools.if_result_match_lab_format(data):
                self.terminate_flag = True
                self.logger.error("Input is not valid.")
                return None

            self.results.append(lab_tools.grouping(
                data, self.param['threshold']))

        self.uniform_results()


class DeterminatorSortByArea(determinator_base.DeterminatorBase):
    """The detected objects in the input inferencer result is
       sorted by area and left the first N largest/smallest ones.

       @param number: number of objects to be left
       @param mode: large or small
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(DeterminatorSortByArea, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        if 'cross_list' not in self.param.keys():
            self.param['cross_list'] = False

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        if self.param['cross_list']:
            anno_all = []
            for data in input_data:
                anno_all.extend(data['annotations'])
            input_data = [input_data[0]]
            input_data[0]['annotations'] = anno_all

        for data in input_data:
            if not lab_tools.if_result_match_lab_format(data):
                self.terminate_flag = True
                self.logger.error("Input is not valid.")
                return None

            self.results.append(lab_tools.sort_by_area(data))
            left_number = min(
                self.param['number'], len(
                    self.results[-1]['annotations']))
            if self.param['mode'] == 'small':
                self.results[-1]['annotations'] = reversed(
                    self.results[-1]['annotations'])
            for j in range(len(self.results[-1]['annotations']) - 1,
                           left_number - 1, -1):
                del self.results[-1]['annotations'][j]

        if self.package or self.param['cross_list']:
            self.results = self.results[0]


class DeterminatorSortByAspect(determinator_base.DeterminatorBase):
    """The detected objects in the input inferencer result is
       sorted by area and left the first N largest/smallest ones.

       @param number: number of objects to be left
       @param mode: large or small
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(DeterminatorSortByAspect, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        if 'cross_list' not in self.param.keys():
            self.param['cross_list'] = False

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        if self.param['cross_list']:
            anno_all = []
            for data in input_data:
                anno_all.extend(data['annotations'])
            input_data = [input_data[0]]
            input_data[0]['annotations'] = anno_all

        for data in input_data:
            if not lab_tools.if_result_match_lab_format(data):
                self.terminate_flag = True
                self.logger.error("Input is not valid.")
                return None

            self.results.append(
                lab_tools.sort_by_aspect_ratio(data))
            left_number = min(
                self.param['number'], len(
                    self.results[-1]['annotations']))
            if self.param['mode'] == 'small':
                self.results[-1]['annotations'] = reversed(
                    self.results[-1]['annotations'])
            for j in range(len(self.results[-1]['annotations']) - 1,
                           left_number - 1, -1):
                del self.results[-1]['annotations'][j]

        if self.package or self.param['cross_list']:
            self.results = self.results[0]


class DeterminatorLastingSec(determinator_base.DeterminatorBase):
    """Determine lasting time of objects by track_id.

    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(DeterminatorLastingSec, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.base_time_list = []

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        input_len = len(input_data)
        if self.base_time_list == []:
            self.base_time_list = [{} for i in range(input_len)]
        elif input_len != len(self.base_time_list):
            self.terminate_flag = True
            self.logger.error("Length of input_data changed.")
            return None

        for idx, data in enumerate(input_data):
            if not lab_tools.if_result_match_lab_format(data):
                self.terminate_flag = True
                self.logger.error("Input is not valid.")
                return None
            self.results.append(
                self.add_lasting_seconds(data, idx))

        self.uniform_results()

    def add_lasting_seconds(self, data, idx):
        """ Add lasting time to lab-format dict. """

        current_time = self.filename_to_time(data['filename'])
        for anno in data['annotations']:
            if 'track_id' not in anno.keys():
                anno['lasting_seconds'] = 0
            elif anno['track_id'] < 0:
                anno['lasting_seconds'] = 0
            elif not anno['track_id'] in self.base_time_list[idx].keys():
                self.base_time_list[idx][anno['track_id']] = current_time
                anno['lasting_seconds'] = 0
            else:
                base_time = self.base_time_list[idx][anno['track_id']]
                anno['lasting_seconds'] = current_time - base_time
                if anno['lasting_seconds'] < 0:
                    self.logger.warning("Lasting time is negative number.")
        return data

    def filename_to_time(self, filename):
        """ Turn filename to time. """

        try:
            return float(filename)
        except ValueError:
            self.logger.warning("Filename is not supported to count "
                                "lasting time.")
            return 0.0


class DeterminatorMotion(determinator_base.DeterminatorBase):
    """Determine angle and distance of objects by track_id.

    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(DeterminatorMotion, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.base_position_list = []

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        input_len = len(input_data)
        if self.base_position_list == []:
            self.base_position_list = [{} for i in range(input_len)]
        elif input_len != len(self.base_position_list):
            self.terminate_flag = True
            self.logger.error("Length of input_data changed.")
            return None

        for idx, data in enumerate(input_data):
            if not lab_tools.if_result_match_lab_format(data):
                self.terminate_flag = True
                self.logger.error("Input is not valid.")
                return None
            self.results.append(
                self.add_angle_distance(data, idx))

        self.uniform_results()

    def add_angle_distance(self, data, idx):
        """ Add angle and distance to lab-format dict. """

        base_position = self.base_position_list[idx]

        # reset appear to false
        for track_id in base_position.keys():
            base_position[track_id]['appear'] = False

        for anno in data['annotations']:

            # assign default value
            anno['motion_distance'] = None
            anno['motion_angle'] = None

            # continue if no valid track_id
            if 'track_id' not in anno.keys() or anno['track_id'] < 0:
                continue

            track_id = anno['track_id']
            base_position = self.base_position_list[idx]

            # calculate center of bounding box
            center_x = (anno['left'] + anno['right']) / 2
            center_y = (anno['top'] + anno['bottom']) / 2

            # calculate motion distance and angle
            if track_id in base_position.keys():
                center_x_pre = base_position[track_id]['x']
                center_y_pre = base_position[track_id]['y']
                if center_x_pre and center_y_pre:
                    x = center_x - base_position[track_id]['x']
                    y = center_y - base_position[track_id]['y']
                    anno['motion_distance'] = np.sqrt(x**2 + y**2)
                    anno['motion_angle'] = -np.arctan2(y, x) * 180 / np.pi

            # update center location
            base_position[track_id] = {
                'x': center_x,
                'y': center_y,
                'appear': True
            }

        # set x and y to none when not appear
        for track_id in base_position.keys():
            if not base_position[track_id]['appear']:
                base_position[track_id]['x'] = None
                base_position[track_id]['y'] = None

        return data


class DeterminatorColorHistogram(determinator_base.DeterminatorBase):
    """Determine color histogram of each object in annotations.

    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(DeterminatorColorHistogram, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        annos = self.uniform_input(self.input_data[0], 'lab-format')
        imgs = self.uniform_input(self.input_data[1], 'ndarray')

        if len(annos) != len(imgs):
            self.terminate_flag = True
            self.logger.error("Length of input dict and images do not match.")
            return None

        for i, img in enumerate(imgs):
            for obj in annos[i]['annotations']:
                obj_img = img[obj['top']: obj['bottom'],
                              obj['left']:obj['right'], :]
                obj['color_hist'] = lab_tools.calculate_color_hist(
                    obj_img,
                    self.param['hist_length'])
            self.results.append(annos[i])

        self.uniform_results()

    def uniform_input(self, input_data, dtype):
        """ Package input_data if it is not a list and
            check input data type
        """

        # package input_data if it is not a list
        input_data = copy.deepcopy(input_data)
        if not isinstance(input_data, list):
            input_data = [input_data]
            self.package = True
        else:
            self.package = False

        # check input data type and
        valid = True
        for data in input_data:
            if dtype == "str":
                if not isinstance(data, str):
                    valid = False
            elif dtype == "lab-format":
                if not lab_tools.if_result_match_lab_format(data):
                    valid = False
            elif dtype == "ndarray":
                if not isinstance(data, np.ndarray):
                    valid = False
            else:
                self.base_logger.warning('dtype is not supported to check')

        # when data type is not valid, raise terminate_flag and
        # return empty list to skip following computation
        if not valid:
            self.base_logger.error('Invalid input data type')
            self.terminate_flag = True
            input_data = []

        return input_data


class DeterminatorReplaceLabel(determinator_base.DeterminatorBase):
    """Replace label according to old_label/new_label pairs defined
       in dyda config.

    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(DeterminatorReplaceLabel, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.old_labels = [pair['old_label']
                           for pair in self.param['replace_pairs']]
        self.new_labels = [pair['new_label']
                           for pair in self.param['replace_pairs']]

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input(self.input_data, 'lab-format')

        for data in input_data:
            for obj in data['annotations']:
                try:
                    idx = self.old_labels.index(obj['label'])
                    obj['label'] = self.new_labels[idx]
                except ValueError:
                    pass
            self.results.append(data)

        self.uniform_results()

    def uniform_input(self, input_data, dtype):
        """ Package input_data if it is not a list and
            check input data type
        """

        # package input_data if it is not a list
        input_data = copy.deepcopy(input_data)
        if not isinstance(input_data, list):
            input_data = [input_data]
            self.package = True
        else:
            self.package = False

        # check input data type and
        valid = True
        for data in input_data:
            if dtype == "str":
                if not isinstance(data, str):
                    valid = False
            elif dtype == "lab-format":
                if not lab_tools.if_result_match_lab_format(data):
                    valid = False
            elif dtype == "ndarray":
                if not isinstance(data, np.ndarray):
                    valid = False
            else:
                self.base_logger.warning('dtype is not supported to check')

        # when data type is not valid, raise terminate_flag and
        # return empty list to skip following computation
        if not valid:
            self.base_logger.error('Invalid input data type')
            self.terminate_flag = True
            input_data = []

        return input_data
