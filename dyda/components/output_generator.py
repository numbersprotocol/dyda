import os
import copy
from dyda.core import output_generator_base
from dt42lab.core import lab_tools


class FullPathGenerator(output_generator_base.OutputGeneratorBase):
    """ Generate output with folder and filename info from FrameReader """

    def __init__(self, dyda_config_path="", param=None):
        """ Initialization function of FullPathGenerator"""

        super(FullPathGenerator, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.results = []

    def main_process(self):
        """ Main function of dyda component. """

        if isinstance(self.input_data, list):
            for lab_result in self.input_data:
                self.results.append(self.get_full_path(lab_result))
        elif isinstance(self.input_data, dict):
            self.results.append(self.get_full_path(self.input_data))
        else:
            self.terminate_flag = True
            self.logger.error("Input data type is neither dict nor list")
        return True

    def get_full_path(self, lab_result):
        """ Function to get full path from lab result """

        try:
            folder = lab_result["folder"]
            filename = lab_result["filename"]
            return os.path.join(folder, filename)
        except BaseException:
            self.logger.error("input_data format is not valid")
            return ""

    def reset_results(self):
        """ reset_results function called by pipeline.py """

        self.results = []


class OutputGeneratorWithFileInfo(output_generator_base.OutputGeneratorBase):
    """ Generate output with folder and filename info from FrameReader """

    def __init__(self, dyda_config_path="", param=None):
        """ Initialization function of OutputGeneratorWithFileInfo"""

        super(OutputGeneratorWithFileInfo, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.fr_results = {}

    def find_frame_reader(self):
        """ Find frame_reader results from metadata """
        for comp in self.metadata[1:]:
            if comp["class_name"] == "FrameReader":
                self.fr_results = comp["results"]

    def main_process(self):
        """ Main function of dyda component. """
        self.find_frame_reader()
        self.results = copy.deepcopy(self.input_data)
        if isinstance(self.results, dict):
            if_valid = lab_tools.if_result_match_lab_format(self.input_data)
            if not if_valid:
                self.terminate_flag = True
                self.logger.error("input_data format is not valid")
                return False
            if "folder" in self.param["modify_fileds"]:
                self.results["folder"] = os.path.dirname(
                    self.fr_results["data_path"][0]
                )
            if "filename" in self.param["modify_fileds"]:
                self.results["folder"] = os.path.basename(
                    self.fr_results["data_path"][0]
                )
        elif isinstance(self.results, list):
            for i in range(0, len(self.results)):
                result = self.results[i]
                if_valid = lab_tools.if_result_match_lab_format(result)
                if not if_valid:
                    self.logger.error(
                        "input_data format is not valid, skip %ith" % i
                    )
                    continue
                if "folder" in self.param["modify_fileds"]:
                    result["folder"] = os.path.dirname(
                        self.fr_results["data_path"][i]
                    )
                if "filename" in self.param["modify_fileds"]:
                    result["filename"] = os.path.basename(
                        self.fr_results["data_path"][i]
                    )


class OutputGeneratorAOI(output_generator_base.OutputGeneratorBase):
    """ Generate output to meet specification.
    """

    def __init__(self, dyda_config_path="", param=None):
        """Initialization function of dyda component.
           This is called when an object is created from the class and
           it allows the class to initialize the attributes of a class.
        """

        super(OutputGeneratorAOI, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        thre = 0
        if "conf_threshold" in self.param.keys():
            thre = self.param["conf_threshold"]
        sigma = 3
        if "sigma" in self.param.keys():
            sigma = self.param["sigma"]

        self.logger.info("Filter results with %i sigma" % sigma)
        self.results = copy.deepcopy(self.input_data)
        for result in self.results:
            error = 0
            result["keep"] = False
            if "error" in result.keys():
                error = result["error"] * sigma
            upper = thre + error
            upper = 1 if upper > 1 else upper
            lower = thre - error
            lower = 0 if lower < 0 else lower
            for anno in result["annotations"]:
                conf = anno["confidence"]
                anno["trust_level"] = 0
                if conf >= upper:
                    anno["trust_level"] = 2
                elif conf >= lower:
                    anno["trust_level"] = 1
                if anno["label"] == "ok" and anno["trust_level"] >= 1:
                    result["keep"] = True


class OutputGeneratorLpr(output_generator_base.OutputGeneratorBase):
    """ Generate output to meet specification.
    """

    def __init__(self, dyda_config_path="", param=None):
        """ Initialization function of dyda component. """

        super(OutputGeneratorLpr, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        # initialize input and output
        self.output_data = []
        self.results = {}
        self.pre_plate_infos = []

    def main_process(self):
        """ Main function of dyda component. """

        # return default results when pipeline status is 0
        if self.pipeline_status == 0:
            self.results = {
                "folder": None,
                "filename": None,
                "timestamp": self.metadata[0],
                "size": {
                    "width": None,
                    "height": None
                },
                "annotations": []
            }
            return

        plate_infos = self.lpr_determination()
        plate_infos = self.check_with_pre_lpr(plate_infos)

        self.results = {
            "folder": None,
            "filename": None,
            "timestamp": image_info["filename"],
            "size": {
                "width": None,
                "height": None
            },
            "annotations": image_info["annotations"]
        }
        for i in range(len(self.results["annotations"])):
            self.results["annotations"][i]["lpr"] = plate_infos[i]["lpr"]
            if not plate_infos[i]["lpr_score"] == []:
                self.results["annotations"][i]["lpr_confidence"] = sum(
                    plate_infos[i]["lpr_score"]) / float(
                    len(plate_infos[i]["lpr_score"]))
            else:
                self.results["annotations"][i]["lpr_confidence"] = 0
            if 'track_score' in self.results["annotations"][i].keys():
                del self.results["annotations"][i]['track_score']

    def lpr_determination(self, plate_infos):
        """ Determine lpr . """

        char_locs = self.input_data[1]
        char_infos = self.input_data[2]

        lprs = []
        scores = []
        for plate_idx, plate_info in enumerate(plate_infos):
            lpr = ''
            score = []
            for i in range(len(char_locs)):
                if not char_locs[i]['plate_id'] == plate_idx:
                    continue
                char_info = char_infos[i]['annotations'][0]
                if char_info['label'] == 'unknown':
                    lpr += '*'
                    score.append(0)
                elif char_info["confidence"] > self.param['confidence_thre']:
                    lpr += char_info['label']
                    score.append(char_info['confidence'])
                else:
                    lpr += '*'
                    score.append(0)
            plate_infos[plate_idx]['lpr'] = lpr
            plate_infos[plate_idx]['lpr_score'] = score
        return(plate_infos)

    def check_with_pre_lpr(self, plate_infos):

        if self.pre_plate_infos == []:
            self.pre_plate_infos = copy.deepcopy(plate_infos)
            return(plate_infos)

        match_result = lab_tools.match_by_overlap_ratio(
            plate_infos,
            self.pre_plate_infos)
        for idx1 in range(len(plate_infos)):
            if idx1 not in match_result['match_index_1']:
                plate_infos[idx1]['lpr'] = ''
                plate_infos[idx1]['lpr_score'] = []
                continue
            j = match_result['match_index_1'].index(idx1)
            idx2 = match_result['match_index_2'][j]
            lpr_now = plate_infos[idx1]['lpr']
            score_now = plate_infos[idx1]['lpr_score']
            lpr_pre = self.pre_plate_infos[idx2]['lpr']
            score_pre = self.pre_plate_infos[idx2]['lpr_score']

            len_now = len(lpr_now)
            len_pre = len(lpr_pre)
            lpr_now_new = '*' * (len_pre - 1) * 2 + \
                lpr_now + '*' * (len_pre - 1)
            score_now_new = [0] * (len_pre - 1) * 2 + \
                score_now + [0] * (len_pre - 1)
            lpr_pre_new = '*' * (len_now - 1) + \
                lpr_pre + '*' * (len_now - 1)
            score_pre_new = [0] * (len_now - 1) + \
                score_pre + [0] * (len_now - 1)
            match_max = 0
            score_max = []
            lpr_max = ''
            for pi in range(len(lpr_now_new) - len_now + 1):
                match = []
                lpr = ''
                score = []
                for qi in range(len(lpr_pre_new)):
                    idx_now = pi + qi
                    idx_pre = qi
                    if idx_pre < 0 or idx_pre >= len(lpr_pre_new):
                        continue
                    if idx_now < 0 or idx_now >= len(lpr_now_new):
                        continue
                    if lpr_now_new[idx_now] == '*' and \
                            lpr_pre_new[idx_pre] == '*':
                        match.append(False)
                    elif lpr_now_new[idx_now] == lpr_pre_new[idx_pre]:
                        match.append(True)
                    else:
                        match.append(False)
                    if score_now_new[idx_now] > score_pre_new[idx_pre]:
                        score.append(score_now_new[idx_now])
                        lpr += lpr_now_new[idx_now]
                    else:
                        score.append(score_pre_new[idx_pre])
                        lpr += lpr_pre_new[idx_pre]
                if sum(match) > match_max:
                    match_max = sum(match)
                    score_max = score
                    lpr_max = lpr

            if match_max >= self.param['match_bit_thre']:
                lpr = lpr_max
                score = score_max
            elif sum(score_pre) > sum(score_now):
                lpr = lpr_pre
                score = score_pre
            else:
                lpr = lpr_now
                score = score_now

            known = [i for i, x in enumerate(lpr) if not x == '*']
            if len(known) > 0:
                lpr = lpr[known[0]:known[-1] + 1]
                score = score[known[0]:known[-1] + 1]

            plate_infos[idx1]['lpr'] = lpr
            plate_infos[idx1]['lpr_score'] = score
        self.pre_plate_infos = copy.deepcopy(plate_infos)

        return(plate_infos)


class OutputGeneratorBehavior(output_generator_base.OutputGeneratorBase):
    """ Generate output to meet specification.
    """

    def __init__(self, dyda_config_path="", param=None):
        """Initialization function of dyda component.
           This is called when an object is created from the class and
           it allows the class to initialize the attributes of a class.
        """

        super(OutputGeneratorBehavior, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        # return default results when pipeline status is 0
        if self.pipeline_status == 0 and self.is_empty_input(self.input_data):
            self.results = []
            for i in range(self.param['channel_num']):
                self.results.append({
                    'timestamp': self.metadata[0],
                    'channel_index': i,
                    'size': {'height': None, 'width': None},
                    'annotations': []
                })
            return

        # initialize input and output
        self.output_data = []
        self.results = []

        self.results = copy.deepcopy(self.input_data)
        for ci, data in enumerate(self.results):
            data['timestamp'] = data['filename']
            data['channel_index'] = ci
            data['annotations'] = []
            del data['folder']
            del data['filename']
            for anno in self.input_data[ci]['annotations']:
                data['annotations'].append({
                    'label': anno['label'],
                    'confidence': anno['confidence'],
                    'top': anno['top'],
                    'bottom': anno['bottom'],
                    'left': anno['left'],
                    'right': anno['right'],
                    'type': anno['type']})
                res = data['annotations'][-1]
                if anno['type'] == 'classification':
                    res['id'] = -1
                    res['event_objects'] = self.get_event_objects(anno)
                elif anno['type'] == 'detection':
                    res['id'] = anno['id']
                    res['track_id'] = anno['track_id']

    def get_event_objects(self, anno):
        if 'event_objects' in anno.keys():
            event_objects = anno['event_objects']
        elif 'track_id' in anno.keys():
            event_objects = anno['track_id']
        else:
            event_objects = []
        if not isinstance(event_objects, list):
            event_objects = [event_objects]
        return event_objects

    def is_empty_input(self, input_data):
        """ Check if input data is empty.
        """

        output = False
        if isinstance(input_data, list):
            if len(input_data) == 0:
                output = True
            elif len(input_data) == 1 and not input_data[0]:
                output = True
        elif isinstance(input_data, dict):
            if not input_data:
                output = True
        return output


class OutputGeneratorAnomalyClassification(
        output_generator_base.OutputGeneratorBase):
    """ Generate output to meet specification.
    """

    def __init__(self, dyda_config_path="", param=None):
        """Initialization function of dyda component.
           This is called when an object is created from the class and
           it allows the class to initialize the attributes of a class.
        """

        super(OutputGeneratorAnomalyClassification, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.default_result = [{
            "timestamp": "",
            "annotations": []
        }]

    def main_process(self):
        """ Main function of dyda component. """

        # return default results when pipeline status is 0
        if self.pipeline_status == 0:
            self.results = copy.deepcopy(self.default_result)
            return

        # reset output
        self.output_data = []
        self.results = []

        # parse input_data
        classification_info = self.input_data[0]
        roi_info = self.input_data[1]
        for ai, anno in enumerate(classification_info):
            self.results.append({
                "timestamp": self.metadata[0],
                "annotations": {
                    "label": anno["annotations"][0]["label"],
                    "confidence": anno["annotations"][0]["confidence"],
                    "top": roi_info[ai]["top"],
                    "bottom": roi_info[ai]["bottom"],
                    "left": roi_info[ai]["left"],
                    "right": roi_info[ai]["right"]
                }
            })
        self.default_result = copy.deepcopy(self.results)


class OutputGeneratorImgLabFormat(output_generator_base.OutputGeneratorBase):
    """ Generate output to meet specification.
    """

    def __init__(self, dyda_config_path="", param=None):
        """Initialization function of dyda component.
           This is called when an object is created from the class and
           it allows the class to initialize the attributes of a class.
        """

        super(OutputGeneratorImgLabFormat, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.default_label = ""
        if "default_label" in self.param.keys():
            self.default_label = self.param["default_label"]

    def reset_results(self):
        self.results = []

    def main_process(self):
        """ Main function of dyda component. """

        # parse input_data
        if isinstance(self.input_data, list):
            for i in range(0, len(self.input_data)):
                img = self.input_data[i]
                shape = img.shape
                self.results.append(
                    lab_tools.output_pred_classification(
                        "", 1.0, self.default_label,
                        img_size=[shape[1], shape[0]]
                    )
                )
                self.results[-1]["annotations"][0]["top"] = 0
                self.results[-1]["annotations"][0]["left"] = 0
                self.results[-1]["annotations"][0]["bottom"] = shape[0]
                self.results[-1]["annotations"][0]["right"] = shape[1]
                self.results[-1]["annotations"][0]["rot_angle"] = 0

        if self.unpack_single_list:
            self.unpack_single_results()


class OutputGeneratorCombineDetCla(output_generator_base.OutputGeneratorBase):
    """ Combine detection and classification results. """

    """ 2019/03/18 George Lin,
        I add a parameter 'behavior' to select whether to replace
        annotations in detection by annotations in classificaion.

        @param "behavior": "replace" or "append". If select "replace",
                           the component would act like before, replacing
                           "label", "confidence", "type", and "labinfo"
                           in detection with those in classification.
                           If select "append", the component will add a
                           prefix, "cla_", in classification's result, and
                           append to detection's result.
        @"cla_key_name": specify the key name of classificaion's result
                         to append to detection's result, when selecting
                         "append" in "behavior". If specify the key name
                         exists in detection's result, classificaion's result
                         will replace detection's result.
                         example: {"label": "person"}, will append
                         classification's "label" in detection's result, with
                         key name, "person".
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(OutputGeneratorCombineDetCla, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        if "behavior" in self.param.keys():
            self.behavior = self.param["behavior"]
        else:
            self.behavior = "replace"

        if "cla_key_name" in self.param.keys():
            self.cla_key_name = self.param["cla_key_name"]
        else:
            self.cla_key_name = {}

    def main_process(self):
        """ Main function of dyda component. """

        # return default results when pipeline status is 0
        if self.pipeline_status == 0:
            self.results = {
                "folder": None,
                "filename": None,
                "timestamp": self.metadata[0],
                "size": {
                    "width": None,
                    "height": None
                },
                "annotations": []
            }
            return

        self.results = []
        self.output_data = []

        input_data = copy.deepcopy(self.input_data)
        det_data, unpack = self.unpack(input_data[0])
        cla_data = input_data[1]

        if len(det_data['annotations']) != len(cla_data):
            self.terminate_flag = True
            self.logger.error(
                "Length of detection result"
                "and classification result do not match.")
            return False

        # classification and detection results must have same length
        # if they don't, search comment "20190506 OutputGeneratorCombineDetCla"
        # in trello card https://trello.com/c/wtjcSrTB

        if self.behavior == "replace":
            for ci, det_anno in enumerate(det_data['annotations']):
                cla_anno = cla_data[ci]['annotations'][0]
                det_anno['label'] = cla_anno['label']
                det_anno['confidence'] = cla_anno['confidence']
                det_anno['type'] = cla_anno['type']
                det_anno['labinfo'] = cla_anno['labinfo']
            self.results = det_data

        elif self.behavior == "append":
            for ci, det_anno in enumerate(det_data['annotations']):
                cla_anno = cla_data[ci]['annotations'][0]
                if 'label' in self.cla_key_name.keys():
                    det_anno[self.cla_key_name['label']] = cla_anno['label']
                else:
                    det_anno['cla_label'] = cla_anno['label']

                if 'confidence' in self.cla_key_name.keys():
                    det_anno[self.cla_key_name['confidence']] = \
                        cla_anno['confidence']
                else:
                    det_anno['cla_confidence'] = cla_anno['confidence']

                if 'type' in self.cla_key_name.keys():
                    det_anno[self.cla_key_name['type']] = cla_anno['type']
                else:
                    det_anno['cla_type'] = cla_anno['type']

                if 'labinfo' in self.cla_key_name.keys():
                    det_anno[self.cla_key_name['labinfo']] = \
                        cla_anno['labinfo']
                else:
                    det_anno['cla_labinfo'] = cla_anno['labinfo']

            self.results = det_data

        if unpack:
            self.results = [self.results]

    def unpack(self, data):
        """Unpack data if it is a list."""

        unpack = False
        if isinstance(data, list):
            if len(data) == 1:
                data = data[0]
                unpack = True
            else:
                self.terminate_flag = True
                self.logger.error(
                    "Support one detection result only")
        return(data, unpack)
