import copy
from dyda_utils import tinycv
from dyda_utils import lab_tools
from dyda.core import validator_base


class BinaryClassificationValidator(validator_base.ValidatorBase):
    """ Validator of binary classification model """

    def __init__(self, dyda_config_path=''):
        """ __init__ of BinaryClassificationValidator """

        super(BinaryClassificationValidator, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

    def add_key_to_results(self, key):
        """ Add a new key to lab_info of self.results """
        self.results["lab_info"][key] = {"p": 0, "t": 0, "m": 0}

    def main_process(self):
        """ define main_process of dyda component """

        if len(self.input_data) != 2:
            self.terminate_flag = True
            self.logger.error(
                "Length of input_data is not 2 but %i." % len(self.input_data)
            )
            return False
        l1 = len(self.input_data[0])
        l2 = len(self.input_data[1])
        if l1 != l2:
            self.terminate_flag = True
            self.logger.error(
                "Lengths of two input lists are not the same."
            )
            return False

        conf_thre = 0.0
        if "conf_thre" in self.param.keys():
            conf_thre = self.param["conf_thre"]

        self.results["lab_info"] = {"total": {"all": 0, "m": 0}}

        all_labels = list(set(self.input_data[1]))
        if len(all_labels) != 2:
            self.terminate_flat = True
            self.logger.error(
                "Wrong N labels, this is for binary classification only"
            )
            return False
        for label in all_labels:
            self.add_key_to_results(label)

        if self.param["sel_label"] not in all_labels:
            self.terminate_flag = True
            self.logger.error(
                "sel_label %s is not in true label set"
                % self.param["sel_label"]
            )
            return False
        label_1 = self.param["sel_label"]
        label_2 = all_labels[0] if all_labels[0] != label_1 else all_labels[1]

        lab_info = self.results["lab_info"]
        for i in range(0, l1):
            # Fist list in input_data should be results from classifier
            # Second list in input_data should be corresponding labels
            anno_result = self.input_data[0][i]
            true_label = self.input_data[1][i]
            if "annotations" not in anno_result:
                self.logger.error("No annotation found in entry %i" % i)
                continue
            pred_label = anno_result["annotations"][0]["label"]
            conf = anno_result["annotations"][0]["confidence"]

            # t: True, p: Predicted, m: Matched
            lab_info["total"]["all"] = lab_info["total"]["all"] + 1
            lab_info[true_label]["t"] = lab_info[true_label]["t"] + 1
            if pred_label == label_1:
                lab_info[label_1]["p"] = lab_info[label_1]["p"] + 1
                if true_label == label_1:
                    lab_info[label_1]["m"] = lab_info[label_1]["m"] + 1
                    lab_info["total"]["m"] = lab_info["total"]["m"] + 1
            else:
                if conf <= conf_thre:
                    lab_info[label_1]["p"] = lab_info[label_1]["p"] + 1
                    if true_label == label_1:
                        lab_info[label_1]["m"] = lab_info[label_1]["m"] + 1
                        lab_info["total"]["m"] = lab_info["total"]["m"] + 1
                else:
                    lab_info[label_2]["p"] = lab_info[label_2]["p"] + 1
                    if true_label == label_2:
                        lab_info[label_2]["m"] = lab_info[label_2]["m"] + 1
                        lab_info["total"]["m"] = lab_info["total"]["m"] + 1

        total_pred = 0
        total_match = 0
        for label in all_labels:
            if lab_info[label]["p"] > 0:
                lab_info[label]["precision"] = \
                    lab_info[label]["m"] / float(lab_info[label]["p"])
            else:
                lab_info[label]["precision"] = 0
            if lab_info[label]["t"] > 0:
                lab_info[label]["recall"] = \
                    lab_info[label]["m"] / float(lab_info[label]["t"])
            else:
                lab_info[label]["recall"] = 0

            total_pred = total_pred + lab_info[label]["p"]
            total_match = total_match + lab_info[label]["m"]

        if lab_info["total"]["all"] > 0:
            recall = total_match / float(lab_info["total"]["all"])
        else:
            recall = 0
        if total_pred > 0:
            self.results["precision"] = total_match / float(total_pred)
        else:
            self.results["precision"] = 0
        # For classification, recall = accuracy
        # = how many are currected detected in all samples
        self.results["recall"] = recall
        self.results["accuracy"] = recall
        self.results["stat_error"] = self.cal_stat_error(
            recall, lab_info["total"]["all"]
        )
        self.results["nsamples"] = lab_info["total"]["all"]

        return True


class ClassificationValidator(validator_base.ValidatorBase):
    """ Validator of classification model """

    def __init__(self, dyda_config_path=''):
        """ __init__ of ClassificationValidator """

        super(ClassificationValidator, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

    def add_key_to_results(self, key):
        """ Add a new key to lab_info of self.results """
        self.results["lab_info"][key] = {"p": 0, "t": 0, "m": 0}

    def main_process(self):
        """ define main_process of dyda component """

        if len(self.input_data) != 2:
            self.terminate_flag = True
            self.logger.error(
                "Length of input_data is not 2 but %i." % len(self.input_data)
            )
            return False
        l1 = len(self.input_data[0])
        l2 = len(self.input_data[1])
        if l1 != l2:
            self.terminate_flag = True
            self.logger.error(
                "Lengths of two input lists are not the same."
            )
            return False

        conf_thre = 0.0
        if "conf_thre" in self.param.keys():
            conf_thre = self.param["conf_thre"]

        self.results["lab_info"] = {"total": {"all": 0, "m": 0}}
        all_labels = set(self.input_data[1])
        for label in all_labels:
            self.add_key_to_results(label)

        lab_info = self.results["lab_info"]
        for i in range(0, l1):
            # Fist list in input_data should be results from classifier
            # Second list in input_data should be corresponding labels
            anno_result = self.input_data[0][i]
            true_label = self.input_data[1][i]
            if "annotations" not in anno_result:
                self.logger.error("No annotation found in entry %i" % i)
                continue
            pred_label = anno_result["annotations"][0]["label"]
            conf = anno_result["annotations"][0]["confidence"]

            # t: True, p: Predicted, m: Matched
            lab_info["total"]["all"] = lab_info["total"]["all"] + 1
            lab_info[true_label]["t"] = lab_info[true_label]["t"] + 1
            if conf > conf_thre:
                lab_info[pred_label]["p"] = lab_info[pred_label]["p"] + 1
            if pred_label == true_label and conf >= conf_thre:
                lab_info[pred_label]["m"] = lab_info[pred_label]["m"] + 1
                lab_info["total"]["m"] = lab_info["total"]["m"] + 1

        total_pred = 0
        total_match = 0
        for label in all_labels:
            if lab_info[label]["p"] > 0:
                lab_info[label]["precision"] = \
                    lab_info[label]["m"] / float(lab_info[label]["p"])
            else:
                lab_info[label]["precision"] = 0
            if lab_info[label]["t"] > 0:
                lab_info[label]["recall"] = \
                    lab_info[label]["m"] / float(lab_info[label]["t"])
            else:
                lab_info[label]["recall"] = 0

            total_pred = total_pred + lab_info[label]["p"]
            total_match = total_match + lab_info[label]["m"]

        if lab_info["total"]["all"] > 0:
            recall = total_match / float(lab_info["total"]["all"])
        else:
            recall = 0
        if total_pred > 0:
            self.results["precision"] = total_match / float(total_pred)
        else:
            self.results["precision"] = 0
        # For classification, recall = accuracy
        # = how many are currected detected in all samples
        self.results["recall"] = recall
        self.results["accuracy"] = recall
        self.results["stat_error"] = self.cal_stat_error(
            recall, lab_info["total"]["all"]
        )
        self.results["nsamples"] = lab_info["total"]["all"]

        return True


class DetectionValidator(validator_base.ValidatorBase):
    """ Validator of detection model """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(DetectionValidator, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def add_key_to_results(self, key):
        """ Add a new key to lab_info of self.results """
        self.results["lab_info"][key] = {
            "true_pos": 0,
            "false_pos": 0,
            "gt_number": 0,
            "ap": 0}

    def div(self, numerator, denominator):
        """ division which return 0 when denominator is 0 """
        if denominator == 0:
            return 0.0
        else:
            return float(numerator) / denominator

    def get_prec_and_recall(self, data):
        """ Calculate precision and recall  """
        data['precision'] = self.div(
            data['true_pos'],
            data['true_pos'] + data['false_pos'])
        data['recall'] = self.div(
            data['true_pos'],
            data['gt_number'])

    def get_voc_ap(self, rec, prec):
        """ Claculate the area under the curve as
            average precision (ap)
        """
        # insert 0.0 at begining of list and 1.0 at end of list
        rec.insert(0, 0.0)
        rec.append(1.0)
        mrec = rec[:]
        prec.insert(0, 0.0)
        prec.append(0.0)
        mpre = prec[:]

        # makes the precision monotonically decreasing
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        # creates a list of indexes where the recall changes
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                i_list.append(i)

        # calculate ap
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i] - mrec[i - 1]) * mpre[i])

        return ap

    def main_process(self):
        """ define main_process of dyda component """

        if len(self.input_data) != 2:
            self.terminate_flag = True
            self.logger.error(
                "Length of input_data is not 2 but %i." % len(self.input_data)
            )
            return False
        l1 = len(self.input_data[0])
        l2 = len(self.input_data[1])
        if l1 != l2:
            self.terminate_flag = True
            self.logger.error(
                "Lengths of two input lists are not the same."
            )
            return False

        conf_thre = 0.0
        if "conf_thre" in self.param.keys():
            conf_thre = self.param["conf_thre"]

        self.results = {
            "true_pos": 0,
            "false_pos": 0,
            "gt_number": 0,
            "mAP": 0,
            "lab_info": {}}

        predictions_data_all = self.input_data[0]
        ground_truth_data_all = self.input_data[1]

        nd = 0
        for data in predictions_data_all:
            nd += len(data['annotations'])

        gt_classes = []
        for data in ground_truth_data_all:
            for anno in data['annotations']:
                label = anno['label']
                if label not in gt_classes:
                    self.add_key_to_results(label)
                    gt_classes.append(label)
                lab_info = self.results["lab_info"]
                lab_info[label]['gt_number'] += 1

        lab_info = self.results["lab_info"]
        for class_index, class_name in enumerate(gt_classes):
            tp = [0] * nd
            fp = [0] * nd
            idx_obj = -1
            for idx, pred in enumerate(predictions_data_all):
                pred = lab_tools.extract_target_class(
                    copy.deepcopy(pred),
                    class_name
                )
                pred_anno = pred['annotations']
                pred_anno.sort(
                    key=lambda x: float(x['confidence']),
                    reverse=True)
                gt_anno = ground_truth_data_all[idx]['annotations']
                for obj in pred_anno:
                    idx_obj += 1
                    ovmax = -1
                    gt_match = -1
                    bb = tinycv.Rect([
                        obj['top'],
                        obj['bottom'],
                        obj['left'],
                        obj['right']])
                    for obj_gt in gt_anno:
                        if obj_gt["label"] != class_name:
                            continue
                        bbgt = tinycv.Rect(
                            [obj_gt['top'],
                             obj_gt['bottom'],
                             obj_gt['left'],
                             obj_gt['right']])
                        bi = [max(bb.l, bbgt.l),
                              max(bb.t, bbgt.t),
                              min(bb.r, bbgt.r),
                              min(bb.b, bbgt.b)]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw <= 0 or ih <= 0:
                            continue
                        ua = (bb.r - bb.l + 1) * \
                            (bb.b - bb.t + 1) + \
                            (bbgt.r - bbgt.l + 1) * \
                            (bbgt.b - bbgt.t + 1) - \
                            iw * ih
                        ov = iw * ih / ua
                        if ov <= ovmax:
                            continue
                        ovmax = ov
                        gt_match = obj_gt
                    min_overlap = self.param['min_overlap']
                    if ovmax >= min_overlap:
                        if "used" not in gt_match.keys(
                        ) or not bool(gt_match["used"]):
                            # true positive
                            tp[idx_obj] = 1
                            gt_match["used"] = True
                            lab_info[class_name]['true_pos'] += 1
                        else:
                            # false positive
                            lab_info[class_name]['false_pos'] += 1
                            fp[idx_obj] = 1
                    else:
                        # false positive
                        lab_info[class_name]['false_pos'] += 1
                        fp[idx_obj] = 1
                        if ovmax > 0:
                            status = "insufficient overlap"

            # get precision and recall
            data = lab_info[class_name]
            self.get_prec_and_recall(data)
            self.results['true_pos'] += data['true_pos']
            self.results['false_pos'] += data['false_pos']
            self.results['gt_number'] += data['gt_number']

            # get voc average precision
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = self.div(
                    tp[idx],
                    lab_info[class_name]['true_pos'])
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = self.div(
                    tp[idx],
                    fp[idx] + tp[idx])

            lab_info[class_name]['ap'] = self.get_voc_ap(rec, prec)
            self.results['mAP'] += lab_info[class_name]['ap']

        self.get_prec_and_recall(self.results)
        self.results['mAP'] = self.div(
            self.results['mAP'],
            len(gt_classes))
