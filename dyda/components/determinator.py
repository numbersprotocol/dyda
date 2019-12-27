import os
import cv2
import sys
import copy
import numpy as np
import traceback
import statistics
import operator
from scipy.stats import mode
from dyda_utils import tinycv
from dyda_utils import tools
from dyda_utils import lab_tools
from dyda_utils import image
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


class DeterminatorCharacter(determinator_base.DeterminatorBase):
    """Determine the location of characters in license plate.
       @param bg_dilation_kernel_size: kernel size for
            background dilation.
       @param bg_dilation_iter_num: iteration number for
            background dilation.
       @param projection_h_length_ratio_thre: length ratio
            threshold for segmentation by horizontal projection.
       @param projection_h_percentile_thre: percentile
            threshold for segmentation by horizontal projection.
       @param projection_v_length_ratio_thre: length ratio
            threshold for segmentation by vertical projection.
       @param projection_v_percentile_thre: percentile
            threshold for segmentation by vertical projection.

    """

    def __init__(self, dyda_config_path='', param=None):
        super(DeterminatorCharacter, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):

        # parse input_data
        img = copy.deepcopy(self.input_data[0])
        detection = copy.deepcopy(self.input_data[1])
        detection['size']['height'], detection['size']['width'] = img.shape[:2]

        detection = lab_tools.extend_detection(
            detection,
            self.param['plate_ext_top'],
            self.param['plate_ext_bottom'],
            self.param['plate_ext_left'],
            self.param['plate_ext_right'])

        # initialize output
        self.output_data = []
        self.results = []

        # extend bounding box
        detection['size']['height'] = img.shape[0]
        detection['size']['width'] = img.shape[1]

        if 'annotations' in detection.keys():
            for pi in range(len(detection['annotations'])):
                char_info, char_imgs = self.extract_char(
                    copy.deepcopy(img), detection, pi
                )
                self.output_data.extend(char_imgs)
                self.results.extend(char_info)

    def post_process(self, out_folder_base=""):
        """ Post_process of dyda component.
            This is called when lab_flag is true
            to save images or dict for checking results.
        """
        results = self.input_data[1]
        if 1:  # len(results["annotations"])>0:
            img = self.input_data[0]

            # show patched image
            keys = ['label']
            color = [0, 0, 255]
            line_width = 6
            space = 5
            for i in range(0, len(results["annotations"])):
                loc = (results["annotations"][i]["top"],
                       results["annotations"][i]["bottom"],
                       results["annotations"][i]["left"],
                       results["annotations"][i]["right"])
                img = tinycv.draw_bounding_box(img, loc, color, line_width)
                text = []

                for key in keys:
                    value = results["annotations"][i][key]
                    if isinstance(value, (float)):
                        value = "{0:.2f}".format(value)
                    elif isinstance(value, (int)):
                        value = str(value)
                    text.append(value)
                    img = tinycv.patch_text(
                        img,
                        value,
                        color=color,
                        loc=(
                            results["annotations"][i]["left"] +
                            space,
                            results["annotations"][i]["top"] +
                            space))

            # show intermediate results
            out_img = img
            for i in range(len(self.out_img)):
                img = self.out_img[i]
                img = img.astype(np.uint8)
                if 0 in img.shape:
                    continue
                img = image.resize_img(img, (out_img.shape[1], None))
                out_img = np.vstack(
                    (out_img, np.ones((img.shape[0], out_img.shape[1], 3),
                                      dtype=np.uint8) * 0))

                out_img[-img.shape[0]:, :img.shape[1], :] = img
            out_fn = os.path.join(out_folder_base, self.metadata[0] + '.jpg')
            cv2.imwrite(out_fn, out_img)
            print('[DeterminatorCharacter] INFO: save image to {}'.format(
                out_fn))

    def extract_char(self, img, detection, plate_idx):

        # initialize output
        char_info = []
        char_imgs = []

        # extract image info
        img_height, img_width = img.shape[:2]

        # extract plate info
        plate_info = detection['annotations'][plate_idx]
        confidence = plate_info['confidence']
        top = plate_info['top']
        bottom = plate_info['bottom']
        left = plate_info['left']
        right = plate_info['right']
        height = bottom - top
        width = right - left

        var_min = np.inf
        found = False

        # search angle
        angle = self.search_angle(img, plate_info)

        # shift to compensate plate detection error
        shift = []
        for si in range(
                self.param['shift_ratio_min_y'],
                self.param['shift_ratio_max_y']):
            for sj in range(
                    self.param['shift_ratio_min_x'],
                    self.param['shift_ratio_max_x']):
                shift.append([int(height * si / 10), int(width * sj / 4)])
        for [shift_y, shift_x] in shift:
            top_p = max(0, top + shift_y)
            bottom_p = min(img_height, bottom + shift_y)
            left_p = max(0, left + shift_x)
            right_p = min(img_width, right + shift_x)
            plate_ori = copy.deepcopy(
                img[top_p:bottom_p, left_p:right_p, :])

            # rgb to gray
            plate_ori_g = cv2.cvtColor(
                plate_ori, cv2.COLOR_BGR2GRAY)

            # global binarization
            thre = int(np.mean(plate_ori_g.mean(axis=1)))
            ret, bin_global = cv2.threshold(
                plate_ori_g, thre, 1, cv2.THRESH_BINARY)

            # rotation
            plate_rot = tinycv.rotate(copy.deepcopy(plate_ori), angle)
            plate_rot_g = tinycv.rotate(copy.deepcopy(plate_ori_g), angle)
            plate_rot_b = tinycv.rotate(copy.deepcopy(bin_global), angle)

            # extract plate
            idx_h0, idx_v0 = self.extract_plate(plate_rot_b)
            plate_seg0 = plate_rot_g[idx_h0[0][0]:idx_h0[0][1],
                                     idx_v0[0][0]:idx_v0[0][1]]

            # local binarization
            for bin_thre in range(
                    self.param['bin_thre_min'],
                    self.param['bin_thre_max'],
                    self.param['bin_thre_gap']):

                filter_size = int(plate_seg0.shape[0] / 2)
                if filter_size % 2 == 0:
                    filter_size += 1

                bin_local = cv2.adaptiveThreshold(
                    copy.deepcopy(plate_seg0), 1,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY, filter_size,
                    bin_thre)

                # segment plate
                idx_h1, idx_v1, length = self.segment_plate(bin_local)

                if len(length) == 0:
                    continue
                var = statistics.pvariance(length)
                height_1 = idx_h1[0][1] - idx_h1[0][0]
                r_min = self.param['char_ratio_min']
                r_max = self.param['char_ratio_max']
                c_num = self.param['char_number']
                if len(idx_v1) in c_num and var < var_min and \
                        height_1 > max(length) * r_min and \
                        height_1 < max(length) * r_max:
                    var_min = var
                    found = True
                    plate_rot_f = copy.deepcopy(plate_rot)
                    plate_seg0_f = copy.deepcopy(plate_seg0)
                    idx_h0_f = copy.deepcopy(idx_h0)
                    idx_v0_f = copy.deepcopy(idx_v0)
                    idx_h1_f = copy.deepcopy(idx_h1)
                    idx_v1_f = copy.deepcopy(idx_v1)

        if found:
            # refine segmentation
            idx_v1_f = self.refine_segmentation(
                idx_v1_f, plate_seg0_f.shape[1])

            for i in range(len(idx_v1_f)):

                # char info
                top_ = idx_h0_f[0][0] + idx_h1_f[0][0]
                bottom_ = idx_h0_f[0][0] + idx_h1_f[0][1]
                left_ = idx_v0_f[0][0] + idx_v1_f[i][0]
                right_ = idx_v0_f[0][0] + idx_v1_f[i][1]
                char_info.append(copy.deepcopy({
                    'type': 'detection',
                    'label': 'character',
                    'confidence': confidence,
                    'top': top_,
                    'bottom': bottom_,
                    'left': left_,
                    'right': right_,
                    'plate_id': plate_idx,
                    'location': (0, i)
                }))

                # char image
                char_img = plate_rot_f[top_:bottom_, left_:right_, :]
                # white balance
                from PIL import Image, ImageOps
                char_img = Image.fromarray(
                    cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB))
                char_img = ImageOps.autocontrast(char_img, 3)
                char_img = cv2.cvtColor(
                    np.asarray(char_img), cv2.COLOR_RGB2BGR)
                # append output char images
                char_imgs.append(
                    image.resize_img(
                        char_img,
                        (self.param['char_size'],
                         self.param['char_size'])))

        return(char_info, char_imgs)

    def search_angle(self, img, plate_info):
        # FIXME: use warpping instead
        init = False
        top = plate_info['top']
        bottom = plate_info['bottom']
        left = plate_info['left']
        right = plate_info['right']
        height = bottom - top
        for si in [self.param['shift_ratio_max_y']]:
            if self.param['shift_ratio_max_y'] > 0:
                shift = int(
                    height * si / (self.param['shift_ratio_max_y'] * 2))
            else:
                shift = 0
            plate_ori = copy.deepcopy(
                img[top + shift:bottom + shift, left:right, :])

            # rgb to gray
            plate_ori_g = cv2.cvtColor(
                plate_ori, cv2.COLOR_BGR2GRAY)

            # global binarization
            thre = int(np.mean(plate_ori_g.mean(axis=1)))
            ret, bin_global = cv2.threshold(
                plate_ori_g, thre, 1, cv2.THRESH_BINARY)

            # search angle
            angle_s = []
            for angle in range(
                    self.param['angle_min'],
                    self.param['angle_max'],
                    self.param['angle_gap']):
                plate_rot = tinycv.rotate(copy.deepcopy(bin_global), angle)
                proj_sum = plate_rot.sum(1)
                high_flag = proj_sum > 0.6 * plate_rot.shape[1]
                low_flag = proj_sum < 0.1 * plate_rot.shape[1]
                angle_score = sum(high_flag)
                if not init or angle_score > score_max:
                    init = True
                    score_max = angle_score
                    angle_s = [angle]
                elif angle_score == score_max:
                    angle_s.append(angle)
            if not angle_s == []:
                angle_f = angle_s[int(np.round(len(angle_s) / 2.0))]
            angle_s = []
            for angle in range(
                    angle_f - self.param['angle_gap'] + 1,
                    angle_f + self.param['angle_gap']):
                plate_rot = tinycv.rotate(copy.deepcopy(bin_global), angle)
                proj_sum = plate_rot.sum(1)
                high_flag = proj_sum > 0.6 * plate_rot.shape[1]
                low_flag = proj_sum < 0.1 * plate_rot.shape[1]
                angle_score = sum(high_flag)
                if angle_score > score_max:
                    score_max = angle_score
                    angle_s = [angle]
                elif angle_score == score_max:
                    angle_s.append(angle)
            if not angle_s == []:
                angle_f = angle_s[int(np.round(len(angle_s) / 2.0))]

        if score_max == 0:
            angle_f = 0
        return(angle_f)

    def extract_plate(self, bin_in):
        bin_h = bin_in.shape[0]
        bin_w = bin_in.shape[1]

        # projection v0
        idx_v0 = tinycv.segmentation_by_projection(
            copy.deepcopy(bin_in), 'v',
            self.param['projection_v0_length_ratio_thre'],
            self.param['projection_v0_percentile_thre'],
            target=1)
        if len(idx_v0) == 0:
            idx_v0.append([0, bin_w])

        # image projection h0
        idx_h0 = tinycv.segmentation_by_projection(
            copy.deepcopy(bin_in), 'h',
            self.param['projection_h0_length_ratio_thre'],
            self.param['projection_h0_percentile_thre'],
            target=1)
        if len(idx_h0) == 0:
            idx_h0.append([0, bin_h])

        return(idx_h0, idx_v0)

    def segment_plate(self, bin_in):
        bin_h = bin_in.shape[0]
        bin_w = bin_in.shape[1]

        # image projection h1
        idx_h1 = tinycv.segmentation_by_projection(
            copy.deepcopy(bin_in), 'h',
            self.param['projection_h1_length_ratio_thre'],
            self.param['projection_h1_percentile_thre'])
        if len(idx_h1) == 0:
            idx_h1.append([0, bin_h])

        # FIXME: support one line first
        length = [x[1] - x[0] for x in idx_h1]
        idx = length.index(max(length))
        idx_h1.insert(0, idx_h1[idx])
        del idx_h1[idx + 1]

        bin_in = copy.deepcopy(bin_in[
            idx_h1[0][0]:idx_h1[0][1], :])

        # image projection v1
        idx_v1 = tinycv.segmentation_by_projection(
            bin_in, 'v',
            self.param['projection_v1_length_ratio_thre'],
            self.param['projection_v1_percentile_thre'])
        length = [x[1] - x[0] for x in idx_v1]

        # calculate mode
        mode_list = []
        for p in length:
            ratio_list = []
            for q in length:
                ratio_list.append(np.round(p / float(q)))
            mode_list.append(int(mode(ratio_list)[0][0]))

        length = [x[1] - x[0] for x in idx_v1]

        return(idx_h1, idx_v1, length)

    def refine_segmentation(self, idx_v1_f, width_max):

        for i in range(0, len(idx_v1_f) - 1):
            gap = idx_v1_f[i + 1][0] - idx_v1_f[i][1]
            width = idx_v1_f[i][1] - idx_v1_f[i][0]
            if gap < width * 0.5:
                mid = int((idx_v1_f[i + 1][0] + idx_v1_f[i][1]) / 2)
                idx_v1_f[i + 1][0] = mid
                idx_v1_f[i][1] = mid
                if i == 1:
                    idx_v1_f[i - 1][0] = max(0,
                                             idx_v1_f[i - 1][0] - int(gap / 2))
                if i == len(idx_v1_f) - 2:
                    idx_v1_f[i + 1][1] = min(width_max,
                                             idx_v1_f[i + 1][1] + int(gap / 2))

        return(idx_v1_f)


class DeterminatorParkingLotStatus(determinator_base.DeterminatorBase):
    """Determine parking lot status including the status of cars
       and parking space.

       @param init_status_path: binery string to represent the
            initial status of the parking space.
       @param diff_threshold: threshold of color difference.
            The pixel is regarded as different when its pixelwise
            difference is larger the diff_threshold.
       @param ratio_threshold: threshold of difference ratio.
            The status of the space would change if its mean of
            diff_ratio is larger than ratio_threshold.
       @param mean_filter_size: filter to calculate mean of
            difference ratio.
       @param stability_threshold: the space in the background
            frame is updated if its stability < stability_threshld
            for a period of time(length_min).
       @param length_min: minimum length of frame to be calculate
            stability.

    """

    def __init__(self, dyda_config_path="", param=None):
        super(DeterminatorParkingLotStatus, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.frame_bg = []

        # initialize status
        init_status = tools.parse_json(
            self.param['init_status_path'])['status']
        space_num = len(init_status)
        self.status = []
        for i in range(space_num):
            if init_status[i] == '1':
                self.status.append(True)
            else:
                self.status.append(False)

        self.ratio_all = self.param['mean_filter_size'] * [
            [0.0] * space_num]

        self.unstable = [False] * space_num
        self.ratio_temporal = [[] for _ in range(space_num)]
        self.mean_ratio = [0.0] * space_num

    def main_process(self):

        self.frame_now = copy.deepcopy(self.input_data[0][0])
        frame_selector_result = copy.deepcopy(self.input_data[1])
        self.results = []

        # initialize background frame
        if self.frame_bg == []:
            self.frame_bg = self.frame_now

        if frame_selector_result['is_key']:
            frame_diff = tinycv.l1_norm_diff_cv2(
                self.frame_bg,
                self.frame_now)

            # space status
            self.output_data = copy.deepcopy(self.input_data[3])
            lab_info = self.output_data['lab_info']
            self.space_dependency = lab_info['space_dependency']
            # car detection
            self.results.append(copy.deepcopy(self.input_data[2]))

            # update space status
            space_annotations = self.output_data['annotations']
            self.ratio_all.pop(0)
            self.ratio_all.append([0.0] * len(self.status))

            for di in range(len(space_annotations) - 1, -1, -1):
                space_id = space_annotations[di]['track_id']
                diff_ratio = self.calculate_ratio(
                    space_annotations[di],
                    frame_diff)
                self.ratio_all[-1][space_id] = diff_ratio

                # calculate mean ratio
                ratio_sum = 0.0
                for ri in range(self.param['mean_filter_size']):
                    ratio_sum += self.ratio_all[ri][space_id]
                mean_ratio = ratio_sum / self.param['mean_filter_size']
                self.mean_ratio[space_id] = mean_ratio

            for di in range(len(space_annotations) - 1, -1, -1):
                space_id = space_annotations[di]['track_id']

                # update space status if mean ratio > ratio_threshold
                ratio_i = self.mean_ratio[space_id]
                if ratio_i > self.param['ratio_threshold'] and \
                        not self.unstable[space_id]:

                    # check local_max
                    local_max = True
                    for space_id_j in self.space_dependency[space_id]:
                        if space_id == space_id_j:
                            continue
                        ratio_j = self.mean_ratio[space_id_j]
                        if ratio_i - ratio_j < self.param['nms_threshold']:
                            local_max = False

                    # update status
                    if local_max:
                        self.unstable[space_id] = True
                        self.status[space_id] = not self.status[space_id]
                        self.ratio_temporal[space_id] = []

                space_annotations[di]['occupied'] = self.status[space_id]

                # update background if stable
                ratio_now = self.ratio_all[-1][space_id]
                self.ratio_temporal[space_id].append(ratio_now)
                length = len(self.ratio_temporal[space_id])
                if length > self.param['length_min']:
                    self.ratio_temporal[space_id].pop(0)
                    max_ratio = max(self.ratio_temporal[space_id])
                    min_ratio = min(self.ratio_temporal[space_id])
                    ratio_range = max_ratio - min_ratio
                    if ratio_range < self.param['ratio_range_threshold']:
                        self.updata_bg(space_id)

                        # check if the status the same as background
                        if self.unstable[space_id]:
                            self.unstable[space_id] = False
                            if ratio_now < self.param['ratio_threshold']:
                                self.status[space_id] = not self.status[
                                    space_id]

                space_annotations[di]['occupied'] = self.status[space_id]
            self.output_data['annotations'] = space_annotations

            # extract moving car
            car_annotations = self.results[0]['annotations']
            for di in range(len(car_annotations) - 1, -1, -1):
                diff_ratio = self.calculate_ratio(
                    car_annotations[di],
                    frame_diff)
                if diff_ratio < self.param['ratio_threshold']:
                    car_annotations.pop(di)
                else:
                    car_annotations[di]['confidence'] = diff_ratio

            # num-maximum suppression
            car_annotations = lab_tools.nms_with_confidence(
                car_annotations, threshold=0, nms_type='one_to_all')

            # num-nunify suppression
            car_annotations = lab_tools.nus_with_size(
                car_annotations + copy.deepcopy(space_annotations),
                num_std=2)

            # extract car only
            car_annotations = lab_tools.extract_target_value(
                {'annotations': car_annotations},
                "label",
                "car")['annotations']

            # append cars occupying space
            append_list = []
            delete_list = []
            for di in range(len(space_annotations)):
                space_id = space_annotations[di]['track_id']

                # avoid double bounding box after append
                if space_annotations[di]['occupied']:
                    if self.unstable[space_id] is False:
                        append_list.append(di)
                    else:
                        overlap_ratio_all = \
                            lab_tools.calculate_overlap_ratio_all(
                                [space_annotations[di]],
                                car_annotations)
                        if overlap_ratio_all.shape[1] > 0:
                            if overlap_ratio_all.max() > 0.8:
                                for j in range(overlap_ratio_all.shape[1]):
                                    if overlap_ratio_all[0][j] > 0:
                                        delete_list.append(j)
                                append_list.append(di)
                            elif overlap_ratio_all.max() < 0.2:
                                append_list.append(di)
                        else:
                            append_list.append(di)

            # delete car
            for di in sorted(list(set(delete_list)), reverse=True):
                car_annotations.pop(di)

            # append car
            for di in append_list:
                new_data = copy.deepcopy(space_annotations[di])
                new_data['track_id'] = -1
                new_data['label'] = 'car'
                del new_data['occupied']
                car_annotations.append(new_data)

            self.results[0]['annotations'] = car_annotations + \
                space_annotations
        else:
            # self.input_data[0] is not a inferencer result to be determined.
            pass

    def updata_bg(self, space_id):
        """Update the space whose status changed and other dependent space
           in the background frame using the current frame.

           @param space_id: the id of the space whose status changed.
        """
        annotations = self.output_data['annotations']
        for i in self.space_dependency[space_id]:
            top = annotations[i]['top']
            bottom = annotations[i]['bottom']
            left = annotations[i]['left']
            right = annotations[i]['right']
            self.frame_bg[top:bottom, left:right,
                          :] = self.frame_now[top:bottom, left:right, :]

    def calculate_stability(self, ratio_temporal):
        """The status is stable if the difference ratio changes slightly
            during a time interval (the length of ratio_temporal).

           @param ratio_temporal: list of different ratio to be calculate.
           @return stability: the higher the lower stable

        """

        stability = 0
        for i in range(len(ratio_temporal) - 1):
            abs_diff = abs(ratio_temporal[i] - ratio_temporal[i + 1])
            if abs_diff > stability:
                stability = abs_diff
        return stability

    def calculate_ratio(self, annotation, frame_diff):
        """ Calculate different pixel ratio of a given bounding box.
           @param annotation: location of the bounding box.
           @param frame_diff: pre-calculated difference between current
            frame and background frame.
           @return difference ratio: ratio of pixel that difference
            is larger than threshold.
        """
        if annotation == []:
            top = 0
            bottom = frame_diff.size[0]
            left = 0
            right = frame_diff.size[1]
        else:
            top = annotation['top']
            bottom = annotation['bottom']
            left = annotation['left']
            right = annotation['right']
        width = right - left
        height = bottom - top

        patch_diff = frame_diff[top:bottom, left:right]
        patch_flag = np.ones((height, width))
        patch_flag = patch_flag.astype(np.bool_)
        patch_flag = np.where(
            patch_diff < self.param['diff_threshold'], 0, 1)
        (N1, N2) = patch_flag.shape
        if N1 == 0:
            self.logger.error("Shape is not correct (%i, %i), please check the"
                              " input data and make sure it is from the"
                              " specified parking lot." % (N1, N2))
            sys.exit(0)

        diff_ratio = float(sum(sum(patch_flag))) / (height * width)
        return(diff_ratio)

    def post_process(self):
        output_parent_folder = self.lab_output_folder
        tools.check_dir(output_parent_folder)
        output_folder = os.path.join(
            output_parent_folder,
            self.__class__.__name__)
        tools.check_dir(output_folder)

        if not self.results == []:

            # write json
            out_filename = os.path.join(
                output_folder,
                tools.replace_extension(
                    self.metadata[0],
                    'json'))
            tools.write_json(self.results, out_filename)

            # write image of car
            out_car_image_filename = os.path.join(
                output_folder,
                tools.replace_extension(
                    self.metadata[0],
                    '.car.png'))
            json_file = lab_tools.extract_target_value(
                copy.deepcopy(self.results[0]),
                "label",
                "car")
            tinycv.patch_bb_by_key(
                json_file,
                color=[
                    0,
                    0,
                    255],
                keys=['label'],
                save=True,
                space=40,
                output_path=out_car_image_filename)

            # write image of space
            out_space_image_filename = os.path.join(
                output_folder,
                tools.replace_extension(
                    self.metadata[0],
                    '.space.png'))
            json_file = lab_tools.extract_target_value(
                copy.deepcopy(self.results[0]),
                "label",
                "space")
            tinycv.patch_bb_by_key(json_file, color=[0, 0, 255],
                                   keys=['occupied'], save=True, space=40,
                                   output_path=out_space_image_filename)


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
            if len(self.results[-1]['annotations']) > 0:
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


class DeterminatorLpr(determinator_base.DeterminatorBase):
    """Determine lpr from character classification results.
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(DeterminatorLpr, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        det_data = copy.deepcopy(self.input_data[0])
        cla_data = copy.deepcopy(self.input_data[1])

        plate_id = -1
        for i, data in enumerate(cla_data):
            change_plate = False
            if det_data[i]['plate_id'] != plate_id:
                self.results.append(copy.deepcopy({
                    'annotations': [{
                        'type': 'lpr',
                        'label': '',
                        'confidence': -1,
                        'labinfo': {
                            'DeterminatorLpr': [],
                            'ClassifierMobileNet': []}
                    }]}))
                plate_id = det_data[i]['plate_id']
                change_plate = True
            label = data['annotations'][0]['label']
            conf = data['annotations'][0]['confidence']
            lab_info = data['annotations'][0]['labinfo']['classifier']
            anno = self.results[-1]['annotations'][0]
            res_info = anno['labinfo']['DeterminatorLpr']
            anno['labinfo']['ClassifierMobileNet'].append(lab_info)
            if label == 'unknown':
                anno['label'] += '*'
                res_info.append(0)
            elif conf > self.param['confidence_thre']:
                anno['label'] += label.upper()
                res_info.append(conf)
            else:
                anno['label'] += '*'
                res_info.append(0)
            if (change_plate and i > 0) or i == len(cla_data) - 1:
                self.parse_rule(anno['label'])
                anno = self.modify_lpr_by_rule(anno)

    def parse_rule(self, label):
        """ Parse plate rule. """

        self.rule = []
        if 'plate_rule' in self.param.keys():
            self.rule = copy.deepcopy(self.param['plate_rule'])

        num = len(label)
        for i in range(len(self.rule) - 1, -1, -1):
            if len(self.rule[i]) != num:
                self.rule.pop(i)

    def modify_lpr_by_rule(self, anno):
        """ Modify lpr by plate rule. """

        rule_score_max = 0
        rule_lpr_max = ''
        rule_lpr_score_max = []
        for rule in self.rule:
            if len(rule) != len(anno['label']):
                continue
            rule_score = 0
            rule_lpr = ''
            rule_lpr_score = []
            confs = anno['labinfo']['ClassifierMobileNet']

            for i, char in enumerate(anno['label']):
                if (str.isdigit(char) and rule[i] is 'N') or \
                   (str.isalpha(char) and rule[i] is 'A'):
                    rule_score += confs[i][char.lower()]
                    rule_lpr += char
                    rule_lpr_score.append(confs[i][char.lower()])
                else:
                    sorted_info = sorted(confs[i].items(),
                                         key=operator.itemgetter(1),
                                         reverse=True)
                    if rule[i] is 'N':
                        done = False
                        for char_, conf in sorted_info:
                            if char_ == 'unknown':
                                continue
                            if str.isdigit(char_) and \
                                    conf > self.param['min_conf']:
                                rule_score += conf
                                rule_lpr += char_.upper()
                                rule_lpr_score.append(conf)
                                done = True
                                break
                        if done is False:
                            rule_score += 0
                            rule_lpr += '*'
                            rule_lpr_score.append(0)
                    elif rule[i] is 'A':
                        done = False
                        for char_, conf in sorted_info:
                            if char_ == 'unknown':
                                continue
                            if str.isalpha(char_) and \
                                    conf > self.param['min_conf']:
                                rule_score += conf
                                rule_lpr += char_.upper()
                                rule_lpr_score.append(conf)
                                done = True
                                break
                        if done is False:
                            rule_score += 0
                            rule_lpr += '*'
                            rule_lpr_score.append(0)
            if rule_score > rule_score_max:
                rule_score_max = rule_score
                rule_lpr_max = rule_lpr
                rule_lpr_score_max = rule_lpr_score
        if rule_score_max > 0:
            anno['label'] = rule_lpr_max
            anno['confidence'] = rule_score_max / len(rule_lpr_max)
            anno['labinfo']['DeterminatorLpr'] = rule_lpr_score_max
        return(anno)


class DeterminatorRefineLpr(determinator_base.DeterminatorBase):
    """ Refine lpr by previous results.
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(DeterminatorRefineLpr, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.pre_data = []

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        if not self.param['continuous']:
            self.results = copy.deepcopy(input_data)
            self.del_labinfo()
            self.uniform_results()
            return

        if self.pre_data == []:
            self.pre_data = copy.deepcopy(input_data)
            self.results = copy.deepcopy(input_data)
            self.del_labinfo()
            self.uniform_results()
            return

        for di, data in enumerate(input_data):
            pre_data = self.pre_data[di]

            match_result = self.match_by_track_id(
                data['annotations'],
                pre_data['annotations'])

            self.results.append(self.refine_lpr(
                data,
                pre_data,
                match_result))

        self.pre_data = copy.deepcopy(input_data)
        self.del_labinfo()
        self.uniform_results()

    def del_labinfo(self):
        """ Delete labinfo.
        """

        for res in self.results:
            for obj in res['annotations']:
                del obj['labinfo']

    def refine_lpr(
            self,
            data,
            pre_data,
            match_result):
        """ Refine lpr.
        """

        anno = data['annotations']
        pre_anno = pre_data['annotations']
        for idx1 in range(len(anno)):
            if idx1 not in match_result['match_index_1'] or \
                'lpr' not in anno[idx1].keys() or \
                    anno[idx1]['lpr'] is None:
                continue
            j = match_result['match_index_1'].index(idx1)
            idx2 = match_result['match_index_2'][j]
            lpr_now = anno[idx1]['lpr']
            score_now = anno[idx1]['labinfo']['DeterminatorLpr']
            if 'lpr' not in pre_anno[idx2].keys() or \
                    pre_anno[idx2]['lpr'] is None:
                continue
            lpr_pre = pre_anno[idx2]['lpr']
            score_pre = pre_anno[idx2]['labinfo']['DeterminatorLpr']

            # FIXME: length of lpr and score are not the same
            if len(lpr_pre) != len(score_pre):
                len_final = min(len(lpr_pre), len(score_pre))
                lpr_pre = lpr_pre[:len_final]
                score_pre = score_pre[:len_final]
            if len(lpr_now) != len(score_now):
                len_final = min(len(lpr_now), len(score_now))
                lpr_now = lpr_now[:len_final]
                score_now = score_now[:len_final]

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

            if len(lpr_max) > self.param['max_bit']:
                if np.mean(score_pre) > np.mean(score_now):
                    lpr = lpr_pre
                    score = score_pre
                else:
                    lpr = lpr_now
                    score = score_now
            elif match_max >= self.param['match_bit_thre']:
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

            anno[idx1]['lpr'] = lpr
            anno[idx1]['labinfo']['DeterminatorLpr'] = score
            anno[idx1]['confidence'] = np.mean(score)

        return data

    def match_by_track_id(
            self,
            detection_result_1,
            detection_result_2):
        """ Bounding boxes matching according to track_id.

        @param detection_result_1: annotations in detection result
        @param detection_result_2: annotations in detection result

        @return match_result: {
            'match_index_1': list of bounding box index in detection_result_1
            'match_index_2': list of bounding box index in detection_result_2
        }

        """

        match_result = {
            'match_index_1': [],
            'match_index_2': []
        }

        number_1 = len(detection_result_1)
        number_2 = len(detection_result_2)
        if number_1 == 0 or number_2 == 0:
            return(match_result)

        for i in range(number_1):
            track_id_1 = detection_result_1[i]['track_id']
            for j in range(number_2):
                track_id_2 = detection_result_2[j]['track_id']
                if track_id_1 < 0:
                    continue
                if track_id_1 == track_id_2:
                    match_result['match_index_1'].append(i)
                    match_result['match_index_2'].append(j)

        return(match_result)


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
