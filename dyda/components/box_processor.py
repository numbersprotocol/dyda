
import copy
import numpy as np
from dt42lab.core import image
from dt42lab.core import tools
from dt42lab.core import tinycv
from dt42lab.core import lab_tools
from dt42lab.core import boxes
from dyda.core import box_processor_base
from scipy.ndimage.interpolation import map_coordinates


class ShrinkBoxProcessor(box_processor_base.BoxProcessorBase):
    """ Shrink bounding box to the specified percentage """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(ShrinkBoxProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.shrink_perc = 0.05
        if "shrink_perc" in self.param.keys():
            self.shrink_perc = self.param["shrink_perc"]
        self.shift_to_pad = False
        if "shift_to_pad" in self.param.keys():
            self.shift_to_pad = self.param["shift_to_pad"]

    def main_process(self):
        """ Main function of dyda component. """

        if isinstance(self.input_data, dict):
            self.logger.debug(
                "Input data is dict"
            )
            if not lab_tools.if_result_match_lab_format(self.input_data):
                self.terminate_flag = False
                self.logger.error(
                    "Can only accept lab format dictionary"
                )
                return False
            input_data = self.input_data
        else:
            self.logger.debug(
                "Input data is other type (expect ndarray)"
            )
            if isinstance(self.input_data, list):
                if len(self.input_data) == 1:
                    input_img = self.input_data[0]
                else:
                    input_img = self.input_data[0]
                    self.logger.warning(
                        "More than one input img array detected"
                    )
            else:
                input_img = self.input_data
            width = input_img.shape[1]
            height = input_img.shape[0]
            input_data = lab_tools.output_pred_classification(
                "", 1.0, "roi"
            )
            input_data["annotations"][0]["top"] = 0
            input_data["annotations"][0]["bottom"] = height
            input_data["annotations"][0]["left"] = 0
            input_data["annotations"][0]["right"] = width

        self.results = boxes.shrink_boxes(
            copy.deepcopy(input_data), self.shrink_perc,
            shift_to_pad=self.shift_to_pad
        )


class ResizeBoxProcessor(box_processor_base.BoxProcessorBase):
    """ Resize annotations to output corresponding bounding boxes
        for resized images.
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(ResizeBoxProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        resize_info = copy.deepcopy(self.input_data[0])
        input_data = copy.deepcopy(self.input_data[1])

        if not isinstance(input_data, list):
            input_data = [input_data]
            resize_info = [resize_info]
            package = True
        else:
            package = False

        for i, data in enumerate(input_data):
            new_size = resize_info[i]['new_size']
            ori_size = resize_info[i]['ori_size']
            self.results.append(lab_tools.resize_detection(
                data,
                new_size['height'] / ori_size['height'],
                new_size['width'] / ori_size['width']))
            data['size'] = new_size

        if package:
            self.results = self.results[0]


class CropBoxProcessor(box_processor_base.BoxProcessorBase):
    """ Crop annotations according to roi given by config.
        Output corresponding bounding boxes for cropped images.
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(CropBoxProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.roi = tinycv.Rect([
            self.param['top'],
            self.param['bottom'],
            self.param['left'],
            self.param['right']])

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input(self.input_data)

        for data in input_data:
            self.results.append(self.crop_bb(data))

        self.uniform_output()

    def crop_bb(self, data):
        """ Crop bounding boxes to roi given by config. """

        # check if roi exceeds image size
        width = data['size']['width']
        height = data['size']['height']
        if not (isinstance(width, int) and isinstance(height, int)):
            self.logger.warning("Can not check boundary "
                                "since no image size is given.")
        else:
            if self.roi.r >= width or self.roi.b >= height:
                self.terminate_flag = True
                self.logger.error("Roi exceeds image size.")
        data['size']['width'] = self.roi.w
        data['size']['height'] = self.roi.h

        # crop bounding boxes
        for idx in range(len(data['annotations']) - 1, -1, -1):
            obj = data['annotations'][idx]
            bb = tinycv.Rect([
                obj['top'],
                obj['bottom'],
                obj['left'],
                obj['right']])

            # check if object in roi
            overlap_ratio = lab_tools.calculate_overlap_ratio(
                [self.roi.t, self.roi.b, self.roi.l, self.roi.r],
                [bb.t, bb.b, bb.l, bb.r],
                denominator_type='area_2')
            if overlap_ratio < self.param['overlap_threshold']:
                data['annotations'].pop(idx)
                continue

            # update bounding box of object
            obj['top'] = tinycv.check_boundary_limit(
                bb.t - self.roi.t, self.roi.h)
            obj['left'] = tinycv.check_boundary_limit(
                bb.l - self.roi.l, self.roi.w)
            obj['bottom'] = tinycv.check_boundary_limit(
                bb.b - self.roi.t, self.roi.h)
            obj['right'] = tinycv.check_boundary_limit(
                bb.r - self.roi.l, self.roi.w)

        return(data)


class UnpadBoxProcessor(box_processor_base.BoxProcessorBase):
    """ Reverse detection results from padded image to unpadded image """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(UnpadBoxProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        if 'type' not in self.param.keys():
            self.type = 'to_unpadded'
        else:
            self.type = self.param['type']
        if 'use_external_meta' not in self.param.keys():
            self.use_external_meta = False
        else:
            self.use_external_meta = self.param['use_external_meta']

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()

        pad_info = copy.deepcopy(self.input_data[0])
        input_data = copy.deepcopy(self.input_data[1])

        if not isinstance(input_data, list):
            input_data = [input_data]
            pad_info = [pad_info]
            package = True
        else:
            package = False

        if self.use_external_meta:
            pad_info = self.external_metadata['UnpadBoxProcessor']

        for i, data in enumerate(input_data):
            if self.type == 'to_padded':
                pad_info[i] = self.modify_pad_info(pad_info[i])
            self.results.append(lab_tools.shift_boxes(
                data,
                (-pad_info[i]['shift_w'], -pad_info[i]['shift_h']),
                (pad_info[i]['ori_w'], pad_info[i]['ori_h'])))
            self.results[-1]['size']['width'] = pad_info[i]['ori_w']
            self.results[-1]['size']['height'] = pad_info[i]['ori_h']

        if package:
            self.results = self.results[0]

    def modify_pad_info(self, pad_info):
        """ Modify pad_info from to_unpadded to to_padded. """

        _pad_info = copy.deepcopy(pad_info)
        _pad_info['shift_w'] = pad_info['shift_w'] * -1
        _pad_info['shift_h'] = pad_info['shift_h'] * -1
        s = max(pad_info['ori_w'], pad_info['ori_h'])
        _pad_info['ori_w'] = s
        _pad_info['ori_h'] = s
        return _pad_info


class TransformBoxProcessor(box_processor_base.BoxProcessorBase):
    """ Transform detection results by transformation matrix """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(TransformBoxProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()

        trans_info = copy.deepcopy(self.input_data[0])
        input_data = self.uniform_input(self.input_data[1])

        for i, data in enumerate(input_data):
            self.results.append(self.transform_anno(
                data, trans_info))

        self.uniform_output()

    def transform_anno(self, data, trans_info):
        """ Transform annotations. """

        for anno in data['annotations']:
            bb = tinycv.Rect([
                anno['top'], anno['bottom'], anno['left'], anno['right']
            ])
            lt = self.transform_point(bb.l, bb.t, trans_info)
            lb = self.transform_point(bb.l, bb.b, trans_info)
            rt = self.transform_point(bb.r, bb.t, trans_info)
            rb = self.transform_point(bb.r, bb.b, trans_info)
            anno['left'] = min(
                max(min([lt[0], lb[0], rt[0], rb[0]]), 0),
                trans_info['size']['width'] - 1)
            anno['right'] = min(
                max(max([lt[0], lb[0], rt[0], rb[0]]), 1),
                trans_info['size']['width'])
            anno['top'] = min(
                max(min([lt[1], lb[1], rt[1], rb[1]]), 0),
                trans_info['size']['height'] - 1)
            anno['bottom'] = min(
                max(max([lt[1], lb[1], rt[1], rb[1]]), 1),
                trans_info['size']['height'])
        data['size'] = trans_info['size']
        return data

    def transform_point(self, x, y, trans_info):
        """ Transform point. """

        if self.param['reverse']:
            point = np.matmul(
                np.linalg.inv(
                    trans_info['trans_mat']), [
                    x, y, 1])
        else:
            point = np.matmul(trans_info['trans_mat'], [x, y, 1])
        point = point / point[2]
        return point[:2].astype(int)


class SplitBoxProcessor(box_processor_base.BoxProcessorBase):
    """ Split bounding boxes """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(SplitBoxProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input(self.input_data)

        for data in input_data:
            self.results.append(lab_tools.split_detection(data, True))

        self.uniform_output()


class UnmergeBoxProcessor(box_processor_base.BoxProcessorBase):
    """ Unmerge dict to four dicts"""

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(UnmergeBoxProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input(self.input_data)

        for data in input_data:
            self.results.append(lab_tools.split_detection(data, False))

        self.uniform_output()


class SquareExtendBoxProcessor(box_processor_base.BoxProcessorBase):
    """ Extend bounding boxes to square """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(SquareExtendBoxProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input(self.input_data)

        for data in input_data:
            self.results.append(lab_tools.square_extend_in_json(data))

        self.uniform_output()


class CatAnnotationsBoxProcessor(box_processor_base.BoxProcessorBase):
    """ Cat annotations from list of dicts in lab format to list of single
        dict in lab format"""

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(CatAnnotationsBoxProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        if 'base_index' not in self.param.keys():
            self.param['base_index'] = 0
        self.del_key = []
        if 'del_key' in self.param.keys():
            self.del_key = self.param['del_key'].split(',')

    def main_process(self):
        """ Main function of dyda component. """

        default_results = {
            "folder": None,
            "filename": None,
            "timestamp": self.metadata[0],
            "size": {
                "width": None,
                "height": None
            },
            "annotations": []
        }

        # return default results when pipeline status is 0
        if self.pipeline_status == 0 and self.is_empty_input(self.input_data):
            self.results = copy.deepcopy(default_results)
            return

        # assign default results when input_data is empty list
        for di, data in enumerate(self.input_data):
            if data == []:
                self.input_data[di] = copy.deepcopy(default_results)

        input_data = self.uniform_input(self.input_data)

        if self.param['base_index'] >= len(input_data):
            self.terminate_flag = True
            self.logger.error("Parameter base_index is not valid.")

        for ci in range(len(input_data) - 1, -1, -1):
            if not input_data[ci]:
                input_data.pop(ci)

        self.results = input_data[self.param['base_index']]
        if isinstance(self.results, list) and len(self.results) == 1:
            self.results = self.results[0]
        for ci, data in enumerate(input_data):
            if ci == self.param['base_index']:
                continue
            if isinstance(data, list) and len(data) == 1:
                data = data[0]
            self.results['annotations'].extend(data['annotations'])

        for obj in self.results['annotations']:
            for key in self.del_key:
                del obj[key]

        self.uniform_output()

    def is_empty_input(self, input_data):
        """ Check if input data is empty.
        """

        if tools.is_empty_list(input_data) or tools.is_empty_dict(input_data):
            return True
        else:
            return False


class CombineCarLprBoxProcessor(box_processor_base.BoxProcessorBase):
    """ Combine car detection and lpr classification results."""

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(CombineCarLprBoxProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        car_data = self.uniform_input(self.input_data[0])

        # assign default results when pipeline status is 0
        if self.pipeline_status == 0 and self.is_empty_input(
                self.input_data[1]):
            lpr_data = [{
                "folder": None,
                "filename": None,
                "timestamp": self.metadata[0],
                "size": {
                    "width": None,
                    "height": None
                },
                "annotations": []
            }]
        else:
            lpr_data = self.uniform_input(self.input_data[1])

        for ci, data in enumerate(car_data):
            for anno in data['annotations']:
                if 'lpr' not in anno.keys():
                    anno['lpr'] = None

        for ci, data in enumerate(car_data):
            self.results.append(self.combine_car_lpr(data, lpr_data[ci]))

        self.uniform_output()

    def combine_car_lpr(self, car_data, lpr_data):
        """ Combine car and lpr detection results.
        """

        if 'annotations' not in lpr_data.keys() or \
                len(car_data['annotations']) == 0 or \
                len(lpr_data['annotations']) == 0:
            return car_data

        overlap_ratio_all = lab_tools.calculate_overlap_ratio_all(
            car_data['annotations'],
            lpr_data['annotations'],
            'area_2')

        # one to one match
        car_num, lpr_num = overlap_ratio_all.shape
        while overlap_ratio_all.max() > self.param['overlap_ratio_th']:
            index = np.argwhere(overlap_ratio_all == overlap_ratio_all.max())
            i = index[0, 0]
            j = index[0, 1]
            car_data['annotations'][i]['lpr'] = \
                lpr_data['annotations'][j]['label']
            car_data['annotations'][i]['lpr_confidence'] = \
                lpr_data['annotations'][j]['confidence']
            car_data['annotations'][i]['labinfo'] = \
                lpr_data['annotations'][j]['labinfo']
            overlap_ratio_all[i, :] = -1
            overlap_ratio_all[:, j] = -1

        return car_data

    def is_empty_input(self, input_data):
        """ Check if input data is empty.
        """

        if tools.is_empty_list(input_data) or tools.is_empty_dict(input_data):
            return True
        else:
            return False


class ExtendBoxProcessor(box_processor_base.BoxProcessorBase):
    """ Extend bounding boxes according to config"""

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(ExtendBoxProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input(self.input_data)

        for data in input_data:
            img_h = data['size']['height']
            img_w = data['size']['width']
            for anno in data['annotations']:
                bb = tinycv.Rect([
                    anno['top'], anno['bottom'], anno['left'], anno['right']
                ])
                self.get_extension(bb.h, bb.w)
                anno['top'] = max(0, bb.t - self.top_extension)
                anno['bottom'] = min(img_h, bb.b + self.bottom_extension)
                anno['left'] = max(0, bb.l - self.left_extension)
                anno['right'] = min(img_w, bb.r + self.right_extension)
            self.results.append(data)

        self.uniform_output()

    def get_extension(self, bb_h, bb_w):
        """ Turn relative extension to absolute extension. """

        if abs(self.param['top_extension']) < 1:
            self.top_extension = int(self.param['top_extension'] * bb_h)
        else:
            self.top_extension = self.param['top_extension']
        if abs(self.param['bottom_extension']) < 1:
            self.bottom_extension = int(self.param['bottom_extension'] * bb_h)
        else:
            self.bottom_extension = self.param['bottom_extension']
        if abs(self.param['left_extension']) < 1:
            self.left_extension = int(self.param['left_extension'] * bb_w)
        else:
            self.left_extension = self.param['left_extension']
        if abs(self.param['right_extension']) < 1:
            self.right_extension = int(self.param['right_extension'] * bb_w)
        else:
            self.right_extension = self.param['right_extension']


class UpdateInRotationBoxProcessor(box_processor_base.BoxProcessorBase):
    """ Update bounding boxes in rotation. """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(UpdateInRotationBoxProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        default_result = {
            'folder': '',
            'filename': '',
            'size': {'width': None, 'height': None},
            'annotations': []
        }
        self.pre_results = [copy.deepcopy(default_result) for i in range(
            self.param['rotation_num'])]

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        channel_index = self.input_data[0]['channel_index']
        if isinstance(self.input_data[1], list):
            if len(self.input_data[1]) == 1:
                data = copy.deepcopy(self.input_data[1][0])
            else:
                self.logger.warning(
                    "Multiple input_data[1] detected"
                )
        else:
            data = copy.deepcopy(self.input_data[1])

        if not lab_tools.if_result_match_lab_format(data):
            self.terminate_flag = True
            self.logger.error("Input_data should be a lab-format dict.")

        self.pre_results[channel_index] = data
        self.results = copy.deepcopy(self.pre_results)


class SelectInRotationBoxProcessor(box_processor_base.BoxProcessorBase):
    """The dicts in self.input_data are output in rotation.
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(SelectInRotationBoxProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.counter = 0

    def main_process(self):
        """ Main function of dyda component. """

        input_data = self.uniform_input(self.input_data)
        if len(input_data) != self.param['rotation_num']:
            self.terminate_flag = True
            self.logger.error("Length of input_data should match "
                              "param rotation_num")

        self.results = input_data[self.counter]

        self.counter += 1
        if self.counter == self.param['rotation_num']:
            self.counter = 0


class SelectByIdBoxProcessor(box_processor_base.BoxProcessorBase):
    """ Select and output one dict in self.input_data[1]
        according to channel id given in self.input_data[0].
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(SelectByIdBoxProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        channel_index = self.input_data[0]['channel_index']
        self.results = copy.deepcopy(self.input_data[1][channel_index])
