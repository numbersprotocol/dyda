import os
import copy
import cv2
import numpy as np
from dt42lab.core import tinycv
from dt42lab.core import lab_tools
from dt42lab.core import image
from dt42lab.core import tools
from dyda.core import data_converter_base
from scipy.ndimage.interpolation import map_coordinates


class LowerUpperConverter(data_converter_base.ConverterBase):
    """ Convert a list of string to lower or upper cases """

    def __init__(self, dyda_config_path='', param=None):
        super(LowerUpperConverter, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.results = []

    def main_process(self):
        """ main_process of LowerUpperConverter """

        for str_data in self.input_data:
            if self.param["conversion"] == "lower":
                self.results.append(str_data.lower())
            elif self.param["conversion"] == "upper":
                self.results.append(str_data.upper())
            else:
                self.logger.debug(
                    "wrong conversion setting, should be lower or upper"
                )

    def reset_results(self):
        """ reset_results for LowerUpperConverter """
        self.results = []


class SnapshotFnameConverter(data_converter_base.ConverterBase):
    """ Create folder and filenames from snapshot_fnames"""

    def __init__(self, dyda_config_path='', param=None):
        super(SnapshotFnameConverter, self).__init__(
            dyda_config_path=dyda_config_path
        )

    def main_process(self):
        """ main_process of SnapshotFnameConverter """

        self.results = []
        for i in range(0, len(self.input_data)):
            folder = os.path.dirname(self.input_data[i])
            filename = os.path.basename(self.input_data[i])
            self.results.append({
                "filename": filename,
                "folder": folder
            })
        if self.unpack_single_list:
            self.unpack_single_results()


class PathLabelConverter(data_converter_base.ConverterBase):
    """ Get labels from data path """

    def __init__(self, dyda_config_path='', param=None):
        super(PathLabelConverter, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.output_data = []
        self.results = []

    def main_process(self):
        """ main_process of PathLabelConverter """

        # If this is run after FrameReader
        if isinstance(self.input_data, dict):
            input_data = copy.deepcopy(self.input_data)['data_path']
        elif isinstance(self.input_data[0], dict):
            input_data = copy.deepcopy(self.input_data[0])['data_path']
        # If this is run on external data list
        elif isinstance(self.input_data, list):
            input_data = self.input_data

        for full_path in input_data:
            labels = os.path.dirname(full_path).split('/')
            self.results.append(labels[0 - self.param['level']])

    def reset_results(self):
        """ reset_results for PathLabelConverter"""
        self.results = []


class PathLabelConverterLab(PathLabelConverter):
    """ Get labels from data path and assign it to lab format"""

    def __init__(self, dyda_config_path='', param=None):
        super(PathLabelConverterLab, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):

        super(PathLabelConverterLab, self).main_process()
        labels = copy.deepcopy(self.results)
        self.results = []
        for label in labels:
            self.results.append(
                lab_tools.output_pred_classification("", 1.0, label)
            )
        self.unpack_single_results()


class NP2ColorMapConverter(data_converter_base.ConverterBase):
    """ This component accepts input_data which is a numpy array or a list of
        numpy arrays with range from 0-255 and convert them to color map(s) """

    def __init__(self, dyda_config_path='', param=None):
        super(NP2ColorMapConverter, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)
        self.norm_min = 0
        if "norm_min" in self.param.keys():
            if self.param["norm_min"] < 0:
                self.logger.warning(
                    "Cannot normalize to a range less than 0, will not use the"
                    " config value norm_min but 0 as the norm_min."
                )
            else:
                self.norm_min = self.param["norm_min"]
        self.norm_max = 255
        if "norm_max" in self.param.keys():
            if self.param["norm_max"] > 255.0:
                self.logger.warning(
                    "Cannot normalize to a range larger than 255, will not use"
                    " the config value norm_max but 255 as the norm_max."
                )
            else:
                self.norm_max = self.param["norm_max"]
        self.color_map = "COLORMAP_JET"
        if "color_map" in self.param.keys():
            self.color_map = self.param["color_map"]

        self.results = {
            "norm": float((self.norm_max - self.norm_min)/255.0),
            "max": self.norm_max,
            "min": self.norm_min
        }

    def main_process(self):
        """ main_process of NP2ColorMapConverter """

        input_data = self.uniform_input()
        cv2_cm = image.get_cv2_color_map(color_map=self.color_map)

        for _im in input_data:
            im = _im * self.results["norm"] + self.norm_min
            # ColorMap only support one channel or three channels
            # of unsigned integer (0 to 255)
            im = cv2.applyColorMap(im.astype(np.uint8), cv2_cm)
            self.output_data.append(im)

        self.uniform_output()


class YTECDataConverter2(data_converter_base.ConverterBase):
    """ New YTEC data converter based on FR dt42-dyda/issues/78 """

    def __init__(self, dyda_config_path='', param=None):
        super(YTECDataConverter2, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.output_data = []
        self.results = []
        self.rect = tinycv.Rect()

    def main_process(self):
        """ main_process of YTECDataConverter2 """

        img = self.input_data[0]
        anno = self.external_metadata

        try:
            self.rect.reset_loc([
                anno["top"], anno["bottom"], anno["left"], anno["right"]
            ])
        except BaseException:
            self.terminate_flag = True
            self.logger.error("Fail to reset loc")
            return False

        rot_angle = anno["degree"]
        rot_img = tinycv.rotate(img, rot_angle)
        cropped = tinycv.crop_img_rect_rgb(rot_img, self.rect)
        self.output_data = [cropped]

        return True

    def reset_results(self):
        """ reset_results of YTECDataConverter"""
        self.results = []


class YTECDataConverter3(data_converter_base.ConverterBase):
    """ Converter2 and 3 both accept json as the format eng team required
        YTECDataConverter2 - use external data and metadata to convert
        YTECDataConverter3 - accept a json including data path and metadata
    """

    def __init__(self, dyda_config_path='', param=None):
        super(YTECDataConverter3, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.results = {}
        self.rect = tinycv.Rect()

    def main_process(self):
        """ main_process of YTECDataConverter """

        if len(self.input_data) != 2:
            self.terminate_flag = True
            self.logger.error(
                "input_data should be a list of [image_array, metadata]."
            )
            return False

        if isinstance(self.input_data[0], np.ndarray):
            img_data = copy.deepcopy(self.input_data[0])
        elif isinstance(self.input_data[0], list):
            img_data = copy.deepcopy(self.input_data[0][0])
            if len(self.input_data[0]) > 1:
                self.logger.warning("Multiple input image arrays detected.")

        if isinstance(self.input_data[1], dict):
            metadata = self.input_data[1]
        elif isinstance(self.input_data[0], list):
            metadata = self.input_data[1][0]
            if len(self.input_data[1]) > 1:
                self.logger.warning("Multiple input metadata detected.")

        if "metadata" not in metadata.keys():
            self.terminate_flag = True
            self.logger.error("annotations key is missing in metadata.")
            return False

        anno = metadata["metadata"]

        try:
            self.rect.reset_loc([
                anno["top"], anno["bottom"], anno["left"], anno["right"]
            ])

        except BaseException:
            self.terminate_flag = True
            self.logger.error("Fail to set location info")
            return False

        rot_angle = 0
        if "degree" in anno.keys():
            rot_angle = anno["degree"]

        if rot_angle != 0:
            rot_img = tinycv.rotate(img_data, rot_angle)
        else:
            rot_img = img_data

        cropped = tinycv.crop_img_rect_rgb(rot_img, self.rect)
        self.output_data = [cropped]
        shape = cropped.shape
        self.results = lab_tools.output_pred_classification(
            "", 1.0, "raw", img_size=[shape[1], shape[0]]
        )
        self.results["folder"] = os.path.join(
            copy.deepcopy(self.snapshot_folder), "output_data"
        )
        self.results["filename"] = copy.deepcopy(self.metadata[0]) + ".jpg.0"
        self.results["annotations"][0]["top"] = 0
        self.results["annotations"][0]["left"] = 0
        self.results["annotations"][0]["bottom"] = shape[0]
        self.results["annotations"][0]["right"] = shape[1]
        self.results["annotations"][0]["rot_angle"] = 0

        return True

    def reset_results(self):
        """ reset_results of YTECDataConverter"""
        self.results = {}


class YTECDataConverter(data_converter_base.ConverterBase):
    """ Use external metadata to crop and rotate data """

    def __init__(self, dyda_config_path='', param=None):
        super(YTECDataConverter, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.results = {}
        self.rect = tinycv.Rect()

    def main_process(self):
        """ main_process of YTECDataConverter """

        if len(self.input_data) != 2:
            self.terminate_flag = True
            self.logger.error(
                "input_data should be a list of [image_array, metadata]."
            )
            return False

        if isinstance(self.input_data[0], np.ndarray):
            img_data = copy.deepcopy(self.input_data[0])
        elif isinstance(self.input_data[0], list):
            img_data = copy.deepcopy(self.input_data[0][0])
            if len(self.input_data[0]) > 1:
                self.logger.warning("Multiple input image arrays detected.")

        if isinstance(self.input_data[1], dict):
            metadata = self.input_data[1]
        elif isinstance(self.input_data[0], list):
            metadata = self.input_data[1][0]
            if len(self.input_data[1]) > 1:
                self.logger.warning("Multiple input metadata detected.")

        if "annotations" not in metadata.keys():
            self.terminate_flag = True
            self.logger.error("annotations key is missing in metadata.")
            return False

        if len(metadata["annotations"]) < 1:
            self.terminate_flag = True
            self.logger.error("No annotation found in metadata.")
            return False

        anno = metadata["annotations"][0]

        try:
            self.rect.reset_loc([
                anno["top"], anno["bottom"], anno["left"], anno["right"]
            ])

        except BaseException:
            self.terminate_flag = True
            self.logger.error("Fail to set location info")
            return False

        rot_angle = 0
        if "rot_angle" in anno.keys():
            rot_angle = anno["rot_angle"]

        if rot_angle != 0:
            rot_img = tinycv.rotate(img_data, rot_angle)
        else:
            rot_img = img_data

        cropped = tinycv.crop_img_rect_rgb(rot_img, self.rect)
        self.output_data = [cropped]
        self.results = copy.deepcopy(metadata)
        self.results["folder"] = os.path.join(
            copy.deepcopy(self.snapshot_folder), "output_data"
        )
        shape = cropped.shape
        self.results["size"]["width"] = shape[1]
        self.results["size"]["height"] = shape[0]
        self.results["filename"] = copy.deepcopy(self.metadata[0]) + ".jpg.0"
        self.results["annotations"][0]["top"] = 0
        self.results["annotations"][0]["left"] = 0
        self.results["annotations"][0]["bottom"] = shape[0]
        self.results["annotations"][0]["right"] = shape[1]
        self.results["annotations"][0]["rot_angle"] = 0

        return True

    def reset_results(self):
        """ reset_results of YTECDataConverter"""
        self.results = {}


class IrConverter(data_converter_base.ConverterBase):
    """A ir image in list format is converted to a gray scale image array.

       @param height_ori: original height of ir image
       @param width_ori: original width of ir image
       @param height: height of converted image in gray scale
       @param width: width of converted image in gray scale

    """

    def __init__(self, dyda_config_path='', param=None):
        super(IrConverter, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):

        data = self.input_data[0]
        self.results = {}

        # resize from 1-D to 2-D
        data = np.resize(data, new_shape=(
            self.param['height_ori'], self.param['width_ori']))
        # rotate 180 degree
        data = np.rot90(data, 2)
        # interpolate to new size
        new_dims = []
        for original_length, new_length in zip(
                data.shape, (self.param['height'], self.param['width'])):
            new_dims.append(np.linspace(0, original_length - 1, new_length))
        coords = np.meshgrid(*new_dims, indexing='ij')
        data = map_coordinates(data, coords)
        self.output_data = data

    def post_process(self, out_folder_base):

        output_parent_folder = out_folder_base
        tools.check_dir(output_parent_folder)
        output_folder = os.path.join(
            output_parent_folder,
            self.__class__.__name__)
        tools.check_dir(output_folder)

        # image enhancement
        data = copy.deepcopy(self.output_data)
        min_value = data.min()
        max_value = data.max()
        ratio = 255 / (max_value - min_value)
        for i in range(self.param['height']):
            for j in range(self.param['width']):
                data[i][j] = (data[i][j] - min_value) * ratio

        # write image
        out_filename = os.path.join(
            output_folder,
            self.metadata[0] + '.png')
        image.save_img(data, fname=out_filename)
        log_info = '[IrConverter] save image to: {}'.format(out_filename)
        print(log_info)


class TimeScaleShiftConverter(data_converter_base.ConverterBase):
    """The detected results on rgb image are converted to auto-labeled
       detected results on other type image (ir or depth) by temporal shift,
       spatial scale and spatial shift.

       @param time_shift: frame index of other type image = frame index of
           rgb image A + time_shift
       @param txt_path: path of txt file in which each line contains
           corresponding points coordinate on source image A and
           destination image B in order of h_A, w_A, h_B, w_B

    """

    def __init__(self, dyda_config_path='', param=None):
        super(TimeScaleShiftConverter, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.param['scale_shift'] = tinycv.scale_shift_param_polyfit(
            self.param['points_path'])
        self.results_queue = []
        self.filename_queue = []

    def main_process(self):

        self.output_data = {}
        self.results = {}
        if len(self.input_data) > 0:
            detection = self.input_data[0]
            if 'annotations' in detection.keys():
                # spatial scale
                detection = lab_tools.resize_detection(
                    detection,
                    self.param['scale_shift'][0],
                    self.param['scale_shift'][2])

                # spatial shift
                detection = lab_tools.shift_detection(
                    detection,
                    shift_h=self.param['scale_shift'][1],
                    shift_w=self.param['scale_shift'][3],
                    height=self.param['height'],
                    width=self.param['width'])

                # delete object shows little part
                for i in range(len(detection['annotations']) - 1, -1, -1):
                    top = detection['annotations'][i]['top']
                    bottom = detection['annotations'][i]['bottom']
                    left = detection['annotations'][i]['left']
                    right = detection['annotations'][i]['right']
                    if right - left < self.param['size_thre'] or \
                       bottom - top < self.param['size_thre']:
                        detection['annotations'].pop(i)

                # refine detection by param
                if not self.param['folder'] == "":
                    detection['folder'] = self.param['folder']
                if not self.param['extension'] == "":
                    detection['filename'] = detection['filename'] + \
                        self.param['extension']

                # temporal shift
                self.results_queue.append(copy.deepcopy(detection))
                self.filename_queue.append(
                    copy.deepcopy(detection['filename']))
                time_shift = abs(self.param['time_shift'])
                if len(self.results_queue) > time_shift:
                    if self.param['time_shift'] > 0:
                        self.results = copy.deepcopy(self.results_queue[0])
                        self.results['filename'] = copy.deepcopy(
                            self.filename_queue[time_shift])
                    else:
                        self.results = copy.deepcopy(
                            self.results_queue[time_shift])
                        self.results['filename'] = copy.deepcopy(
                            self.filename_queue[0])
                    self.results_queue.pop(0)
                    self.filename_queue.pop(0)

    def post_process(self, out_folder_base):

        output_parent_folder = out_folder_base
        tools.check_dir(output_parent_folder)
        output_folder = os.path.join(
            output_parent_folder,
            self.__class__.__name__)
        tools.check_dir(output_folder)

        if 'filename' in self.results.keys():
            # write json
            if self.param['folder'] == "":
                self.results['folder'] = os.path.join(
                    output_parent_folder, 'IrConverter')
            out_filename = os.path.join(
                output_folder,
                tools.replace_extension(self.results['filename'], 'json'))
            tools.write_json(self.results, out_filename)
            log_info = '[TimeScaleShiftConverter] save json to: {}'.format(
                out_filename)
            print(log_info)


class VocImagePath2XmlPathConverter(data_converter_base.ConverterBase):
    """ Convert image path to xml path according to file structure
        defined by VOC dataset.
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(VocImagePath2XmlPathConverter, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input(self.input_data, 'str')

        for path in input_data:
            path = self.change_folder(path)
            path = self.change_extension(path)
            self.results.append(path)

        self.uniform_output()

    def change_folder(self, path):
        """ Change folder from JPEGImages to Annotations. """

        return path.replace("JPEGImages", "Annotations")

    def change_extension(self, path):
        """ Change extension from jpg to xml. """

        return tools.replace_extension(path, 'xml')
