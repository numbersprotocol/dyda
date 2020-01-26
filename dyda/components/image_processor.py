import cv2
import copy
import numpy as np
from dyda_utils import image
from dyda_utils import lab_tools
from dyda_utils import tinycv
from dyda.core import image_processor_base


class RotateImageProcessor(image_processor_base.ImageProcessorBase):
    """ Simple image rotate processor """

    def __init__(self, dyda_config_path=''):
        """ __init__ of RotateImageProcessor """

        super(RotateImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

        self.rot_direction = "ccw"
        if "rot_direction" in self.param.keys():
            self.rot_direction = self.param["rot_direction"]
        if self.rot_direction not in ["cw", "ccw"]:
            self.logger.error(
                "Wrong direction, should be cw or ccw"
            )

    def main_process(self):
        """ define main_process of dyda component """

        input_data = self.uniform_input()
        self.output_data = input_data

        for i in range(0, len(self.output_data)):
            self.output_data[i] = tinycv.rotate_ccw_opencv(
                self.output_data[i], direction=self.rot_direction
            )

        self.uniform_output()


class DirAlignImageProcessor(image_processor_base.ImageProcessorBase):
    """ allign all images to the specified direction """

    def __init__(self, dyda_config_path=''):
        """ __init__ of DirAlignImageProcessor """

        super(DirAlignImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

        self.chosen_dir = "vertical"
        if "chosen_direction" in self.param.keys():
            self.chosen_dir = self.param["chosen_direction"]

        self.rotate_dir = "ccw"
        self.rot_angle = 0
        if "rotate_direction" in self.param.keys():
            self.rotate_dir = self.param["rotate_direction"]
        if self.rotate_dir == "ccw":
            self.rot_angle = -90
        else:
            self.rot_angle = 90

    def main_process(self):
        """ define main_process of dyda component """
        self.output_data = copy.deepcopy(self.input_data)
        self.results = []
        if not isinstance(self.output_data, list):
            self.output_data = [self.output_data]

        for i in range(0, len(self.output_data)):
            data_matrix = self.output_data[i]
            shape = data_matrix.shape
            width = shape[1]
            height = shape[0]
            angle = 0

            if self.chosen_dir == "vertical":
                if width > height:
                    self.output_data[i] = tinycv.rotate_ccw(
                        self.output_data[i], direction=self.rotate_dir
                    )
                    width = shape[0]
                    height = shape[1]
                    angle = self.rot_angle

            elif self.chosen_dir == "horizontal":
                if height > width:
                    self.output_data[i] = tinycv.rotate_ccw(
                        self.output_data[i], direction=self.rotate_dir
                    )
                    width = shape[0]
                    height = shape[1]
                    angle = self.rot_angle

            self.results.append(
                lab_tools.output_pred_classification(
                    "", -1, "aligned", img_size=[width, height]
                )
            )
            self.results[-1]["annotations"][0]["rot_angle"] = angle

        if self.unpack_single_list:
            self.unpack_single_output()
            self.unpack_single_results()


class PatchSysInfoImageProcessor(image_processor_base.ImageProcessorBase):
    """ patch system info such as ROI in metadata to images
        this component is for patching non-annotaion results
    """

    def __init__(self, dyda_config_path=''):
        """ __init__ of  """

        super(PatchSysInfoImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

        self.color = [255, 0, 0]
        if "patch_color" in self.param.keys():
            self.color = self.param["patch_color"]

        self.patch_meta_roi = False
        if "patch_external_meta_roi" in self.param.keys():
            self.patch_meta_roi = self.param["patch_external_meta_roi"]

        self.company_info = "All right reserved. DT42 confidencial"
        if "company_info" in self.param.keys():
            self.company_info = self.param["company_info"]

        self.attach_company_info = True
        if "attach_company_info" in self.param.keys():
            self.attach_company_info = int(self.param["attach_company_info"])

        self.keys_to_patch = []
        if "keys_to_patch" in self.param.keys():
            self.keys_to_patch = self.param["keys_to_patch"]

        self.line_width = 6
        if "patch_line_width" in self.param.keys():
            self.line_width = int(self.param["patch_line_width"])

        self.space = 10
        if "text_space" in self.param.keys():
            self.space = self.param["text_space"]

        self.font_scale = 1.0
        if "font_scale" in self.param.keys():
            self.font_scale = self.param["font_scale"]

    def main_process(self):
        """ define main_process of dyda component """

        self.output_data = copy.deepcopy(self.input_data[0])
        self.results = copy.deepcopy(self.input_data[1])
        self.output_data, self.results = \
            self.compare_single_inputs(self.output_data, self.results)
        if isinstance(self.output_data, np.ndarray):
            if isinstance(self.results, dict):
                self.output_data = [self.output_data]
                self.results = [self.results]
            else:
                self.logger.error("length of two input arrays do not match")
                self.terminate_flag = True
                return False
        elif len(self.output_data) != len(self.results):
            if len(self.results) == 0:
                self.logger.debug("No results found, skip patch")
                return True
            else:
                self.logger.error("length of two input arrays do not match")
                self.terminate_flag = True
                return False

        self.patch_image_arrays(self.output_data, self.results)

        if self.unpack_single_list:
            self.unpack_single_output()
            self.unpack_single_results()

        return True

    def patch_image_array(self, img_array, result):
        """ Patch image with bounding boxes """

        if self.patch_meta_roi:
            if "roi" in self.external_metadata:
                for roi in self.external_metadata["roi"]:
                    roi_bb = [
                        roi["top"],
                        roi["bottom"],
                        roi["left"],
                        roi["right"]
                    ]
                    if -1 in roi_bb:
                        continue
                    rect_roi = tinycv.Rect(roi_bb)
                    img_array = tinycv.patch_rect_img(
                        img_array, rect_roi,
                        color=self.color, line_width=self.line_width
                    )
            else:
                self.logger.warning(
                    "patch_meta_roi is on, but there is no"
                    " roi key found in external metadata,"
                )
        img_width = img_array.shape[1]
        img_height = img_array.shape[0]

        key_to_patch = ""
        counter = -1
        for key in result.keys():
            if key not in self.keys_to_patch:
                continue
            if counter < 0:
                key_to_patch = str(key) + ":" + "{}".format(result[key])
                counter = counter + 1
            else:
                key_to_patch = key_to_patch + "," + str(key) + ":"
                key_to_patch = key_to_patch + "{}".format(result[key])

        text_loc = (self.space, self.space * 3)

        img_array = tinycv.patch_text(
            img_array, key_to_patch, loc=text_loc, color=self.color,
            fontscale=self.font_scale
        )

        if self.attach_company_info:

            text_loc = (self.space, img_height - self.space)

            img_array = tinycv.patch_text(
                img_array, self.company_info, loc=text_loc, color=self.color,
                fontscale=self.font_scale
            )
        return True

    def patch_image_arrays(self, img_arrays, result_arrays):
        """ Patch all images with bounding boxes """

        for i in range(0, len(img_arrays)):
            img_array = img_arrays[i]
            result = result_arrays[i]
            self.patch_image_array(img_array, result)


class PatchImageProcessor(image_processor_base.ImageProcessorBase):
    """ Patch lab-format detection results from self.input_data[1]
        to image from self.input_data[0].
    """

    def __init__(self, dyda_config_path=''):
        """ __init__ of PatchImageProcessor """

        super(PatchImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

        self.color = [0, 0, 255]
        self.color_from = None
        if "patch_color" in self.param.keys():
            self.color = self.param["patch_color"]
        elif "color_from" in self.param.keys():
            self.color_from = self.param["color_from"]
        self.line_width = 6
        if "patch_line_width" in self.param.keys():
            self.line_width = int(self.param["patch_line_width"])
        self.space = 10
        if "text_space" in self.param.keys():
            self.space = self.param["text_space"]
        self.key_to_patch = "label"
        if "key_to_patch" in self.param.keys():
            self.key_to_patch = self.param["key_to_patch"]
        self.font_scale = 1.0
        if "font_scale" in self.param.keys():
            self.font_scale = self.param["font_scale"]
        self.check_lab_format = True
        if "check_lab_format" in self.param.keys():
            self.check_lab_format = self.param["check_lab_format"]
        self.customized_color = {}
        if "customized_color" in self.param.keys():
            self.customized_color = self.param["customized_color"]
        if not isinstance(self.customized_color, dict):
            self.logger.error(
                "customized_color has to be dict, "
                "fall back to empty dict."
            )
            self.customized_color = {}

    def main_process(self):
        """ define main_process of dyda component """

        self.output_data = copy.deepcopy(self.input_data[0])
        self.results = copy.deepcopy(self.input_data[1])
        self.output_data, self.results = \
            self.compare_single_inputs(self.output_data, self.results)
        if isinstance(self.output_data, np.ndarray):
            if isinstance(self.results, dict):
                self.output_data = [self.output_data]
                self.results = [self.results]
            else:
                self.logger.error("length of two input arrays do not match")
                self.terminate_flag = True
                return False
        elif len(self.output_data) != len(self.results):
            if len(self.results) == 0:
                self.logger.debug("No results found, skip patch")
                return True
            else:
                self.logger.error("length of two input arrays do not match")
                self.terminate_flag = True
                return False
        self.patch_image_arrays(self.output_data, self.results)
        return True

    def patch_image_array(self, img_array, result, color,
                          line_width, space):
        """ Patch image with bounding boxes """

        if self.check_lab_format:
            if not lab_tools.if_result_match_lab_format(result, loose=True):
                self.terminate_flag = True
                self.logger.error(
                    "Results format does not match lab definition"
                )
                return False

        keys = []
        if isinstance(self.key_to_patch, str):
            keys = [self.key_to_patch]
        elif isinstance(self.key_to_patch, list):
            keys = self.key_to_patch
        else:
            self.logger.error("Wrong key_to_patch type in dyda.config")

        for anno in result["annotations"]:
            if self.color_from is not None:
                import matplotlib.cm as cm
                color_n = cm.jet((anno[self.color_from] * 20) % 255)
                color = [int(color_n[0] * 255),
                         int(color_n[1] * 255),
                         int(color_n[2] * 255)]

            anno_keys = anno.keys()
            counter = -1
            key_to_patch = ''
            _color = color
            for key in keys:
                if key not in anno_keys:
                    continue
                if counter < 0:
                    key_to_patch = str(key) + ":" + str(anno[key])
                    counter = counter + 1
                else:
                    key_to_patch = key_to_patch + "," + str(key) + ":"
                    key_to_patch = key_to_patch + str(anno[key])
                if anno[key] in self.customized_color.keys():
                    _color = self.customized_color[anno[key]]

            rect = lab_tools.conv_lab_anno_to_rect(anno)
            img_array = tinycv.patch_rect_img(
                img_array, rect, color=_color, line_width=line_width
            )
            text_left = max(0, anno["left"] + space)
            text_top = max(0, anno["top"] + space)
            text_loc = (text_left, text_top)

            img_array = tinycv.patch_text(
                img_array, key_to_patch, loc=text_loc, color=_color,
                fontscale=self.font_scale
            )
        return True

    def patch_image_arrays(self, img_arrays, result_arrays):
        """ Patch all images with bounding boxes """

        for i in range(0, len(img_arrays)):
            img_array = img_arrays[i]
            result = result_arrays[i]
            self.patch_image_array(
                img_array, result, self.color, self.line_width, self.space
            )


class PadImageProcessor(image_processor_base.ImageProcessorBase):
    """  """

    def __init__(self, dyda_config_path=''):
        """ Initialization function of dyda component. """

        super(PadImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        for img in input_data:
            ori_h, ori_w = img.shape[:2]
            s = max(ori_h, ori_w)
            h = int((s - ori_h) / 2)
            w = int((s - ori_w) / 2)
            self.results.append({
                'ori_h': ori_h, 'ori_w': ori_w,
                'shift_h': h, 'shift_w': w})
            self.output_data.append(tinycv.image_padding(
                copy.deepcopy(img)))

        self.uniform_output()


class PaddingResizeImageProcessor(image_processor_base.ImageProcessorBase):
    """ Padding and resize image
        resize_to = (width, height)
        padding_to = center/bottom-right/top-left
    """

    def __init__(self, dyda_config_path=''):
        """ __init__ of PaddingResizeImageProcessor """

        super(PaddingResizeImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

        self.padding_to = "center"
        if "padding_to" in self.param.keys():
            self.padding_to = self.param["padding_to"]
        padding_types = ["center", "bottom-right", "top-left"]
        if self.padding_to not in padding_types:
            self.logger.warning(
                "Wrong padding_to: (%s), no padding applied" % self.padding_to
            )
            self.padding_to = None

        # resize_to should be [width, height]
        self.resize_to = None
        if "resize_to" in self.param.keys():
            self.resize_to = self.param["resize_to"]

        if not isinstance(self.resize_to, list):
            self.resize_to = None
            self.logger.warning(
                "Wrong resize_to variable type, it will not be applied."
            )
        self.reset_output_data()

    def reset_output_data(self):
        """ overwrite reset_output_data in dyda_base """
        self.output_data = []

    def main_process(self):
        """ define main_process of dyda component """

        if isinstance(self.input_data, list):
            for img in self.input_data:
                self.process_img(img)
        else:
            self.process_img(self.input_data)

        self.unpack_single_results()
        self.unpack_single_output()

    def process_img(self, img):

        self.append_info(img)
        if self.padding_to is not None:
            img = image.auto_padding(img, mode=self.padding_to)
        if self.resize_to is not None:
            img = image.resize_img(
                img, size=(self.resize_to[0], self.resize_to[1])
            )
        return self.output_data.append(img)

    def append_info(self, img):
        """ Append result to self.results,
            follow the format defined in ResizeImageProcessor
        """

        ori_img_shape = img.shape
        if self.resize_to is None:
            resize_to = (-1, -1)
        else:
            resize_to = self.resize_to
        append_dict = {
            "new_size": {
                "width": resize_to[0],
                "height": resize_to[1]
            },
            "ori_size": {
                "width": ori_img_shape[1],
                "height": ori_img_shape[0]
            },
            "padding_to": self.padding_to
        }
        self.results.append(append_dict)

    def reset_results(self):
        self.results = []


class MergeImageProcessor(image_processor_base.ImageProcessorBase):
    """ Merge 4 images to one. """

    def __init__(self, dyda_config_path=''):
        """ Initialization function of dyda component. """

        super(MergeImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = copy.deepcopy(self.input_data)

        package = False
        if len(input_data) == 4:
            if not isinstance(input_data[0], list):
                input_data = [input_data]
                package = True

        for imgs in input_data:
            if not len(imgs) == 4:
                self.terminate_flag = True
                self.logger.error("[MergeImageProcessor] input_data \
                    is not valid.")
            size = []
            for img in imgs:
                if not isinstance(img, np.ndarray):
                    self.terminate_flag = True
                    self.logger.error("[MergeImageProcessor] input_data \
                        is not valid. ")
                if size == []:
                    size = img.shape
                elif not img.shape == size:
                    self.terminate_flag = True
                    self.logger.error("[MergeImageProcessor] input_data \
                        is not valid. ")

            out_image, ori_w, ori_h = tinycv.merge_4_channel_images_opencv(
                imgs)
            self.output_data.append(out_image)
            self.results.append({
                'ori_h': ori_h, 'ori_w': ori_w})

        if package:
            self.output_data = self.output_data[0]
            self.results = self.results[0]


class CropRoiImageProcessor(image_processor_base.ImageProcessorBase):
    """ Crop Roi region according to config. """

    def __init__(self, dyda_config_path=''):
        """ Initialization function of dyda component. """

        super(CropRoiImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)
        if self.param['top'] == -1 or \
           self.param['bottom'] == -1 or \
           self.param['left'] == -1 or \
           self.param['right'] == -1 or \
           self.param['top'] >= self.param['bottom'] or \
           self.param['left'] >= self.param['right']:
            self.terminate_flag = True
            self.logger.error("[CropRoiImageProcessor] parameter \
                is not valid.")

    def find_ResizeImageProcessor(self):
        """ Find ResizeImageProcessor results from metadata """

        found = False
        for comp in self.metadata[1:]:
            if comp['class_name'] == 'ResizeImageProcessor':
                if isinstance(comp['results'], dict):
                    new_width = comp['results']['new_size']['width']
                    ori_width = comp['results']['ori_size']['width']
                elif isinstance(comp['results'], list):
                    new_width = comp['results'][0]['new_size']['width']
                    ori_width = comp['results'][0]['ori_size']['width']
                self.resize_ratio = float(new_width / ori_width)
                found = True
        if found is False:
            self.resize_ratio = 1

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()
        self.find_ResizeImageProcessor()
        for img in input_data:
            top = int(self.param['top'] * self.resize_ratio)
            bottom = int(self.param['bottom'] * self.resize_ratio)
            left = int(self.param['left'] * self.resize_ratio)
            right = int(self.param['right'] * self.resize_ratio)
            self.output_data.append(img[top: bottom, left: right, :])
            self.results.append({
                'top': top,
                'bottom': bottom,
                'left': left,
                'right': right})
        self.uniform_output()


class CropMultiUseAnnoImageProcessor(image_processor_base.ImageProcessorBase):
    """ Use external metadata to crop and rotate data
        The input_data needs to be a list and contain more than one data points
    """

    def __init__(self, dyda_config_path='', param=None):
        super(CropMultiUseAnnoImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.results = {}
        self.rect = tinycv.Rect()
        if 'output_single_list' not in self.param.keys():
            self.single_list = True
        else:
            self.single_list = self.param['output_single_list']

    def main_process(self):
        """ main_process of CropMultiUseAnnoImageProcessor """

        if len(self.input_data) != 2:
            self.terminate_flag = True
            self.logger.error(
                "input_data should be a list of [image_array, metadata]."
            )
            return False

        if len(self.input_data[0]) != len(self.input_data[1]):
            self.terminate_flag = True
            self.logger.error(
                "Size of image_array, metadata does not match."
            )
            return False

        self.output_data = []
        self.results = {}
        for i in range(len(self.input_data[1])):
            counter = 0
            img_data = copy.deepcopy(self.input_data[0][i])
            metadata = self.input_data[1][i]
            cropped_list = []
            if "annotations" in metadata.keys():
                if len(metadata["annotations"]) >= 1:
                    for anno in metadata["annotations"]:
                        try:
                            self.rect.reset_loc([
                                anno["top"], anno["bottom"],
                                anno["left"], anno["right"]
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
                        if not self.single_list:
                            cropped_list.append(cropped)
                        else:
                            self.output_data.append(cropped)
                        counter = counter + 1
            if not self.single_list:
                self.output_data.append(cropped_list)
            self.results[i] = counter
        return True

    def reset_results(self):
        """ reset_results of CropUseAnnoImageProcessor"""
        self.results = {}


class CropUseAnnoImageProcessor(image_processor_base.ImageProcessorBase):
    """ Use external metadata to crop and rotate data """

    def __init__(self, dyda_config_path='', param=None):
        super(CropUseAnnoImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.results = {}
        self.rect = tinycv.Rect()

    def main_process(self):
        """ main_process of CropUseAnnoImageProcessor """

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
                self.logger.warning(
                    "Multiple input image arrays detected, "
                    "but only the first one will be proceeded."
                )

        if isinstance(self.input_data[1], dict):
            metadata = self.input_data[1]
        elif isinstance(self.input_data[1], list):
            metadata = self.input_data[1][0]
            if len(self.input_data[1]) > 1:
                self.logger.warning(
                    "Multiple input metadata detected, "
                    "but only the first one will be proceeded."
                )

        if "annotations" not in metadata.keys():
            self.terminate_flag = True
            self.logger.error("annotations key is missing in metadata.")
            return False

        if len(metadata["annotations"]) < 1:
            self.logger.warning("No annotation found in metadata.")
            return True

        self.output_data = []
        self.results = copy.deepcopy(metadata)
        for anno in metadata["annotations"]:
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
            self.output_data.append(cropped)

        return True

    def reset_results(self):
        """ reset_results of CropUseAnnoImageProcessor"""
        self.results = {}


class ResizeImageProcessor(image_processor_base.ImageProcessorBase):
    """ Resize image according to config. """

    def __init__(self, dyda_config_path=''):
        """ Initialization function of dyda component. """

        super(ResizeImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        for img in input_data:
            ori_size = img.shape[:2]
            self.get_new_size(ori_size)
            self.output_data.append(
                image.resize_img(
                    img, size=(self.new_size[1], self.new_size[0])))
            self.results.append({
                'ori_size': {
                    'height': ori_size[0],
                    'width': ori_size[1]},
                'new_size': {
                    'height': self.new_size[0],
                    'width': self.new_size[1]}
            })

        self.uniform_output()

    def get_new_size(self, ori_size):
        if self.param['height'] == -1 and self.param['width'] == -1:
            self.new_size = ori_size
        elif self.param['height'] == -1:
            self.new_size = (
                int(self.param['width'] / ori_size[1] * ori_size[0]),
                self.param['width'])
        elif self.param['width'] == -1:
            self.new_size = (
                self.param['height'], int(
                    self.param['height'] / ori_size[0] * ori_size[1]))
        else:
            self.new_size = (self.param['height'], self.param['width'])


class CalibrateImageProcessor(image_processor_base.ImageProcessorBase):
    """ Calibrate image to align input image to given background image.
    """

    def __init__(self, dyda_config_path=''):
        """ Initialization function of dyda component. """

        super(CalibrateImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)
        self.bg_img = image.read_img(self.param['bg_img_path'])

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        for img in input_data:
            [img_calibrated, img_bg] = tinycv.image_calibration(
                img, self.bg_img)
            self.output_data.append(img_calibrated)

        self.uniform_output()


class Gray2COLORImageProcessor(image_processor_base.ImageProcessorBase):
    """ Turn Gray image to Color scale.
    """

    def __init__(self, dyda_config_path=''):
        """ __init__ of Gray2COLORImageProcessor """

        super(Gray2COLORImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)
        self.order = "bgr"
        if "color_order" in self.param.keys():
            self.order = self.param["color_order"]

    def main_process(self):
        """ define main_process of dyda component """
        self.results = []
        self.output_data = []
        for img in self.input_data:
            if not image.is_rgb(img):
                img = image.conv_color(img, order=self.order)
            self.output_data.append(img)


class BGR2GrayImageProcessor(image_processor_base.ImageProcessorBase):
    """ Turn BGR image to gray scale.
    """

    def __init__(self, dyda_config_path=''):
        """ __init__ of BGR2GrayImageProcessor """

        super(BGR2GrayImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

    def main_process(self):
        """ define main_process of dyda component """

        self.reset_output()
        input_data = self.uniform_input()

        for img in input_data:
            if image.is_rgb(img):
                img = image.conv_gray(img)
            self.output_data.append(img)

        self.uniform_output()


class BGR2HSVImageProcessor(image_processor_base.ImageProcessorBase):
    """ Turn BGR image to HSV image.
    """

    def __init__(self, dyda_config_path=''):
        """ Initialization function of dyda component. """

        super(BGR2HSVImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        for img in input_data:
            self.output_data.append(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

        self.uniform_output()


class ChannelSplitImageProcessor(image_processor_base.ImageProcessorBase):
    """ Channel split and leave one according to config.
    """

    def __init__(self, dyda_config_path=''):
        """ Initialization function of dyda component. """

        super(ChannelSplitImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        for img in input_data:
            channels = [[] for i in range(3)]
            channels[0], channels[1], channels[2] = cv2.split(img)
            self.output_data.append(channels[self.param['channel_index']])

        self.uniform_output()


class HistEqualizeImageProcessor(image_processor_base.ImageProcessorBase):
    """ Histogram equalization
    """

    def __init__(self, dyda_config_path=''):
        """ Initialization function of dyda component. """

        super(HistEqualizeImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        for img in input_data:
            self.output_data.append(cv2.equalizeHist(img))

        self.uniform_output()


class CannyEdgeImageProcessor(image_processor_base.ImageProcessorBase):
    """ Canny edge detection.
    """

    def __init__(self, dyda_config_path=''):
        """ Initialization function of dyda component. """

        super(CannyEdgeImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        for img in input_data:
            self.output_data.append(cv2.Canny(
                img,
                self.param['min_val'],
                self.param['max_val']))

        self.uniform_output()


class BinarizeImageProcessor(image_processor_base.ImageProcessorBase):
    """ Binarize one channel image.
    """

    def __init__(self, dyda_config_path=''):
        """ Initialization function of dyda component. """

        super(BinarizeImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        for img in input_data:
            ret, thresholded = cv2.threshold(
                img, self.param['threshold'], 255, cv2.THRESH_BINARY)
            self.output_data.append(thresholded)

        self.uniform_output()


class AdaptiveBinarizeImageProcessor(image_processor_base.ImageProcessorBase):
    """ Binarize one channel image by adaptive threshold.
    """

    def __init__(self, dyda_config_path=''):
        """ Initialization function of dyda component. """

        super(AdaptiveBinarizeImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        for img in input_data:
            thre = int(np.mean(img.mean(axis=1)))
            ret, thresholded = cv2.threshold(
                img, thre + self.param['thre_bias'], 255, cv2.THRESH_BINARY)
            self.output_data.append(thresholded)

        self.uniform_output()


class FindContoursImageProcessor(image_processor_base.ImageProcessorBase):
    """ Find contours.
    """

    def __init__(self, dyda_config_path=''):
        """ Initialization function of dyda component. """

        super(FindContoursImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

    def main_process(self):
        """ Main function of dyda component. """

        img_ori = self.input_data[0]
        self.input_data = self.input_data[1]

        self.reset_output()
        input_data = self.uniform_input()

        for img in input_data:
            output_data, contours, hierarchy = cv2.findContours(
                img,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
            # get the largest N contour
            if not self.param['number'] == -1:
                contours = sorted(contours, key=cv2.contourArea,
                                  reverse=True)[:self.param['number']]
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c,
                                      self.param['peri_ratio'] * peri, True)
            if len(approx) == self.param['vertex_number']:
                self.output_data.append(
                    cv2.drawContours(copy.deepcopy(img_ori),
                                     [approx], -1, (0, 255, 0), 3))

        self.uniform_output()


class LBPImageProcessor(image_processor_base.ImageProcessorBase):
    """ Extract LBP feature.
    """

    def __init__(self, dyda_config_path=''):
        """ __init__ of LBPImageProcessor """

        super(LBPImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

    def main_process(self):
        """ define main_process of dyda component """
        self.results = []
        self.output_data = []
        for img in self.input_data:
            feat = image.LBP(
                img,
                parms=(
                    self.param['point'],
                    self.param['radius']),
                subtract=self.param['subtract'])
            self.output_data.append(feat)


class BgSubtractImageProcessor(image_processor_base.ImageProcessorBase):
    """ Background subtraction by Gaussian mixture models.
    """

    def __init__(self, dyda_config_path=''):
        """ Initialization function of dyda component. """

        super(BgSubtractImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

        self.channel_num = 1
        if 'channel_num' in self.param.keys():
            self.channel_num = self.param['channel_num']

        # initialize background subtractor
        self.bg_subtractor = []
        for i in range(self.channel_num):
            self.bg_subtractor.append(cv2.createBackgroundSubtractorMOG2(
                self.param['history'], self.param['var_threshold'],
                self.param['detect_shadows']))

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()
        for idx, img in enumerate(input_data):
            fg_mask = self.bg_subtractor[idx].apply(
                img, learningRate=self.param['learning_rate'])
            self.output_data.append(fg_mask)

        self.uniform_output()


class MorphOpenImageProcessor(image_processor_base.ImageProcessorBase):
    """ Morphological opening. """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(MorphOpenImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        # create kernel
        kernel_size = self.param['kernel_size']
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        for mask in input_data:
            self.output_data.append(
                cv2.morphologyEx(
                    mask,
                    cv2.MORPH_OPEN,
                    self.kernel,
                    self.param['iter_number']))

        self.uniform_output()


class MorphCloseImageProcessor(image_processor_base.ImageProcessorBase):
    """ Morphological closing. """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(MorphCloseImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        # create kernel
        kernel_size = self.param['kernel_size']
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        for mask in input_data:
            self.output_data.append(
                cv2.morphologyEx(
                    mask,
                    cv2.MORPH_CLOSE,
                    self.kernel,
                    self.param['iter_number']))

        self.uniform_output()


class CCLImageProcessor(image_processor_base.ImageProcessorBase):
    """ Connected components labeling. """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(CCLImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        for img in input_data:
            img = img.astype(np.uint8)
            img_height, img_width = img.shape[:2]
            area = img_height * img_width

            # area threshold
            if self.param['area_min_thre'] < 1:
                area_min_thre = int(area * self.param['area_min_thre'])
            else:
                area_min_thre = self.param['area_min_thre']
            if self.param['area_max_thre'] < 1:
                area_max_thre = int(area * self.param['area_max_thre'])
            else:
                area_max_thre = self.param['area_max_thre']

            # connected components labeling
            nlabels, labels, stats, centroids = \
                cv2.connectedComponentsWithStats(
                    img, self.param['connectivity'], cv2.CV_32S)

            # ignore index 0 which is for background
            annos = []
            output_data = []
            for i in range(1, nlabels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < area_min_thre or area > area_max_thre:
                    continue
                left = max(0, int(stats[i, cv2.CC_STAT_LEFT]))
                top = max(0, int(stats[i, cv2.CC_STAT_TOP]))
                width = min(img_width - 1,
                            int(stats[i, cv2.CC_STAT_WIDTH]))
                height = min(img_height - 1,
                             int(stats[i, cv2.CC_STAT_HEIGHT]))
                right = left + width
                bottom = top + height

                # annos is list of [label, confidence, bounding_box]
                roi = labels[top:bottom, left: right]
                roi = np.where(roi > 0, 1, 0)
                score = sum(sum(roi)) / (width * height)
                annos.append([self.param['label'], score,
                              [top, bottom, left, right]])
                mask = np.zeros(img.shape, np.uint8)
                mask = np.where(labels == i, 255, 0)
                mask = mask.astype(np.uint8)
                output_data.append(mask)

            self.output_data.append(output_data)
            self.results.append(lab_tools.output_pred_detection(
                input_path=self.metadata[0], annotations=annos,
                img_size=(img_width, img_height)))

        self.uniform_output()


class ExtractNonBlackImageProcessor(image_processor_base.ImageProcessorBase):
    """ Extract non-black region. """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(ExtractNonBlackImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        for img in input_data:
            seg_h, seg_w = self.extract_non_black(img)
            self.results.append({"roi": [
                {"top": seg_h[0][0], "bottom": seg_h[0][1],
                 "left": seg_w[0][0], "right": seg_w[0][1],
                 "overlap_threshold": 0.5}]})
            self.output_data.append(img[
                seg_h[0][0]:seg_h[0][1],
                seg_w[0][0]:seg_w[0][1], :])

        self.uniform_output()

    def extract_non_black(self, img):

        lower = np.array([0, 0, 0], dtype="uint8")
        upper = np.array([50, 50, 50], dtype="uint8")
        bin_in = cv2.inRange(img, lower, upper)
        bin_in = bin_in / 255
        bin_h = bin_in.shape[0]
        bin_w = bin_in.shape[1]

        # projection v0
        idx_v0 = tinycv.segmentation_by_projection(
            copy.deepcopy(bin_in), 'v', 0, 0.98, target=0)
        if len(idx_v0) == 0:
            idx_v0.append([0, bin_w])

        # image projection h0
        idx_h0 = tinycv.segmentation_by_projection(
            copy.deepcopy(bin_in), 'h', 0, 0.98, target=0)
        if len(idx_h0) == 0:
            idx_h0.append([0, bin_h])

        return(idx_h0, idx_v0)


class SelectByIdImageProcessor(image_processor_base.ImageProcessorBase):
    """ Select and output one image in self.input_data[1]
        according to channel id given in self.input_data[0].
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(SelectByIdImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function of dyda component. """

        channel_index = self.input_data[0]['channel_index']
        self.output_data = copy.deepcopy(self.input_data[1][channel_index])


class BGR2RGBImageProcessor(image_processor_base.ImageProcessorBase):
    """ Turn BGR image to RGB image.
    """

    def __init__(self, dyda_config_path=''):
        """ Initialization function of dyda component. """

        super(BGR2RGBImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        for img in input_data:
            self.output_data.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        self.uniform_output()


class TransformImageProcessor(image_processor_base.ImageProcessorBase):
    """ Image transformation.
    """

    def __init__(self, dyda_config_path=''):
        """ Initialization function of dyda component. """

        super(TransformImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

    def parse_param(self):
        """ Parse parameters to extract transform info. """

        self.type = 'perspective'
        if 'type' in self.param:
            self.type = self.param['type']
        self.use_external_meta = False
        if 'use_external_meta' in self.param:
            self.use_external_meta = self.param['use_external_meta']
        self.use_input_data = False
        if 'use_input_data' in self.param:
            self.use_input_data = self.param['use_input_data']

        if self.use_external_meta:
            if isinstance(self.external_metadata, list) \
                    and len(self.external_metadata) == 1:
                if 'trans_info' not in \
                        self.external_metadata[0].keys():
                    self.logger.warning(
                        "No transform info given and skip transformation")
                    return True
                trans_info = self.external_metadata[0][
                    'trans_info']
            elif isinstance(self.external_metadata, dict):
                if 'trans_info' not in \
                        self.external_metadata.keys():
                    self.logger.warning(
                        "No transform info given and skip transformation")
                    return True
                trans_info = self.external_metadata[
                    'trans_info']
            else:
                self.logger.error("Wrong external_metadata type")
                self.terminate_flag = True
                return True
        elif self.use_input_data:
            if isinstance(self.input_data[1], dict):
                if 'trans_info' not in \
                        self.input_data[1].keys():
                    self.logger.warning(
                        "No transform info given and skip transformation")
                    return True
                trans_info = self.input_data[1][
                    'trans_info']
            else:
                self.logger.error("Wrong external_metadata type")
                self.terminate_flag = True
                return True
        else:
            if 'trans_info' in self.param.keys():
                trans_info = self.param['trans_info']
            else:
                self.logger.warning(
                    "No transform info given and skip transformation")
                return True

        # Coordinates of quadrangle vertices in the source image.
        self.tl_s = trans_info['quadrangle_src']['top_left']
        self.tr_s = trans_info['quadrangle_src']['top_right']
        self.bl_s = trans_info['quadrangle_src']['bottom_left']
        self.br_s = trans_info['quadrangle_src']['bottom_right']
        # Coordinates of the corresponding quadrangle vertices in the
        # destination image.
        self.tl_d = trans_info['quadrangle_dst']['top_left']
        self.tr_d = trans_info['quadrangle_dst']['top_right']
        self.bl_d = trans_info['quadrangle_dst']['bottom_left']
        self.br_d = trans_info['quadrangle_dst']['bottom_right']
        self.w_d = trans_info['image_size_dst']['width']
        self.h_d = trans_info['image_size_dst']['height']

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        skip = self.parse_param()
        if self.use_input_data:
            input_data = self.uniform_input(self.input_data[0], 'ndarray')
        else:
            input_data = self.uniform_input(self.input_data, 'ndarray')
        if skip:
            self.output_data = input_data
        else:
            for img in input_data:
                if self.type == 'perspective':
                    dst, M = self.perspective_transformation(img)
                    self.output_data.append(dst)
                    self.results.append(
                        {'trans_mat': M,
                         'size': {'width': self.w_d, 'height': self.h_d}})
                else:
                    self.logger.warning(
                        "Skip transformation since type is not supported")
                    self.output_data.append(img)

        self.uniform_output()

    def perspective_transformation(self, img):
        """ Calculate a perspective transform from four pairs
            of the corresponding points. """

        pts1 = np.float32([
            [self.tl_s['x'], self.tl_s['y']],
            [self.bl_s['x'], self.bl_s['y']],
            [self.tr_s['x'], self.tr_s['y']],
            [self.br_s['x'], self.br_s['y']]])
        pts2 = np.float32([
            [self.tl_d['x'], self.tl_d['y']],
            [self.bl_d['x'], self.bl_d['y']],
            [self.tr_d['x'], self.tr_d['y']],
            [self.br_d['x'], self.br_d['y']]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (self.h_d, self.w_d))
        return(dst, M)

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
