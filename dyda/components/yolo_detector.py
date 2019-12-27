import os
import sys
import random
from ctypes import c_void_p
from ctypes import c_float
from ctypes import cast
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_long
from ctypes import c_ubyte
from ctypes import CDLL
from ctypes import Structure
from ctypes import RTLD_GLOBAL
from ctypes import POINTER
from os.path import join as pjoin

from dt42lab.core import lab_tools
from dyda.core import detector_base


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1


def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


class OBJECT_INFO(Structure):
    _fields_ = [("confidence", c_float),
                ("label", c_int),
                ("left", c_int),
                ("top", c_int),
                ("right", c_int),
                ("bottom", c_int)]


class OBJECT_INFO_LIST(Structure):
    _fields_ = [("data", POINTER(OBJECT_INFO)),
                ("len", c_int)]


class DetectorYOLO(detector_base.DetectorBase):
    """ Modified from darknet.py example in Darknet """

    def __init__(self, dyda_config_path=''):
        """ __init__ of ClassifierInceptionv3

        Trainer Variables:
            input_data: a list of image array
            results: defined by lab_tools.output_pred_detection (dict)

        Arguments:
            testing -- True to call set_testing_params. Anything written in
                       dyda.config will be overwritten.
        """

        super(DetectorYOLO, self).__init__(
            dyda_config_path=dyda_config_path
        )

        self.set_param(self.class_name)
        self.check_param_keys()
        param = self.param

        try:
            self.lib = CDLL(param["lib_path"], RTLD_GLOBAL)
            self.lib.network_width.argtypes = [c_void_p]
            self.lib.network_width.restype = c_int
            self.lib.network_height.argtypes = [c_void_p]
            self.lib.network_height.restype = c_int

            self.make_boxes = self.lib.make_boxes
            self.make_boxes.argtypes = [c_void_p]
            self.make_boxes.restype = POINTER(BOX)

            self.free_ptrs = self.lib.free_ptrs
            self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

            self.free_boxes = self.lib.free_boxes
            self.free_ptrs.argtypes = [c_void_p]

            self.num_boxes = self.lib.num_boxes
            self.num_boxes.argtypes = [c_void_p]
            self.num_boxes.restype = c_int

            self.make_probs = self.lib.make_probs
            self.make_probs.argtypes = [c_void_p]
            self.make_probs.restype = POINTER(POINTER(c_float))

            self.detect = self.lib.network_predict
            self.detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float,
                                    POINTER(BOX), POINTER(POINTER(c_float))]

            self.free_image = self.lib.free_image
            self.free_image.argtypes = [IMAGE]

            self.load_image = self.lib.load_image_color
            self.load_image.argtypes = [c_char_p, c_int, c_int]
            self.load_image.restype = IMAGE

            self.network_detect = self.lib.network_detect
            self.network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float,
                                            c_float, POINTER(BOX),
                                            POINTER(POINTER(c_float))]

            self.network_detect_objinfo = self.lib.network_detect_objinfo
            self.network_detect_objinfo.argtypes = [c_void_p,
                                                    IMAGE,
                                                    c_float,
                                                    c_float,
                                                    c_float,
                                                    POINTER(BOX),
                                                    POINTER(POINTER(c_float))]
            self.network_detect_objinfo.restype = OBJECT_INFO_LIST
            self.free_object_info_list = self.lib.free_object_info_list
            self.free_object_info_list.argtypes = [OBJECT_INFO_LIST]

            libnp_filepath = pjoin(
                os.path.dirname(param["lib_path"]),
                "libdarknet_numpy.so")
            self.libnp = CDLL(libnp_filepath, RTLD_GLOBAL)
            self.ndarray_image = self.libnp.ndarray_to_image
            self.ndarray_image.argtypes = [POINTER(c_ubyte),
                                           POINTER(c_long),
                                           POINTER(c_long)]
            self.ndarray_image.restype = IMAGE

            # load yolo net
            load_net = self.lib.load_network
            load_net.argtypes = [c_char_p, c_char_p, c_int]
            load_net.restype = c_void_p
            self.net = load_net(param["net_cfg"].encode("ascii"),
                                param["net_weights"].encode("ascii"), 0)

            load_meta = self.lib.get_metadata
            self.lib.get_metadata.argtypes = [c_char_p]
            self.lib.get_metadata.restype = METADATA
            self.meta = load_meta(param["net_meta"].encode("ascii"))

        except Exception as error:
            print(error)
            print("[yolo_detector] ERROR: loading darknet library fails %s"
                  % param["lib_path"])
            raise

    def check_param_keys(self):
        """
        Check if any default key is missing in the self.param.
        """

        default_keys = [
            "lib_path", "net_cfg", "net_weights", "net_meta",
            "thresh", "hier_thresh", "nms"
        ]
        for _key in default_keys:
            if _key not in self.param.keys():
                print("[yolo_detector] ERROR: %s missing in self.param" % _key)
                self.terminate_flag = True
            else:
                continue
        print("[yolo_detector] INFO: keys of self.param are checked")

    def main_process(self, thresh=.5, hier_thresh=.5, nms=.45):
        # FIXME: the first version of DetectorYOLO accepts a list of image
        #       paths as input_data, this should be fixed (issue #30)
        nms = self.param["nms"]
        hier_thresh = self.param["hier_thresh"]
        thresh = self.param["thresh"]
        annos = None

        def nparray_to_image(img):
            data = img.ctypes.data_as(POINTER(c_ubyte))
            image = self.ndarray_image(
                data, img.ctypes.shape, img.ctypes.strides
            )
            return image

        if not isinstance(self.input_data, list):
            self.input_data = [self.input_data]
            package = True
        else:
            package = False

        for img_array in self.input_data:
            im = nparray_to_image(img_array)
            img_shape = img_array.shape
            img_w = img_shape[1]
            img_h = img_shape[0]
            boxes = self.make_boxes(self.net)
            probs = self.make_probs(self.net)
            num = self.num_boxes(self.net)
            obj_info_list = self.network_detect_objinfo(
                self.net, im, thresh, hier_thresh, nms, boxes, probs
            )
            objs = obj_info_list.data[:obj_info_list.len]
            annos = []
            for obj in objs:
                label = str(self.meta.names[obj.label], 'utf-8')
                conf = obj.confidence
                bb = [obj.top, obj.bottom, obj.left, obj.right]
                annos.append([label, conf, bb])
            self.free_image(im)
            self.free_object_info_list(obj_info_list)
            self.free_ptrs(cast(probs, POINTER(c_void_p)), num)
            self.free_boxes(boxes)
            res = lab_tools.output_pred_detection(self.metadata[0], annos)
            self.results.append(res)

        if package:
            self.results = self.results[0]
        return annos

    def convert_box_to_lab_bb(self, box, img_w, img_h):
        """ convert yolo box to a bb defined in lab spec """

        x = box.x
        y = box.y
        w = box.w
        h = box.h
        top = int(y - h/2)
        top = top if top > 0 else 0
        # Note: top + bottom = 2y
        bottom = int(2*y - top)
        bottom = bottom if bottom < img_h else img_h - 1
        left = int(x - w/2)
        left = left if left > 0 else 0
        # Note: left + right = 2x
        right = int(2*x - left)
        right = right if right < img_w else img_w - 1
        return [top, bottom, left, right]
