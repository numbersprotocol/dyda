import pickle
import copy
import numpy as np
from face_recognition import compare_faces, face_locations, face_encodings
from dt42lab.core import tools
from dt42lab.core import lab_tools
from dyda.core import data_analyzer_base
from dyda.core import detector_base
from dyda.core import image_processor_base


""" wrote down at 2019/7/25, due to dependency concern, we move
    all face_recognition components to this file
"""


class FaceMatchAnalyzer(data_analyzer_base.DataAnalyzerBase):
    """ Match unknown face encodings with known face encodings.

        @param tolerance: the max distance that FaceMatchAnalyzer will
                          determine two encodings mathing.
        @pickle_path: the path of pickle storing known faces. If
                      not specified, will automatically assume
                      that the name of pickle file is
                      encodings.pickle and under snapshot folder
                      of FaceEncodingImageProcessor
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(FaceMatchAnalyzer, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        if "pickle_path" in self.param.keys():
            self.pickle_path = self.param["pickle_path"]
        else:
            self.pickle_path = ""

        if "tolerance" in self.param.keys():
            self.tolerance = self.param["tolerance"]
        else:
            self.tolerance = 0.6

        self.first_time = True

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        if not isinstance(self.input_data, list):
            self.input_data = [self.input_data]
            self.package = True
        else:
            self.package = False

        # let input_data will always be list of list
        if not any(isinstance(i, list) for i in self.input_data):
            self.input_data = [self.input_data]
            self.package_2 = True
        else:
            self.package_2 = False

        # open and load the pickle file only at first frame
        if self.first_time:

            if self.pickle_path == "":
                self.pickle_path = self.snapshot_folder.replace(
                    "FaceMatchAnalyzer",
                    "FaceEncodingImageProcessor/encodings.pickle")

            with open(self.pickle_path, 'rb') as rfp:
                self.known_faces = pickle.load(rfp)

            self.first_time = False

        for encodings in self.input_data:

            temp_results = []

            for encoding in encodings:
                matches = compare_faces(self.known_faces["encodings"],
                                        encoding,
                                        tolerance=self.tolerance)

                if True not in matches:
                    whose_face = "unknown"
                else:
                    candidates = [i for (i, j) in
                                  zip(self.known_faces['names'], matches) if j]
                    whose_face = max(set(candidates), key=candidates.count)

                temp_results.append(lab_tools.output_pred_classification(
                                    "", -1, whose_face))

            self.results.append(temp_results)

            if self.package_2:
                self.unpack_single_results()
        self.uniform_output()


class FaceDetector(detector_base.DetectorBase):
    """ Detect face in image. And return the face location in results"""

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component.

            @param model: can choose hog or cnn, hog is faster but
                          less accurate, cnn is slower but more accurate
                          usually.
            @upsample_times: times to upsample the image. Higher number
                             might find smaller faces in image.
        """

        super(FaceDetector, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        if "model" in self.param.keys():
            self.model = self.param["model"]
        else:
            self.model = "hog"

        if "upsample_times" in self.param.keys():
            self.upsample_times = self.param["upsample_times"]
        else:
            self.upsample_times = 1

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        self.input_data = self.uniform_input(self.input_data, 'ndarray')

        for input_img in self.input_data:
            width = input_img.shape[1]
            height = input_img.shape[0]

            # the boxes face_locations returns are in the order
            # of (top, right, bottom, left)
            boxes = face_locations(
                input_img,
                model=self.model,
                number_of_times_to_upsample=self.upsample_times)

            # box in lab_format is [top, bottom, left, right]
            annotations = [["face", -1.0,
                            [i[0], i[2], i[3], i[1]]] for i in boxes]
            self.results.append(lab_tools.output_pred_detection(
                "", annotations, img_size=[width, height]))

        self.uniform_output()


class FaceEncodingImageProcessor(image_processor_base.ImageProcessorBase):
    """ Convert face into 128-d vector.

        @param save_encodings: boolean, whether to save vectors in pickle
                               file. if want to save encodings, should notice
                               that image contains only one face
        @pickle_path: the path of saving pickle file. if not specified, will
                      automatically generate a pickle file named
                      encodings.pickle under the snapshot folder of
                      FaceEncodingImageProcessor
        @num_jitters: the times to re-sample the face
        @last_frame: the path of last frame. feed this parameter
                     to tell this component is time to write pickle,
                     when set save_encodings equals true.

       note: because there might be several faces in a image, the output will
             always be like [np.ndarray, ...] or [[np.ndarray, ...],
             [np.ndarray, ...], ...]. the first list store face encodings of
             first image, although there is only one face, output will also
             be a list. and same for second and following images
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(FaceEncodingImageProcessor, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        if "save_encodings" in self.param.keys():
            self.save_encodings = self.param["save_encodings"]
            self.last_frame = self.param["last_frame"]
        else:
            self.save_encodings = False

        if "pickle_path" in self.param.keys():
            self.pickle_path = self.param["pickle_path"]
        else:
            self.pickle_path = ""

        if "num_jitters" in self.param.keys():
            self.num_jitters = self.param["num_jitters"]
        else:
            self.num_jitters = 1

        self.names = []
        self.known_encodings = []

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()

        imgs, lab_dicts = self.input_data
        if not isinstance(imgs, list):
            imgs = [imgs]
            self.package_1 = True
        else:
            self.package_1 = False

        if not isinstance(lab_dicts, list):
            lab_dicts = [lab_dicts]
            self.package_2 = True
        else:
            self.package_2 = False

        if self.package_1 != self.package_2:
            self.logger.warning(
                "Inputs should be both list, or both not list")
            self.terminate_flag = True

        if self.package_1 and self.package_2:
            self.package = True
        else:
            self.package = False

        for img, lab_dict in zip(imgs, lab_dicts):
            annotations = lab_dict["annotations"]
            boxes = self.get_bounding_boxes(annotations)
            # face_encodings always return a list of np.ndarray here
            encodings = face_encodings(img, known_face_locations=boxes,
                                       num_jitters=self.num_jitters)
            self.output_data.append(encodings)

            if self.save_encodings:

                face_name = self.external_data[0].split('/')[-2]

                # only registeration one face in one frame
                if len(encodings) != 1:
                    self.logger.warning("detect {} faces in {}/{}".format(
                        len(encodings), face_name, self.metadata[0]))
                    break

                # automatically use the folder name as the face name
                # if image's path is like /home/shared/NAME/image1.png
                # use NAME as the face name

                self.names.append(face_name)

                encoding = encodings[0]
                self.known_encodings.append(encoding)

                data = {"encodings": self.known_encodings,
                        "names": self.names}

                if self.external_data[0] == self.last_frame:

                    if self.pickle_path == "":

                        tools.check_dir(self.snapshot_folder)

                        self.pickle_path = self.snapshot_folder + \
                            '/encodings.pickle'

                    self.logger.info("saving face encodings to {}".format(
                        self.pickle_path))

                    with open(self.pickle_path, 'wb') as wfp:
                        pickle.dump(data, wfp)

        self.uniform_output()

    def get_bounding_boxes(self, annotations):
        """ Actually, we have a function annotation_to_boxes()
            in dt42lab.lab_tools. But here I want the boxes return
            like [(top, right, bottom, left)], not [[top, bottom, left,
            right]] which annotation_to_boxes() returns.
        """
        boxes = []
        for anno in annotations:
            boxes.append((anno["top"],
                          anno["right"],
                          anno["bottom"],
                          anno["left"]))
        return boxes
