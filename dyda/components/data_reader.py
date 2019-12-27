import os
import copy
import cv2
import numpy as np
import pandas as pd
from dyda.core import data_reader_base
from dyda_utils import lab_tools
from dyda_utils import tools


class Video2FrameReader(data_reader_base.DataReaderBase):

    def __init__(self, dyda_config_path="", param=None):
        """ Init function of Video2FrameReader """
        super(Video2FrameReader, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.concat = True
        if "concat" in self.param.keys() and self.param["concat"] is False:
            self.concat = False
        self.fps = 30
        if "FPS" in self.param.keys():
            self.fps = self.param["fps"]

    def main_process(self):
        """ Main function called by the external code """

        self.results['total_frames'] = []
        if isinstance(self.input_data, str):
            input_data = [self.input_data]
        else:
            input_data = self.input_data
        for i in range(0, len(input_data)):
            if not self.concat:
                input_data.append([])
            video_path = input_data[i]
            if not isinstance(video_path, str):
                self.terminate_flag = True
                self.logger.error(
                    "item of input_data should be str of input video path"
                )
                return False
            if not tools.check_exist(video_path):
                self.terminate_flag = True
                self.logger.error(
                    "item of input_data should be str of input video path"
                )
                return False
            count = 0
            try:
                # FPS selection in this component is a workaround because
                # setting CAP_PROP_FRAME_FRAMES does not working
                # details see https://goo.gl/yVimzd and https://goo.gl/GeuwX1
                sel_count = int(30 / self.fps)
                vidcap = cv2.VideoCapture(video_path)
                success, img = vidcap.read()
                success = True
                while success:
                    count += 1
                    if count % sel_count == 0:
                        if self.concat:
                            self.output_data.append(img)
                        else:
                            self.output_data[i].append(img)
                    success, img = vidcap.read()
            except BaseException:
                self.terminate_flag = True
                self.logger.error("Fail to read %ith frame" % count)
                return False

            self.results['data_path'].append(input_data)
            self.results['data_type'] = 'array'
            self.results['total_frames'].append(count)


class MetaROIAsAnnoReader(data_reader_base.DataReaderBase):

    def __init__(self, dyda_config_path="", param=None):
        super(MetaROIAsAnnoReader, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.reset_results()

    def main_process(self):
        """ Main function called by the external code """

        if isinstance(self.input_data, list):
            input_data_list = self.input_data
        else:
            input_data_list = [self.input_data]

        for i in range(0, len(input_data_list)):
            input_data = input_data_list[i]
            self.results.append(lab_tools._output_pred(""))
            if "roi" not in input_data.keys():
                self.logger.error("No ROI info found in external_meta")
                self.terminate_flag = True
                return False
            for j in range(0, len(input_data["roi"])):
                self.results[i]["annotations"].append(
                    lab_tools._lab_annotation_dic()
                )
                for boundary in input_data["roi"][i].keys():
                    self.results[i]["annotations"][j][boundary] = \
                        input_data["roi"][j][boundary]

    def reset_results(self):
        """ reset_results function called by pipeline """
        self.results = []


class InputDataAsResultsReader(data_reader_base.DataReaderBase):

    def __init__(self, dyda_config_path="", param=None):
        super(InputDataAsResultsReader, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.force_lab_format = False
        if "force_lab_format" in self.param.keys():
            self.force_lab_format = self.param["force_lab_format"]

    def main_process(self):
        """ Main function called by the external code """

        try:
            self.results = copy.deepcopy(self.input_data)

            if not self.force_lab_format:
                return

            if isinstance(self.results, list):
                for i in range(0, len(self.results)):
                    r = self.results[i]
                    if not lab_tools.is_lab_format(r):
                        self.results[i] = lab_tools._output_pred("")
            elif isinstance(self.results, dict):
                if not lab_tools.is_lab_format(self.results, loose=True):
                    self.results = lab_tools._output_pred("")

            else:
                self.logger.error("format of input_data is not right")
        except Exception as e:
            print(e)

    def reset_results(self):
        """ reset_results function called by pipeline """

        pass


class MultiChannelUpdateReader(data_reader_base.DataReaderBase):

    def __init__(self, dyda_config_path="", param=None):
        super(MultiChannelUpdateReader, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.channel_num = 0
        if "channel_num" in self.param.keys():
            self.channel_num = self.param["channel_num"]
        else:
            self.logger("No channel_num found")
            raise

        self.queue = [np.zeros((1, 1, 3), np.uint8)] * self.channel_num

    def main_process(self):
        """ Main function called by the external code """

        channel_index = self.input_data[0]['channel_index']
        if isinstance(self.input_data[1], list):
            if len(self.input_data[1]) == 1:
                data = self.input_data[1][0]
            else:
                self.logger.warning(
                    "Multiple input_data[1] detected"
                )
        else:
            data = self.input_data[1]

        self.queue[channel_index] = copy.deepcopy(data)
        self.output_data = self.queue


class MultiChannelDataReader(data_reader_base.DataReaderBase):

    def __init__(self, dyda_config_path="", param=None):
        super(MultiChannelDataReader, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.input_data = []

    def main_process(self):
        """ Main function of dyda component. """

        self.output_data = self.input_data
        self.results = copy.deepcopy(self.external_metadata)

        if isinstance(self.results, dict):
            self.add_channe_index(self.results)
        elif isinstance(self.results, list):
            for meta in self.results:
                self.add_channe_index(meta)
        else:
            self.warning("Wrong external_mata type")

    def add_channe_index(self, meta):
        if "channel_id" in meta.keys():
            meta['channel_index'] = meta["channel_id"]
        elif "channel_index" in meta.keys():
            meta['channel_index'] = meta["channel_index"]
        else:
            meta['channel_index'] = -1


class BinaryDataReader(data_reader_base.DataReaderBase):

    def __init__(self, dyda_config_path="", param=None):
        super(BinaryDataReader, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.input_data = []

    def main_process(self):
        """ Main function called by the external code """

        self.results['data_path'].append('DIRECT_BINARY_INPUT')
        self.results['data_type'] = 'array'
        # Do nothing, assign input_data to output_data
        self.output_data = self.input_data


class BinaryDataReaderUseChanId(data_reader_base.DataReaderBase):

    def __init__(self, dyda_config_path="", param=None):
        super(BinaryDataReaderUseChanId, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.input_data = []
        empty_img = np.zeros(
            [self.param['empty_img_height'], self.param['empty_img_width'], 3],
            dtype=np.uint8)
        self.pre_output = [copy.deepcopy(empty_img) for i in range(
            self.param['channel_num'])]

    def main_process(self):
        """ Main function called by the external code """

        chan_id = self.external_metadata['channel_id']
        if chan_id < 0 or chan_id >= self.param['channel_num']:
            self.logger.warning(
                "Skip update due to invalid channel id."
            )
        elif self.input_data == []:
            self.logger.warning(
                "Skip update due to empty input_data."
            )
        else:
            self.unpack_single_input()
            self.pre_output[chan_id] = copy.deepcopy(self.input_data)

        self.output_data = copy.deepcopy(self.pre_output)


class BatchJsonReader(data_reader_base.DataReaderBase):
    """ Read in dict data from a list of json files """

    def __init__(self, dyda_config_path="", param=None):
        super(BatchJsonReader, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.results = []

    def reset_results(self):
        """ Reset results of the component """
        self.results = []

    def check_folder(self, json_path):
        """ Check if folder given by config.
            Replace folder if yes.
        """
        if "folder" in self.param.keys():
            json_path = os.path.join(
                self.param["folder"],
                os.path.basename(json_path))
        return json_path

    def main_process(self):
        """ Main function called by the external code """

        encoding = None
        if "encoding" in self.param.keys():
            encoding = self.param["encoding"]

        if isinstance(self.input_data, str):
            json_path = self.check_folder(self.input_data)
            self.results = self.get_metadata(json_path)
            self.metadata[0] = tools.remove_extension(
                self.input_data, return_type='base-only'
            )

        elif isinstance(self.input_data, list):
            for json_path in self.input_data:
                json_path = self.check_folder(json_path)
                self.results.append(self.get_metadata(json_path))

        return True


class JsonReader(data_reader_base.DataReaderBase):
    """Read in dict data from a json file.
       The folder and filename extension is given by parameter.
       The filename is given by self.metadata[0].

       @param folder: folder of json files
       @param extension: filename extension
    """

    def __init__(self, dyda_config_path="", param=None):
        super(JsonReader, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def reset_results(self):
        """ Reset results of the component """
        # FIXME: Not sure what should be the right way to reset results yet
        #        Need Jocelyn to fix it.
        pass

    def main_process(self):
        """ Main function called by the external code """
        if self.param["folder"] == "":
            self.param["folder"] = os.path.dirname(self.input_data[0])
        filename = os.path.join(
            self.param["folder"],
            self.metadata[0] + self.param["extension"])
        self.results = self.get_metadata(filename)


class JsonReaderFix(data_reader_base.DataReaderBase):
    """Read in dict data from a json file.
       The folder and filename is given by parameter.
       Specific post process to the data can be applied
       according to the type given by parameter.

       @param folder: folder of the json file
       @param filename: filename of the json file
       @param type:
         "space_map": space dependency will be calculated and
         recorded in lab_info
    """

    def __init__(self, dyda_config_path="", param=None):
        super(JsonReaderFix, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.results = []

    def reset_results(self):
        """ Reset results of the component """
        # FIXME: Not sure what should be the right way to reset results yet
        #       Need Jocelyn to fix it.
        pass

    def main_process(self):
        """ Main function called by the external code """
        if self.results == []:
            filename = os.path.join(
                self.param["folder"],
                self.param["filename"])
            self.results = self.get_metadata(filename)
            if self.param["type"] == "space_map":
                self.calculate_space_dependency()

    def calculate_space_dependency(self):
        self.results = lab_tools.extract_target_value(
            self.results,
            target_key='label',
            target_value='space')
        annotations = self.results["annotations"]
        overlap_ratio_all = lab_tools.calculate_overlap_ratio_all(
            annotations, annotations)
        space_num = len(annotations)
        lab_info = {
            "space_dependency": [[] for _ in range(space_num)]}
        for i in range(space_num):
            space_index = annotations[i]['track_id']
            for j in range(space_num):
                if overlap_ratio_all[i][j] > 0:
                    related_index = annotations[j]['track_id']
                    lab_info["space_dependency"][space_index].append(
                        related_index)
        self.results["lab_info"] = lab_info


class VocXmlReader(data_reader_base.DataReaderBase):
    """Read in xml file in format defined by Voc and
       turn it to lab-format dict.
    """

    def __init__(self, dyda_config_path="", param=None):
        """ Initialization function of dyda component. """

        super(VocXmlReader, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.default_result = {
            'folder': None,
            'filename': None,
            'annotations': [],
            'size': {'width': None, 'height': None}}

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input(self.input_data, 'str')

        for path in input_data:
            self.results.append(self.xml2dict(path))

        self.uniform_output()

    def xml2dict(self, path):
        """ Turn xml path to lab-format dict. """

        if os.path.isfile(path):
            result = tools.voc_xml_to_dict(path)
            result = self.modify_path(result)
        else:
            result = self.default_result
            self.logger.warning(
                "xml file %s does not exist. " % path
            )
        return result

    def modify_path(self, result):
        """ modify folder and filename """

        if 'force_folder' in self.param.keys():
            result['folder'] = self.param['force_folder']
        if 'set_basename_as_filename' in self.param.keys():
            if self.param['set_basename_as_filename']:
                result['filename'] = self.metadata[0]
        if 'force_extension' in self.param.keys():
            result['filename'] = tools.replace_extension(
                result['filename'],
                self.param['force_extension'])
        return result

    def reset_results(self):
        """ reset_results function called by pipeline """

        self.results = []


class CsvDataReader(data_reader_base.DataReaderBase):

    def __init__(self, dyda_config_path="", param=None):
        """ Initialization function of dyda component. """

        super(CsvDataReader, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        if 'thousands' in (self.param.keys()):
            self.thousands = self.param['thousands']
        else:
            self.thousands = ','
        if 'index_col' in (self.param.keys()):
            self.index_col = self.param['index_col']
        else:
            self.index_col = None

    def main_process(self):
        """ Main function called by the external code """
        self.pack_input_as_list()

        for input_path in self.input_data:
            df = pd.read_csv(input_path,
                             thousands=self.thousands,
                             index_col=self.index_col)
            self.output_data.append(df)

        self.unpack_single_output()
