import os
import cv2
import sys
import copy
import numpy as np
import traceback
import statistics
from scipy.stats import mode
from dt42lab.core import tinycv
from dt42lab.core import tools
from dt42lab.core import pandas_data
from dt42lab.core import lab_tools
from dt42lab.core import image
from dyda.core import determinator_base
from dyda.core import data_accumulator_base


class AccumulatorObjectNumber(determinator_base.DeterminatorBase):
    """Accumulate the number of each class according to track_id.  """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(AccumulatorObjectNumber, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.label_count_list = list()
        self.frame_counter = list()

        # check if parameter valid
        if 'reset_frame_num' in (self.param.keys()):
            self.N = self.param['reset_frame_num']
            if not isinstance(self.N, int) or self.N < -1 or self.N == 0:
                self.logger.warning(
                    "only accept positive integer or -1(default)"
                    " as reset_frame_num, following use default number")
                self.N = -1
        else:
            self.N = -1
        if 'appear_num_thre' not in self.param.keys():
            self.appear_num_thre = 0
        else:
            self.appear_num_thre = self.param['appear_num_thre']

    def main_process(self):
        """ Main function of dyda component. """

        self.reset_output()
        input_data = self.uniform_input()

        input_len = len(input_data)
        if self.label_count_list == []:
            self.label_count_list = [{} for i in range(input_len)]
        elif input_len != len(self.label_count_list):
            self.terminate_flag = True
            self.logger.error("Length of input_data changed.")
            return None

        if self.frame_counter == []:
            self.frame_counter = [{'frame cumulative number': 0,
                                   'starting time': -1}
                                  for i in range(input_len)]
        elif input_len != len(self.frame_counter):
            self.terminate_flag = True
            self.logger.error("Length of input_data changed.")
            return None

        for idx, data in enumerate(input_data):
            if not lab_tools.if_result_match_lab_format(data):
                self.terminate_flag = True
                self.logger.error("Input is not valid.")
                return None

            if self.frame_counter[idx]['starting time'] == -1:
                self.frame_counter[idx]['starting time'] = \
                    self.filename_to_time(data['filename'])

            # restart counting by given N
            N_now = self.frame_counter[idx]['frame cumulative number']
            if N_now % self.N == 0 and self.N != -1:
                self.label_count_list = [{} for i in range(input_len)]
                self.frame_counter[idx]['starting time'] = \
                    self.filename_to_time(data['filename'])

            self.countframe(idx)
            self.add_track_id_to_count_list(data, idx)
            count_results = self.count_label_amount(self.label_count_list[idx])
            self.results.append(dict(object_counting=count_results))

        self.uniform_results()

    def countframe(self, idx):
        """ Count frame number. """

        self.frame_counter[idx]['frame cumulative number'] += 1

    def add_track_id_to_count_list(self, data, idx):
        """ Add track_id to label. """

        for anno in data['annotations']:

            # check if there is enough info
            if 'label' not in anno.keys() or \
               'track_id' not in anno.keys() or \
               anno['track_id'] < 0:
                continue

            # create new list for new label
            if anno['label'] not in self.label_count_list[idx].keys():
                self.label_count_list[idx][anno['label']] = {}

            # add new track_id to count list
            if anno['track_id'] not in \
                    self.label_count_list[idx][anno['label']]:
                self.label_count_list[idx][anno['label']][anno['track_id']] = 1
            else:
                self.label_count_list[idx][anno['label']
                                           ][anno['track_id']] += 1

    def count_label_amount(self, label_count):
        """ Count amount of each label. """

        i = copy.deepcopy(label_count)
        pop_num = 0
        for key, value in i.items():
            for track_id, appear_num in value.items():
                if appear_num < self.appear_num_thre:
                    pop_num += 1
            i[key] = len(value) - pop_num
        return i

    def filename_to_time(self, filename):
        """ Turn filename to time. """

        try:
            return float(filename)
        except ValueError:
            self.logger.warning("Filename is not supported to count "
                                "lasting time.")
            return 0.0


class AnnoAccumulatorFast(data_accumulator_base.AccumulatorBase):
    """ This component accumulates annotation results per id.
        Note: This component is fast but it can only handle limited number of
              ids. If actual numbers of total ids >= self.total_ids
              the results will be put to input_id % self.total_ids
        benchmark: <1ms per run on gc
        input_data: dict of lab results
        output_data: lab results
    """

    def __init__(self, dyda_config_path="", param=None):
        """Initialization function of dyda component."""

        super(AnnoAccumulatorFast, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.comp_rule = "mean"
        if "comp_rule" in self.param.keys():
            self.comp_rule = self.param["comp_rule"]

        if self.comp_rule not in ["mean", "sum"]:
            self.logger.error(
                "%s is not a supported rule" % self.comp_rule
            )
            sys.exit(0)

        # key to group aggregated results
        self.group_key = "id"
        if "group_key" in self.param.keys():
            self.group_key = self.param["group_key"]

        # key to calculate sum or mean
        self.agg_key = "confidence"
        if "agg_key" in self.param.keys():
            self.agg_key = self.param["agg_key"]

        # define aggregated number of results
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
        self.total_ids = 100
        if "total_ids" in self.param.keys():
            self.total_ids = int(self.param["total_ids"])

        # create fixed length map
        self.sum = [0] * self.total_ids
        self.count = [0] * self.total_ids
        self.last_frame = [0] * self.total_ids
        self.counter = 0
        self.cal = {}

    def main_process(self):
        """ Main function of dyda component. """

        self.results = []
        self.pack_input_as_list()
        input_data = copy.deepcopy(self.input_data)
        pop_id = self.counter % self.agg_num

        self.results = copy.deepcopy(self.input_data)
        for i in range(0, len(input_data)):
            if i not in self.cal.keys():
                self.cal[i] = {}
                self.cal[i]['sum'] = [0] * self.total_ids
                self.cal[i]['count'] = [0] * self.total_ids
                self.cal[i]['queue'] = [
                    [0] * self.total_ids for m in range(0, self.agg_num)]

            if "annotations" not in input_data[i].keys():
                self.logger.warning(
                    "no annotation found, skip %ith input)" % i
                )
                continue

            id_list = []
            if len(self.input_data[i]['annotations']) == 0:
                continue
            for anno in self.input_data[i]['annotations']:
                try:
                    id_value = int(anno[self.group_key])
                    # if actual numbers of total ids >= self.total_ids
                    # id_value = input_id % self.total_ids
                    if id_value >= self.total_ids:
                        id_value = id_value % self.total_ids
                except KeyError:
                    self.logger.warning(
                        "no group_key %s found, skip." % self.group_key
                    )
                    continue
                except ValueError:
                    self.logger.error(
                        "id value cannot be converted to int, terminate."
                    )
                    self.terminate_flag = True
                try:
                    agg_value = float(anno[self.agg_key])
                except KeyError:
                    self.logger.warning(
                        "no agg_key %s found, skip." % self.agg_key
                    )
                    continue
                except ValueError:
                    self.logger.error(
                        "agg value cannot be converted to float, terminate."
                    )
                    self.terminate_flag = True

                # clear queue if there were no results for a long time
                if self.counter - self.last_frame[id_value] >= self.agg_num:
                    self.cal[i]['sum'][id_value] = 0
                    self.cal[i]['count'][id_value] = 0
                    for queue_id in range(0, self.agg_num):
                        self.cal[i]['queue'][queue_id][id_value] = 0
                    self.sum[queue_id] = 0
                    self.count[queue_id] = 0
                # remember the last frame number for this object
                self.last_frame[id_value] = self.counter

                id_list.append(id_value)
                # pop the oldest result
                if self.counter >= self.agg_num:
                    self.cal[i]['sum'][id_value] = \
                        self.cal[i]['sum'][id_value] - self.cal[i][
                            'queue'][pop_id][id_value]
                # add new result to sum
                self.cal[i]['sum'][id_value] = \
                    self.cal[i]['sum'][id_value] + agg_value
                # replace the result in queue
                self.cal[i]['queue'][pop_id][
                    id_value] = copy.deepcopy(agg_value)

            for id_ix in range(0, self.total_ids):
                if id_ix in id_list:
                    self.cal[i]['count'][id_ix] = self.agg_num
                else:
                    # reset for pop_id row
                    self.cal[i]['queue'][pop_id][id_ix] = 0
            if self.comp_rule == "mean":
                agg_results = np.divide(
                                np.array(self.cal[i]['sum']),
                                np.array(self.cal[i]['count'])
                              )
            else:
                agg_results = np.array(self.cal[i]['sum'])

            for anno in self.results[i]['annotations']:
                anno[self.agg_key] = agg_results[anno[self.group_key]]
            self.results[i]['additional_info'] = {
                "agg_results": list(agg_results)
                }

        # reset counter if it runs for 7 days under fps~15
        if self.counter >= 10000000:
            self.counter = 1
        else:
            self.counter = self.counter + 1
        self.unpack_single_results()


class AnnoAccumulatorPandas(data_accumulator_base.AccumulatorBase):
    """ This component accumulates result dict arrays using pandas to convert
        lab dicts to dataframe. The performance is bad on low-end machines
        such as Pi (~400ms per run), but it is ok to run it on normal machines
        with Intel CPU. The good thing of this component is that it does not
        put any limitation to input numbers of ids (or track_id).

        benchmark: < 30ms per run on gc
        input_data: dict of lab results
        output_data: lab results
    """

    def __init__(self, dyda_config_path="", param=None):
        """Initialization function of dyda component."""

        super(AnnoAccumulatorPandas, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.comp_rule = "mean"
        if "comp_rule" in self.param.keys():
            self.comp_rule = self.param["comp_rule"]

        if self.comp_rule not in ["mean", "sum"]:
            self.logger.error(
                "%s is not a supported rule" % self.comp_rule
            )
            sys.exit(0)

        # key to group aggregated results
        self.group_key = "id"
        if "group_key" in self.param.keys():
            self.group_key = self.param["group_key"]

        # key to calculate sum or mean
        self.agg_key = "confidence"
        if "agg_key" in self.param.keys():
            self.agg_key = self.param["agg_key"]

        # define aggregated number of results
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
        self.agg_results = {}

    def main_process(self):
        """ Main function of dyda component. """

        self.results = []
        self.pack_input_as_list()
        input_data = copy.deepcopy(self.input_data)

        for i in range(0, len(input_data)):
            # If there exists same filename, the df index will be the same and
            # cause inception. It's safer to remove filename in copied input
            input_data[i]["filename"] = ""

            # aggregate results seperately for each input_data
            if i not in self.agg_results.keys():
                self.agg_results[i] = []

            # drop the oldest one when a new data comes
            self.agg_results[i].append(input_data[i])
            if len(self.agg_results[i]) > self.agg_num:
                self.agg_results[i] = self.agg_results[i][1:]

            # concat aggregated results as Pandas DataFrame and group by keys
            df = pandas_data.create_anno_df_and_concat(
                    copy.deepcopy(self.agg_results[i]), debug=False
            )
            sel_list = [self.group_key, self.agg_key]

            # calculate mean or sum
            # if obj(id=3) appear only in the first and the fifth detection
            # mean(id=3, fifth) = (result(id=3, first) + result(id=3, fifth))/2

            group_df = pandas_data.group_df(
                df[sel_list], [self.group_key], comp_rule=self.comp_rule
            )

            self.results.append(copy.deepcopy(self.input_data[i]))
            for anno in self.results[i]['annotations']:
                anno[self.agg_key] = group_df.loc[anno[self.group_key]][
                                        self.agg_key]

        self.unpack_single_results()


class ImageAccumulator(data_accumulator_base.AccumulatorBase):
    """ This component accumulates image arrays and reset the queue based on
        the reset_at parameter in dyda config """

    def __init__(self, dyda_config_path='', param=None):
        super(ImageAccumulator, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)
        self.reset_at = 10
        if "reset_at" in self.param.keys():
            self.reset_at = int(self.param["reset_at"])
        self.output_data = []

    def main_process(self):
        """ main_process of ImageAccumulator """

        input_data = self.uniform_input()
        if not isinstance(self.output_data, list):
            self.output_data = [self.output_data]

        if self.results["ncount"] == self.reset_at:
            self.results["ncount"] = 0
            self.output_data = []

        if self.results["ncount"] == 0:
            self.output_data = copy.deepcopy(input_data)
        else:
            for i, im in enumerate(input_data):
                self.output_data[i] = self.output_data[i] + im

        self.results["ncount"] = self.results["ncount"] + 1
        self.uniform_output()

    def reset_output_data(self):
        pass
