import os
import sys
import cv2
import numpy as np
from dyda_utils import data
from dyda_utils import tools
from dyda_utils import image
from dyda.core import data_converter_base
from dt42lab.utility import dict_comparator as dict_tools
from dyda.components import relabeler


class BatchRelabeler(data_converter_base.ConverterBase):
    def __init__(self):
        super(BatchRelabeler, self).__init__()
        self.deposit_list = list()
        self.config_data = {}
        self.use_key = True
        self.action = 0
        self.window_name = ''
        self.current_lbset = {}
        self.sv_path = ''
        self.json_paths = list()
        self.config_path = ''
        self.sv_flag = False
        self.key = ''

    def init_deposit_list(self, input_data):
        idx = 0
        self.input_data = input_data
        for labels in self.input_data:
            annots = labels["annotations"]
            for annot in annots:
                relabel_flag = self.deter_relabel(annot)
                if relabel_flag:
                    ixy = (annot["left"], annot["top"])
                    xy = (annot["right"], annot["bottom"])
                    labelid = self.config_data["class_info"][annot["label"]]
                    #relabel = self.deter.det_relabel()
                    filename = labels["filename"]
                    folder = labels["folder"]
                    img_path = os.path.join(folder, filename)

                    label_data = {'labelid': labelid, 'bbox': [ixy, xy],
                                  'relabel': relabel_flag,
                                  'annot': annot,
                                  'img_path': img_path}
                    rebler = relabeler.Relabeler()
                    rebler.lab_flag = self.lab_flag
                    rebler.fig_name = self.window_name
                    rebler.set_config(label_data,
                                      self.config_data,
                                      self.use_key)
                    deposit_data = {"idx": idx, "rebler": rebler}
                    self.deposit_list.append(deposit_data)
                    idx += 1

        self.current_lbset = self.get_lbset(0)

    def setFromDict(self, dataDict, mapList, value):
        last_layer = value
        for i in range(1, len(mapList)):
            tmp = dict_tools.getFromDict(dataDict, mapList[:-i])
            if isinstance(mapList[-i], list):
                tmp[mapList[-i][0]][mapList[-i][1]] = last_layer
            else:
                tmp[mapList[-i]] = last_layer
            last_layer = tmp

    def set_field_value(self, data, field, value):
        mapList_arr = list()
        dict_tools.get_mapList_arr(data, list(), mapList_arr)
        field_list = list()
        for mapList in mapList_arr:
            ref_val = dict_tools.getFromDict(data, mapList)
            query_flag = dict_tools.check_ignore(field, mapList)
            if query_flag:
                self.setFromDict(data, mapList, value)

    def get_field_value(self, data, field):
        mapList_arr = list()
        dict_tools.get_mapList_arr(data, list(), mapList_arr)
        field_list = list()
        for mapList in mapList_arr:
            ref_val = dict_tools.getFromDict(data, mapList)
            query_flag = dict_tools.check_ignore(field, mapList)
            if query_flag:
                return dict_tools.getFromDict(data, mapList)

    def deter_relabel(self, annot):
        return True 

    def deposit_result(self, deposit_data):
        idx = deposit_data["idx"]
        self.deposit_list[idx] = deposit_data

    def get_lbset(self, idx):
        rebler = self.deposit_list[idx]["rebler"]
        if self.lab_flag:
            rebler.update_disp_img()
            if not self.use_key:
                cv2.setMouseCallback(rebler.fig_name,
                                     rebler.mousce_action)
        return self.deposit_list[idx]

    def init_disp_window(self):
        length = len(self.deposit_list)-1
        print(length)
        cv2.namedWindow(self.window_name)
        trackbar_name = 'deposit list idx'
        cv2.createTrackbar(trackbar_name, self.window_name, 0,
                           length, self.onchange)

    def onchange(self, trackbarValue):
        self.pick_lbset(trackbarValue)

    def pick_lbset(self, idx):
        if idx < len(self.deposit_list):
            new_lbset = self.get_lbset(idx)
            current_idx = self.current_lbset["idx"]
            if current_idx != idx:
                self.deposit_result(self.current_lbset)
                self.current_lbset = new_lbset

    def deposit2output(self):
        idx = 0
        self.output_data = list(self.input_data)
        for labels in self.output_data:
            annots = labels["annotations"]
            rm_list = list()
            for annot in annots:
                relabel_flag = self.deter_relabel(annot)
                if relabel_flag:
                    rebler = self.deposit_list[idx]["rebler"]
                    label_data = rebler.output_data
                    labelid = label_data["labelid"]
                    class_num = self.config_data["class_number"]
                    if labelid > class_num:
                        rm_list.append(annot)
                    label_name = self.get_labelname(labelid)
                    annot["label"] = label_name
                    annot["confidence"] = 1

                    idx += 1

            labels["annotations"] = (
                [x for x in labels["annotations"] if x not in rm_list])

        self.results["additional_info"] = self.output_data

    def get_labelname(self, labelid):
        name_dict = self.config_data["class_info"]
        for key, val in name_dict.items():
            if labelid == val:
                return key

    def set_change_flag(self, idx):
        self.pick_lbset(idx)
        self.current_lbset["rebler"].set_change_flag()

    def set_remove_flag(self, idx):
        self.pick_lbset(idx)
        self.current_lbset["rebler"].set_remove_flag()

    def pre_process(self):
        self.current_lbset["rebler"].key = self.key

    def main_process(self):
        self.results['convert_type'] = 'change_batch_label'
        self.current_lbset["rebler"].run()
        if self.sv_flag:
            self.deposit2output()

    def post_process(self):
        if self.sv_flag:
            tools.check_dir(self.sv_folder)
            for data in self.output_data:
                sv_path = os.path.join(self.sv_folder, data["filename"])
                sv_path = tools.replace_extension(sv_path, '.json')
                tools.write_json(data, sv_path)
                dict_tools.restore_ordered_json(sv_path)
