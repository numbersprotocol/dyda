import os
import json
import numbers
import numpy as np
import argparse
from dyda_utils import tools
from functools import reduce
import collections


def restore_ordered_json(json_path):
    json_data = tools.parse_json(json_path)
    with open(json_path, 'w') as outfile:
        outfile.write(json.dumps(json_data,
                      indent=4, sort_keys=True))


def get_mapList_arr(json_data, i_mapList, mapList_arr):
    if isinstance(json_data, list):
        idx = 0
        for x in json_data:
            mapList = list(i_mapList)
            mapList.append(idx)
            get_mapList_arr(x, mapList, mapList_arr)
            idx += 1
    elif isinstance(json_data, dict):
        for k, v in json_data.items():
            if isinstance(v, dict):
                mapList = list(i_mapList)
                mapList.append(k)
                get_mapList_arr(v, mapList, mapList_arr)
            elif isinstance(v, list):
                idx = 0
                for x in v:
                    mapList = list(i_mapList)
                    mapList.append([k, idx])
                    get_mapList_arr(x, mapList, mapList_arr)
                    idx += 1
            else:
                mapList = list(i_mapList)
                mapList.append(k)
                mapList_arr.append(mapList)
    else:
        mapList = list(i_mapList)
        mapList_arr.append(mapList)


def getFromDict(dataDict, mapList):
    return reduce(getitem, mapList, dataDict)


def getitem(dataDict, idx):
    if isinstance(idx, list):
        return dataDict[idx[0]][idx[1]]
    else:
        return dataDict[idx]

def setFromDict(dataDict, mapList, value):
    last_layer = value
    for i in range(1,len(mapList)):
        tmp = getFromDict(dataDict, mapList[:-i])
        if isinstance(mapList[-i], list):
            tmp[mapList[-i][0]][mapList[-i][1]] = last_layer
        else:
            tmp[mapList[-i]] = last_layer
        last_layer = tmp

def init_error_dict():
    error_dict = {"ref_key": [], "ref_val": [],
                  "tar_key": [], "tar_val": []}
    return error_dict


def init_status_dict():
    status_dict = {"mismatch_val": [], "missing_field": [],
                   "extra_field": []}
    return status_dict

def get_field_value(data, field):
    mapList_arr = list()
    get_mapList_arr(data, list(), mapList_arr)
    field_list = list()
    for mapList in mapList_arr:
        ref_val = getFromDict(data, mapList)
        query_flag = check_ignore(field, mapList)
        if query_flag:
            return getFromDict(data, mapList)


def convert2dict(unknowtype, isref):
    o_dict = []
    if isinstance(unknowtype, list):
        o_dict = unknowtype
    elif isinstance(unknowtype, dict):
        o_dict = unknowtype
    elif os.path.isfile(unknowtype):
        o_dict = tools.parse_json(unknowtype)
    else:
        print("Not a valid input")
        if isref:
            print("There does not exists file on ref_json_path")
        else:
            print("There does not exists file on tar_json_path")
    return o_dict


def check_ignore(ignore_keys, mapList):
    inlist = False
    for element in mapList:
        if isinstance(element, list):
            inlist = check_ignore(ignore_keys, element)
        else:
            if element in ignore_keys:
                inlist = True

    return inlist


def get_diff(ref, tar, precision=2, ignore_keys=[]):
    # ref and tar can be json_path or dictionary
    ref_dict = convert2dict(ref, isref=True)
    tar_dict = convert2dict(tar, isref=False)

    status_dict = init_status_dict()
    mapList_arr = list()
    get_mapList_arr(ref_dict, list(), mapList_arr)
    for mapList in mapList_arr:
        ref_val = getFromDict(ref_dict, mapList)
        try:
            tar_val = getFromDict(tar_dict, mapList)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            err_dict = init_error_dict()
            err_dict['ref_key'] = mapList
            err_dict['ref_val'] = ref_val
            status_dict['missing_field'].append(err_dict)
            continue

        ignore_flag = check_ignore(ignore_keys, mapList)

        if ref_val != tar_val and not ignore_flag:
            err_dict = init_error_dict()
            mismatch_flag = False
            if isinstance(ref_val, numbers.Number) and\
               isinstance(tar_val, numbers.Number):
                rtol = 10**(-1*precision)
                if abs(tar_val-ref_val) > rtol:
                    mismatch_flag = True
            else:
                mismatch_flag = True

            if mismatch_flag:
                err_dict['ref_key'] = mapList
                err_dict['ref_val'] = ref_val
                err_dict['tar_key'] = mapList
                err_dict['tar_val'] = tar_val
                status_dict['mismatch_val'].append(err_dict)

    mapList_arr = list()
    get_mapList_arr(tar_dict, list(), mapList_arr)
    for mapList in mapList_arr:
        tar_val = getFromDict(tar_dict, mapList)
        try:
            ref_val = getFromDict(ref_dict, mapList)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            err_dict = init_error_dict()
            err_dict['tar_key'] = mapList
            err_dict['tar_val'] = tar_val
            status_dict['extra_field'].append(err_dict)
            continue
    return status_dict


def show_report(report):
    report = collections.OrderedDict(sorted(report.items()))
    for k, v in report.items():
        print(k)
        for err in v:
            print("ref: ", err["ref_key"], err["ref_val"])
            print("tar: ", err["tar_key"], err["tar_val"])


def compare(out_path=None, ref_path=None,
            tar_path=None, show=True,
            precision=2, ignore_keys=[]):
    """ compare two json file """

    if ref_path is None:
        from dyda_utils import lab_tools
        ref_url = ('https://gitlab.com/DT42/galaxy42/dt42-lab-lib/uploads/'
                   '5a82dca00757a21c82c681d7c8a8b773/cls_ref.json')
        ref_path = '/tmp/cls_ref.json'
        _ref = lab_tools.pull_json_from_gitlab(ref_url, save_to=ref_path)
    else:
        ref_path = ref_path
    if tar_path is None:
        from dyda_utils import lab_tools
        tar_url = ('https://gitlab.com/DT42/galaxy42/dt42-lab-lib/uploads/'
                   '8c755ccdc97494bab294e5706ca738f7/cls_tar.json')
        tar_path = '/tmp/cls_tar.json'
        _tar = lab_tools.pull_json_from_gitlab(tar_url, save_to=tar_path)
    else:
        tar_path = tar_path

    report = get_diff(ref_path, tar_path, precision, ignore_keys)
    if show:
        show_report(report)
    else:
        pass

    if out_path is not None:
        tools.write_json(report, out_path)
    else:
        pass
