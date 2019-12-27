"""
Updated 2018/02/15 by Tammy Yang

Functions of this module has been merged into dyda_utils.lab_tools.
The file is kept for preserving the compatibility.
There should NOT be any new functions or changed added into this file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
import copy
import cv2
import math
import random
import numpy as np
import matplotlib.cm as cm

from dyda_utils import image
from dyda_utils import data
from dyda_utils import tools


def box_append(
        filename,
        annotations):
    """ Append annotations to the file

    @param filename: filename of the file appended

    @param annotations: annotations of a detection result

    """

    if os.path.isfile(filename):
        base = data.parse_json(filename)
    else:
        base = []
    for i in range(len(annotations)):
        base.append(annotations[i])
    data.write_json(base, filename)


def box_interpolate(
    index_start,
    json_data_start,
    index_end,
    json_data_end,
    index_inter
):
    """ Bounding box interpolation

    @param index_start: index of the start frame

    @parme json_data_start: a detection result with following informations
        {
        'label': ,
        'top': ,
        'bottom': ,
        'left': ,
        'right' ,
        'confidence'
        }

    @param index_end: index of the end frame

    @parme json_data_end: a detection result with the same informations
        as json_data_start

    @param index_inter: index of the interpolated frame

    @return interpolate_result: interpolation results
        {
        'label': ,
        'top': ,
        'bottom': ,
        'left': ,
        'right' ,
        'confidence'
        }
    """

    interpolate_result = {
        'label': json_data_start['label'],
        'top': int(interpolate(
            index_start,
            index_inter,
            index_end,
            json_data_start['top'],
            json_data_end['top'])),
        'bottom': int(interpolate(
            index_start,
            index_inter,
            index_end,
            json_data_start['bottom'],
            json_data_end['bottom'])),
        'left': int(interpolate(
            index_start,
            index_inter,
            index_end,
            json_data_start['left'],
            json_data_end['left'])),
        'right': int(interpolate(
            index_start,
            index_inter,
            index_end,
            json_data_start['right'],
            json_data_end['right'])),
        'confidence': interpolate(
            index_start,
            index_inter,
            index_end,
            json_data_start['confidence'],
            json_data_end['confidence'])
    }
    return interpolate_result


def interpolate(
    index_start,
    index_inter,
    index_end,
    value_start,
    value_end
):
    """ Interpolation

    """

    value_inter = value_start + (value_end - value_start) * \
        (index_inter - index_start) / (index_end - index_start)
    return(value_inter)


def square_extend_in_json(
        json_data):
    """ Extend location of bounding box in json to square

    @param json_data: json data from detector result

    @return json_data: json data after extended

    """

    annotations = json_data['annotations']
    width = json_data['size']['width']
    height = json_data['size']['height']
    for i in range(len(annotations)):
        loc = (annotations[i]['top'],
               annotations[i]['bottom'],
               annotations[i]['left'],
               annotations[i]['right'])
        out_loc = square_extend(loc, width, height)
        (annotations[i]['top'],
         annotations[i]['bottom'],
         annotations[i]['left'],
         annotations[i]['right']) = out_loc
    json_data['annotations'] = annotations
    return json_data


def square_extend(loc, width, height):
    """ Extend location of bounding box to square

    @param loc: (top, bottom, left, right) of bounding box
    @param width: width of original image
    @param height: height of original image

    @return out_loc: extended location of bounding box

    """
    (top, bottom, left, right) = loc
    half_length = max(bottom - top, right - left) / 2
    center_y = (bottom + top) / 2
    center_x = (right + left) / 2
    out_loc = (
        int(max(center_y - half_length, 0)),
        int(min(center_y + half_length, height - 1)),
        int(max(center_x - half_length, 0)),
        int(min(center_x + half_length, width - 1)))
    return out_loc


def combined_json(
        detection_json_dir,
        classification_json_dir):
    """ Combine results from detector and classifier

    @param detection_json_dir: directory to detector results
    @param classification_json_dir: directory to classifier results

    """

    for json_file in tools.find_files(detection_json_dir):
        json_data = data.parse_json(json_file)
        annotations = json_data['annotations']
        for i in range(len(annotations)):
            classification_json_file = os.path.join(
                classification_json_dir,
                os.path.basename(json_file).split('.')[0]
                + '_' + str(i) + '.json')
            if os.path.exists(classification_json_file):
                classification_annotations = data.parse_json(
                    classification_json_file)['annotations'][0]
                annotations[i]['type'] = 'detection_classification'
                annotations[i]['label'] += '_' + \
                    classification_annotations['label']
                annotations[i]['labinfo'] = classification_annotations[
                    'labinfo']
        json_data['annotations'] = annotations
        out_json = os.path.join(
            os.path.dirname(json_file),
            os.path.basename(json_file).split('.')[0] + '_combined.json')
        with open(out_json, 'w') as wf:
            json.dump(json_data, wf)


def extract_target_value(
        json_data,
        target_key,
        target_value):
    """ Extract target class in detector results

    @param json_data: json data from detector result
    @param target_key: target key to extract
    @param target_value: target value to extract

    @return json_data: json data of detector results only with target value

    """

    annotations = json_data['annotations']
    for i in range(len(annotations) - 1, -1, -1):
        if target_key not in annotations[i].keys():
            annotations.pop(i)
        elif not annotations[i][target_key] == target_value:
            annotations.pop(i)
    json_data['annotations'] = annotations
    return json_data


def extract_target_class(
        json_data,
        target_class):
    """ Extract target class in detector results

    @param json_data: json data from detector result
    @param target_class: target class to extract

    @return json_data: json data of detector results only with target class

    """

    annotations = json_data['annotations']
    for i in range(len(annotations) - 1, -1, -1):
        if not annotations[i]['label'] == target_class:
            annotations.pop(i)
    json_data['annotations'] = annotations
    return json_data


def reverse_padding(
        in_json,
        shift,
        size,
        padded_dir,
        frame_dir):
    """ Reverse detection result of padded image
    to detection result fo original image

    @param in_json: input json filename
    @param shift: (shift_x, shift_y)
    @param size: (width, height), size of original image
    @param padded_dir: directory of padded images
    @param frame_dir: directory of original images

    """

    shift_x = shift[0]
    shift_y = shift[1]
    width = size[0]
    height = size[1]
    json_data = data.parse_json(in_json)
    if json_data['folder'] == padded_dir:
        json_data = shift_boxes(json_data, (shift_x, shift_y), (width, height))
        json_data['folder'] = frame_dir
        with open(in_json, 'w') as wf:
            json.dump(json_data, wf)


def shift_boxes(json_data, shift, size):
    """Shift the bounding box in a json file from a detector.
    The lacation of bounding box (x, y) is shifted to (x+shift_x, y+shift_y).

    @param json_data: json_data of detection results before shifting
    @param shift: (shift_x, shift_y)
    @param size: (width, height), size of original image

    @return json_data: json_data of detection results after shifting

    """

    shift_x = shift[0]
    shift_y = shift[1]
    width = size[0]
    height = size[1]
    annotations = json_data['annotations']
    for ai in range(len(annotations)):
        annotations[ai]['top'] = max(0,
                                     int(annotations[ai]['top'] + shift_y))
        annotations[ai]['bottom'] = min(
            height - 1, int(annotations[ai]['bottom'] + shift_y))
        annotations[ai]['left'] = max(0,
                                      int(annotations[ai]['left'] + shift_x))
        annotations[ai]['right'] = min(width - 1,
                                       int(annotations[ai]['right'] + shift_x))
    json_data['annotations'] = annotations
    return json_data


def shrink_boxes(json_data, shrink_perc, shift_to_pad=False, verbose=False):
    """Shrink the bounding box in a json file from a detector.

    @param json_data: json_data of detection results before shrinking
    @param shrink_perc: percentage to shrink
    @param shift_to_pad: True to shift box to padded results

    @return json_data: json_data of detection results after shifting

    """

    from dyda_utils import lab_tools
    annotations = json_data['annotations']
    new_results = copy.deepcopy(json_data)
    for ai in range(0, len(annotations)):
        anno = annotations[ai]
        rect = lab_tools.conv_lab_anno_to_rect(anno)

        delta_w = int(float(rect.w) * shrink_perc / 2)
        delta_h = int(float(rect.h) * shrink_perc / 2)

        min_len = min(rect.h, rect.w)
        max_len = max(rect.h, rect.w)
        if shift_to_pad:
            if rect.h > rect.w:
                delta_w = max(delta_w, delta_w + int((max_len - min_len)/2))
            else:
                delta_h = max(delta_h, delta_h + int((max_len - min_len)/2))

        new_t = rect.t + delta_h
        new_b = rect.b - delta_h
        if new_t < new_b:
            new_results["annotations"][ai]["top"] = new_t
            new_results["annotations"][ai]["bottom"] = new_b
        else:
            if verbose:
                print("[dt42lab] new_t >= new_b, no shrink apply to y axis")
        new_l = rect.l + delta_w
        new_r = rect.r - delta_w
        if new_l < new_r:
            new_results["annotations"][ai]["left"] = new_l
            new_results["annotations"][ai]["right"] = new_r
        else:
            if verbose:
                print("[dt42lab] new_l >= new_r, no shrink apply to x axis")
    return new_results


def resize_boxes_in_json(json_data, resize_ratio):
    """Resize the bounding box in a json file from a detector.
    The resize_ratio is (length in out_json) / (length in in_json).

    @param json_data: json data of detection results before resize
    @param resize_ratio: resize ratio

    @return json_data: json data of detection results after resize
    """

    annotations = json_data['annotations']
    for ai in range(len(annotations)):
        annotations[ai]['top'] = int(
            float(annotations[ai]['top']) * resize_ratio)
        annotations[ai]['left'] = int(
            float(annotations[ai]['left']) * resize_ratio)
        annotations[ai]['bottom'] = int(
            float(annotations[ai]['bottom']) * resize_ratio)
        annotations[ai]['right'] = int(
            float(annotations[ai]['right']) * resize_ratio)
    json_data['annotations'] = annotations
    return json_data


def grouping(json_data, grouping_ratio=0.5):
    """Group bounding_box when horizontally close and vertically overlap.

    @param json_data: json data of detectoin results befor grouping
    @param grouping_ratio: ration of overlap used to decide group or not

    @return json_data: json data of detectoin results after grouping
    """

    annotations = json_data['annotations']
    bounding_box, confidence = annotations_to_boxes(annotations)
    number = [1] * len(annotations)

    if bounding_box.shape[0] < 2:
        return json_data

    change = True
    while change:
        person_number = bounding_box.shape[0]
        change = False
        group = np.array(range(person_number))
        for i in range(0, person_number - 1):
            for j in range(i + 1, person_number):
                right_i = bounding_box[i][3]
                right_j = bounding_box[j][3]
                left_i = bounding_box[i][2]
                left_j = bounding_box[j][2]
                bottom_i = bounding_box[i][1]
                bottom_j = bounding_box[j][1]
                top_i = bounding_box[i][0]
                top_j = bounding_box[j][0]
                person_width = max(right_i - left_i, right_j - left_j)
                diff_rl = float(max(right_i, right_j) - min(left_i, left_j))
                diff_tb = min(bottom_i, bottom_j) - max(top_i, top_j)
                # group bounding_box when
                # 1) horizontally close enough:
                #       diff_rl < person_width * (2 + grouping_ratio)
                # 2) vertically overlap: diff_tb > person_width
                # *grouping_ratio.
                if (diff_rl < person_width * (2 + grouping_ratio)
                        and diff_tb) > person_width * grouping_ratio:
                    group[j] = group[i]
                    change = True

        out_bounding_box = np.array([], dtype=np.int).reshape(0, 4)
        out_confidence = []
        out_number = []
        for i in range(0, person_number):
            index = (group == i).nonzero()[0]
            if len(index) > 0:
                left = bounding_box[index[0]][2]
                top = bounding_box[index[0]][0]
                right = bounding_box[index[0]][3]
                bottom = bounding_box[index[0]][1]
                score = confidence[index[0]]
                num = number[index[0]]
                for j in range(1, len(index)):
                    left = min(bounding_box[index[j]][2], left)
                    top = min(bounding_box[index[j]][0], top)
                    right = max(bounding_box[index[j]][3], right)
                    bottom = max(bounding_box[index[j]][1], bottom)
                    score = max(confidence[index[j]], score)
                    num = number[index[j]] + num
                out_bounding_box = np.row_stack(
                    [out_bounding_box, [top, bottom, left, right]])
                out_confidence.append(score)
                out_number.append(num)
        bounding_box = out_bounding_box
        confidence = out_confidence
        number = out_number

    data_all = []
    for di in range(len(number)):
        data = {
            "type": "detection",
            "label": "person",
            "person_number": number[di],
            "confidence": confidence[di],
            "top": bounding_box[di][0],
            "bottom": bounding_box[di][1],
            "left": bounding_box[di][2],
            "right": bounding_box[di][3],
            "id": -1}
        data_all.append(data)
    json_data['annotations'] = data_all
    return json_data


def annotations_to_boxes(json_data):
    """Extract bounding boxes from json data.

    @param json_data: json data to be extracted

    @return bounding_bex: bounding boxes in json data
    @return confidence: confidence scores in json data
    """

    bounding_box = np.array([], dtype=np.int).reshape(0, 4)
    confidence = []
    for di in range(len(json_data)):
        bounding_box = np.row_stack([
            bounding_box,
            [
                json_data[di]['top'],
                json_data[di]['bottom'],
                json_data[di]['left'],
                json_data[di]['right']
            ]
        ])
        confidence.append(json_data[di]['confidence'])
    return bounding_box, confidence


def sort_by_area(
        json_data,
        is_multi_channel=False,
        channel_index=[]):
    """Sort bounding boxes by area from largest to smallest.

    @param json_data: json data of detection results before sorting
    @param is_multi_channel: multi channel or not
    @param channel_index: channel index for multi channel

    @return json_data: json data of detection results after sorting
    """

    annotations = json_data['annotations']
    bounding_box = annotations_to_boxes(annotations)[0]

    out_bounding_box = bounding_box
    person_number = len(bounding_box)
    z = np.zeros((person_number, 1), int)
    bounding_box = np.c_[bounding_box, z]
    for i in range(0, person_number):
        bounding_box[i][4] = ((bounding_box[i][1] - bounding_box[i][0]) *
                              (bounding_box[i][3] - bounding_box[i][2]) * (-1))
    order = np.argsort(bounding_box[:, 4])
    json_data['annotations'] = [annotations[x] for x in order]

    if is_multi_channel:
        channel_index = [channel_index[x] for x in order]
        for i in range(0, person_number):
            json_data['annotations'][i]['channel_index'] = channel_index[i]
    return json_data


def sort_by_aspect_ratio(
        json_data,
        is_multi_channel=False,
        channel_index=[]):
    """Sort bounding_box by aspect_ratio(width/height)-
    from largest to smallest.

    @param json_data: json data of detection results before sorting
    @param is_multi_channel: multi channel or not
    @param channel_index: channel index for multi channel

    @return json_data: json data of detection results after sorting

    """

    annotations = json_data['annotations']
    bounding_box = annotations_to_boxes(annotations)[0]

    out_bounding_box = bounding_box
    person_number = len(bounding_box)
    z = np.zeros((person_number, 1), float)
    bounding_box = np.c_[bounding_box, z]
    for i in range(0, person_number):
        height = float(bounding_box[i][1] - bounding_box[i][0])
        width = float(bounding_box[i][3] - bounding_box[i][2])
        bounding_box[i][4] = width / height
    order = np.argsort(bounding_box[:, 4] * -1)
    json_data['annotations'] = [annotations[x] for x in order]

    if is_multi_channel:
        channel_index = [channel_index[x] for x in order]
        for i in range(0, person_number):
            json_data['annotations'][i]['channel_index'] = channel_index[i]

    return json_data
