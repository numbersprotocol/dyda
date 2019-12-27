'''
lab_tools module provides the functions to help users meet spec
defined by the dt42lab spec and the trainer spec
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import requests
import json
import os
import cv2
import copy
import numpy as np

from dyda_utils import image
from dyda_utils import data
from dyda_utils import tools
from dyda_utils import tinycv


def tf_label_map_to_dict(label_map_path, nline_in_pack=5, key="id"):
    """Convert TensorFlow label_map to dict

    The label_map should match the format

        item {
            name: "/m/01g317"
            id: 1
            display_name: "person"
        }

    @param label_map_path: Path of the label map file
    @param nline_in_pack: How many lines should be counted as one pack
    @param key: The default key for finding the unique id

    @return label_dict

    """
    map_list = tools.txt_to_list(label_map_path)
    label_dict = {}
    label_item = {}
    label_id = None
    for i, _item in enumerate(map_list):
        if i % nline_in_pack == 0 and label_id is not None:
            label_dict[label_id] = label_item
            label_item = {}
        item_list = _item.replace("\"", "").replace(":", "").split(" ")
        # Only proceed if both key and value are found
        if len(item_list) < 2:
            continue
        if item_list[0] == key:
            label_id = int(item_list[1])
        label_item[item_list[0]] = item_list[1]
    label_dict[label_id] = label_item
    return label_dict


def pull_json_from_gitlab(json_url, save_to="",
                          token_path="./gitlab_token.json"):
    """ Pull json from gitlab issue attachment """

    token = get_gitlab_token(token_path)
    headers = {'PRIVATE-TOKEN': token}
    response = requests.get(json_url, headers=headers)
    status = response.status_code

    if status == 200:
        try:
            json_content = response.json()
            if len(save_to) > 1:
                try:
                    tools.write_json(json_content, fname=save_to)
                except BaseException:
                    print('[dt42lab] ERROR: Fail to write output.')
                    raise
            return json_content

        except BaseException:
            print('[dt42lab] ERROR: Fail to get json from gitlab.'
                  'Check if token is set correctly or if the url is right')
            sys.exit(1)
    else:
        print('[dt42lab] ERROR: Fail with status_code %i.' % status)
        sys.exit(1)


def pull_img_from_gitlab(img_url, save_to="",
                         token_path="./gitlab_token.json"):
    """ Pull json from gitlab issue attachment """

    token = get_gitlab_token(token_path)
    headers = {'PRIVATE-TOKEN': token}
    response = requests.get(img_url, headers=headers, stream=True)
    status = response.status_code

    if status == 200:
        try:
            raw = bytearray(response.content)
            img = cv2.imdecode(np.array(raw), flags=1)
            if len(save_to) > 1:
                try:
                    cv2.imwrite(save_to, img)
                except BaseException:
                    print('[dt42lab] ERROR: Fail to save image.')
                    raise
            return img
        except BaseException:
            print('[dt42lab] ERROR: Cannot get cv2 array correctly.')
            raise
    else:
        print('[dt42lab] ERROR: Fail with status_code %i.' % status)
        sys.exit(1)


def get_gitlab_token(token_path):
    """Read gitlab token from the given path
    The content of the token json should be {'token': $TOKEN}

    @param token_path: Path of the gitlab token

    """
    _bexist = tools.check_exist(token_path, log=False)
    if _bexist:
        try:
            content = tools.parse_json(token_path)
            if 'token' in content.keys():
                if isinstance(content['token'], str):
                    return content['token']
                else:
                    print('[dt42lab] ERROR: Please check token'
                          ' in %s' % token_path)
                    sys.exit(1)
            else:
                print('[dt42lab] ERROR: Cannot fine "token"'
                      ' key in %s' % token_path)
                sys.exit(1)
        except BaseException:
            print('[dt42lab] ERROR: %s exists, but cannot'
                  'be read.' % token_path)
            sys.exit(1)
    else:
        try:
            token = os.environ['CI_JOB_TOKEN']
            return token
        except KeyError:
            print('[dt42lab] ERROR: No token file or CI_JOB_TOKEN found.')
            sys.exit(1)


def _lab_annotation_dic():
    """ Return empty dic of lab annotation """

    empty_anno = {
        "type": "",
        "id": -1,
        "label": "",
        "top": -1,
        "bottom": -1,
        "left": -1,
        "right": -1,
        "confidence": -1.0,
        "track_id": -1,
        "rot_angle": -1.0,
        "labinfo": {}
    }
    return empty_anno


def _output_pred(input_path, img_size=[], timestamp=None):
    """ Output prediction result based on dt42lab spec https://goo.gl/So46Jw

    @param input_path: File path of the input

    Arguments:

    img_size -- List of image size, dimension should be 2, such as [128, 128]

    """

    real_file_exist = tools.check_exist(input_path, log=False)
    input_file = ""
    folder = ""

    input_file = os.path.basename(input_path)
    folder = os.path.dirname(input_path)

    if len(img_size) == 2:
        input_size = img_size
    else:
        if real_file_exist:
            input_size = image.get_img_info(input_path)[0]
        else:
            input_size = [-1, -1]

    timestamp_str = tools.create_timestamp(datetime_obj=timestamp)
    pred_info = {
        "filename": input_file, "folder": folder, "timestamp": timestamp_str
    }
    pred_info["size"] = {"width": input_size[0], "height": input_size[1]}
    pred_info["annotations"] = []
    # According to the discussion with dev team, suggest not to cal shasum
    # pred_info["sha256sum"] = get_sha256(input_path)

    return pred_info


def output_pred_classification(input_path, conf, label, img_size=[],
                               labinfo={}, save_json=False, timestamp=None):
    """ Output classification result based on spec https://goo.gl/So46Jw

    @param input_path: File path of the input
    @param conf: Confidence score
    @param label: Label of the result

    Arguments:

    img_size  -- List of image size, dimension should be 2, such as [128, 128]
    labinfo   -- Additional results
    save_json -- True to save output json file to {$FOLDER/$FILENAME}.json

    """

    pred_info = _output_pred(
        input_path, img_size=img_size, timestamp=timestamp
    )

    result = {
        "type": "classification",
        "id": 0,
        "label": label,
        "top": 0,
        "bottom": pred_info["size"]["height"],
        "left": 0,
        "right": pred_info["size"]["width"],
        "confidence": conf,
        "labinfo": labinfo
    }
    pred_info["annotations"] = [result]

    if save_json:
        json_file = os.path.join(
            pred_info["folder"],
            pred_info["filename"].split('.')[0] + '.json'
        )
        tools.write_json(pred_info, fname=json_file)

    return pred_info


def output_pred_detection(input_path, annotations, img_size=[],
                          labinfo={}, save_json=False,
                          anno_in_lab_format=False):
    """ Output detection result based on spec https://goo.gl/So46Jw

    @param input_path: File path of the input
    @param annotations: A list of annotations [[label, conf, bb]]
                        where bb is [top, bottom, left, right]

    Arguments:

    img_size  -- List of image size, dimension should be 2, such as [128, 128]
    labinfo   -- Additional results
    save_json -- True to save output json file to {$FOLDER/$FILENAME}.json

    """

    pred_info = _output_pred(input_path, img_size=img_size)

    if anno_in_lab_format:
        pred_info["annotations"] = annotations
    else:
        idx = 0
        for anno in annotations:
            check_detection_anno(anno)
            bb = anno[2]
            result = {
                "type": "detection",
                "id": idx,
                "label": anno[0],
                "top": bb[0],
                "bottom": bb[1],
                "left": bb[2],
                "right": bb[3],
                "confidence": anno[1],
                "labinfo": labinfo
            }
            pred_info["annotations"].append(result)
            idx += 1

    if save_json:
        json_file = os.path.join(
            pred_info["folder"],
            pred_info["filename"].split('.')[0] + '.json'
        )
        tools.write_json(pred_info, fname=json_file)

    return pred_info


def check_detection_anno(anno):
    """ Check if it is a valid annotation of detection output """

    if not isinstance(anno, list):
        print("[dt42lab] ERROR: Input annotation is not a list")
        return False
    if len(anno) != 3:
        print("[dt42lab] ERROR: Not a valid annotation (len(bb) != 3)")
        return False
    if not isinstance(anno[0], str):
        print("[dt42lab] ERROR: The first element of annotation should be"
              " a string of detection output label")
        return False
    if not isinstance(anno[1], float):
        print("[dt42lab] ERROR: The second element of annotation should be"
              " a float of detection output score")
        return False
    bb = anno[2]
    if not isinstance(bb, list):
        print("[dt42lab] ERROR: Input bounding box is not a list")
        return False
    if len(bb) != 4:
        print("[dt42lab] ERROR: Not a valid bb (len(bb) != 4)")
        return False
    for member in bb:
        if not isinstance(member, int):
            print("[dt42lab] ERROR: Not a valid bb, all top, bottom, left,"
                  " right should be integers")
            return False
    if bb[1] < bb[0]:
        print("[dt42lab] ERROR: Not a valid bb (bottom < top)")
        return False
    if bb[3] < bb[2]:
        print("[dt42lab] ERROR: Not a valid bb (right < left)")
        return False


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


def box_interpolate(index_start, json_data_start, index_end,
                    json_data_end, index_inter):

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


def interpolate(index_start, index_inter, index_end, value_start, value_end):
    """ Interpolation

    """

    value_inter = value_start + (value_end - value_start) * \
        (index_inter - index_start) / (index_end - index_start)

    return value_inter


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
                os.path.basename(
                    json_file).split('.')[0] + '_' + str(i) + '.json'
            )
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


def delete_target_value(
        json_data,
        target_key,
        target_value):
    """ Delete target class in detector results

    @param json_data: json data from detector result
    @param target_key: target key to delete
    @param target_value: target value to delete

    @return json_data: json data of detector results without target value

    """

    if not isinstance(target_value, list):
        target_value = [target_value]

    annotations = json_data['annotations']
    for i in range(len(annotations) - 1, -1, -1):
        if target_key in annotations[i].keys():
            if annotations[i][target_key] in target_value:
                annotations.pop(i)
    json_data['annotations'] = annotations
    return json_data


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
        elif not str(annotations[i][target_key]) == str(target_value):
            annotations.pop(i)
    json_data['annotations'] = annotations
    return json_data


def remove_target_value(
        json_data,
        target_key,
        target_value):
    """ Remove target class in detector results

    @param json_data: json data from detector result
    @param target_key: target key to remove
    @param target_value: target value to remove

    @return json_data: json data of detector results without target value

    """

    annotations = json_data['annotations']
    for i in range(len(annotations) - 1, -1, -1):
        if target_key in annotations[i].keys() and \
                str(annotations[i][target_key]) == str(target_value):
            annotations.pop(i)
    json_data['annotations'] = annotations
    return json_data


def extract_target_class(
        json_data,
        target_class):
    """ Extract target class in detector results

    @param json_data: json data from detector result
    @param target_class: list of target classes to extract

    @return json_data: json data of detector results only with target class

    """

    if isinstance(target_class, str):
        target_class = [target_class]
    annotations = json_data['annotations']
    for i in range(len(annotations) - 1, -1, -1):
        if not annotations[i]['label'] in target_class:
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


def resize_detection(detection, resize_ratio_h, resize_ratio_w):
    """Resize the bounding box in detection results.

    @param detection: detection result before resize
    @param resize_ratio_h: (height in output) / (height in input)
    @param resize_ratio_w: (width in output) / (width in input)

    @return detection: detection result after resize
    """

    annotations = detection['annotations']
    for ai in range(len(annotations)):
        annotations[ai]['top'] = int(
            float(annotations[ai]['top']) * resize_ratio_h)
        annotations[ai]['left'] = int(
            float(annotations[ai]['left']) * resize_ratio_w)
        annotations[ai]['bottom'] = int(
            float(annotations[ai]['bottom']) * resize_ratio_h)
        annotations[ai]['right'] = int(
            float(annotations[ai]['right']) * resize_ratio_w)
    detection['annotations'] = annotations
    return detection


def extend_detection(detection, ext_top, ext_bottom, ext_left, ext_right):
    """Extend the bounding box in detection results.

    @param detection: detection result before extension
    @param ext_top, ext_bottom, ext_left, ext_right:
        if the value < 1, the value means ratio, else, the value means pixel

    @return detection: detection result after extension
    """

    annotations = detection['annotations']
    im_height = detection['size']['height']
    im_width = detection['size']['width']
    for ai in range(len(annotations)):
        height = annotations[ai]['bottom'] - annotations[ai]['top']
        width = annotations[ai]['right'] - annotations[ai]['left']
        if ext_top < 1:
            ext_top_ = int(ext_top * height)
        else:
            ext_top_ = int(ext_top)
        if ext_bottom < 1:
            ext_bottom_ = int(ext_bottom * height)
        else:
            ext_bottom_ = int(ext_bottom)
        if ext_left < 1:
            ext_left_ = int(ext_left * width)
        else:
            ext_left_ = int(ext_left)
        if ext_right < 1:
            ext_right_ = int(ext_right * width)
        else:
            ext_right_ = int(ext_right)
        annotations[ai]['top'] = max(0, annotations[ai]['top'] - ext_top_)
        annotations[ai]['left'] = max(0, annotations[ai]['left'] - ext_left_)
        annotations[ai]['bottom'] = min(
            im_height, annotations[ai]['bottom'] + ext_bottom_)
        annotations[ai]['right'] = min(
            im_width, annotations[ai]['right'] + ext_right_)
    detection['annotations'] = annotations
    return detection


def shift_detection(detection, shift_h, shift_w, height=[], width=[]):
    """Shift the bounding box in detection result.

    @param detection: detection result before shifting
    @param shift_h: (top in output) = (top in input) + shift_h
                    (bottom in output) = (bottom in input) + shift_h
    @param shift_w: (left in output) = (left in input) + shift_w
                    (right in output) = (right in input) + shift_w
    @param height: (bottom in output) < height
    @param width: (right in output) < width

    @return detection: detection result after shifting

    """

    if width == []:
        width = detection['size']['width']
    if height == []:
        height = detection['size']['height']

    annotations = detection['annotations']
    for ai in range(len(annotations)):
        annotations[ai]['top'] = min(
            height - 1, max(0, int(annotations[ai]['top'] + shift_h)))
        annotations[ai]['bottom'] = max(0, min(
            height - 1, int(annotations[ai]['bottom'] + shift_h)))
        annotations[ai]['left'] = min(
            width - 1, max(0, int(annotations[ai]['left'] + shift_w)))
        annotations[ai]['right'] = max(
            0, min(width - 1, int(annotations[ai]['right'] + shift_w)))
    detection['annotations'] = annotations

    return detection


def shift_boxes(json_data, shift, size=[]):
    """Shift the bounding box in a json file from a detector.
    The lacation of bounding box (x, y) is shifted to (x+shift_x, y+shift_y).

    @param json_data: json_data of detection results before shifting
    @param shift: (shift_x, shift_y)
    @param size: (width, height), size of original image

    @return json_data: json_data of detection results after shifting

    """

    shift_x = shift[0]
    shift_y = shift[1]
    if size == []:
        width = json_data['size']['width']
        height = json_data['size']['height']
    else:
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


def flip_detection(json_data, direction):
    """Flip the bounding box in a json file from a detector.

    @param json_data: json_data of detection results before fliplr
    @param size: (width, height), size of original image

    @return json_data: json_data of detection results after fliplr

    """

    width = json_data['size']['width']
    height = json_data['size']['height']
    annotations = json_data['annotations']
    if direction == 'h':
        for ai in range(len(annotations)):
            left = annotations[ai]['left']
            right = annotations[ai]['right']
            annotations[ai]['left'] = width - 1 - right
            annotations[ai]['right'] = width - 1 - left
    elif direction == 'v':
        for ai in range(len(annotations)):
            top = annotations[ai]['top']
            bottom = annotations[ai]['bottom']
            annotations[ai]['top'] = height - 1 - bottom
            annotations[ai]['bottom'] = height - 1 - top

    json_data['annotations'] = annotations
    return json_data


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
    bounding_box, confidence, track_id = annotations_to_boxes(annotations)
    number = [1] * len(annotations)

    if bounding_box.shape[0] < 2:
        if bounding_box.shape[0] == 1:
            json_data['annotations'][0]['person_number'] = 1
            json_data['annotations'][0]['event_objects'] = [
                json_data['annotations'][0]['track_id']]
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
        out_track_id = []
        for i in range(0, person_number):
            index = (group == i).nonzero()[0]
            if len(index) > 0:
                left = bounding_box[index[0]][2]
                top = bounding_box[index[0]][0]
                right = bounding_box[index[0]][3]
                bottom = bounding_box[index[0]][1]
                score = confidence[index[0]]
                num = number[index[0]]
                if isinstance(track_id[index[0]], list):
                    tid = track_id[index[0]]
                else:
                    tid = [track_id[index[0]]]
                for j in range(1, len(index)):
                    left = min(bounding_box[index[j]][2], left)
                    top = min(bounding_box[index[j]][0], top)
                    right = max(bounding_box[index[j]][3], right)
                    bottom = max(bounding_box[index[j]][1], bottom)
                    score = max(confidence[index[j]], score)
                    num = number[index[j]] + num
                    tid.append(track_id[index[j]])
                out_bounding_box = np.row_stack(
                    [out_bounding_box, [top, bottom, left, right]])
                out_confidence.append(score)
                out_number.append(num)
                out_track_id.append(tid)
        bounding_box = out_bounding_box
        confidence = out_confidence
        number = out_number
        track_id = out_track_id

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
            "event_objects": track_id[di],
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
    track_id = []
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
        if 'track_id' in json_data[di].keys():
            track_id.append(json_data[di]['track_id'])
        else:
            track_id.append(-1)
    return bounding_box, confidence, track_id


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


def nus_with_size(detection_data, num_std=3):
    """ Non-unify suppression with bounding box size

    @param detection_data: annotations in detection result

    @return detection_data: annotations after nus

    """
    width_list = []
    height_list = []
    for bi in range(len(detection_data)):
        width_list.append(
            detection_data[bi]['right'] - detection_data[bi]['left'])
        height_list.append(
            detection_data[bi]['bottom'] - detection_data[bi]['top'])
    width_mean = np.mean(width_list)
    width_std = np.std(width_list)
    height_mean = np.mean(height_list)
    height_std = np.std(height_list)
    for bi in range(len(detection_data) - 1, -1, -1):
        width_diff = abs(width_list[bi] - width_mean)
        height_diff = abs(height_list[bi] - height_mean)
        if width_diff > num_std * width_std or \
                height_diff > num_std * height_std:
            detection_data.pop(bi)
    return detection_data


def nms_with_confidence(detection_data, threshold=0.3, nms_type='one_to_one'):
    """ Non-maximum suppression with confidence score

    @param detection_data: annotations in detection result
    @param threshold: only bounding box with highest confidence score left
         if overlap ratio > threshold
    @param nms_type: 'one_to_one' means one object suppress only one annother
         object; 'one_to_all' means one object could suppress all other
         occluded objects.

    @return detection_data: annotations after nms

    """
    overlap_ratio_all = calculate_overlap_ratio_all(
        detection_data,
        detection_data)
    data_number = len(detection_data)

    for di in range(data_number):
        overlap_ratio_all[di, di] = 0

    if len(overlap_ratio_all) > 0:
        max_value = overlap_ratio_all.max()
    else:
        max_value = 0

    suppression_list = []
    while max_value > threshold:
        max_index = overlap_ratio_all.argmax()
        index_1 = int(max_index / data_number)
        index_2 = int(max_index % data_number)
        if detection_data[index_1]['confidence'] < \
           detection_data[index_2]['confidence']:
            suppression_list.append(index_1)
        else:
            suppression_list.append(index_2)
        if nms_type == 'one_to_one':
            overlap_ratio_all[index_1, :] = 0
            overlap_ratio_all[:, index_2] = 0
            overlap_ratio_all[index_2, :] = 0
            overlap_ratio_all[:, index_1] = 0
        else:
            overlap_ratio_all[index_1, index_2] = 0
            overlap_ratio_all[index_2, index_1] = 0
        max_value = overlap_ratio_all.max()

    for di in sorted(list(set(suppression_list)), reverse=True):
        del detection_data[di]
    return detection_data


def calculate_color_hist(img, bin_num, max_val=[256, 256, 256]):
    """ Calculate color histogram. """

    chans = cv2.split(img)
    feat = []
    for i, chan in enumerate(chans):
        hist = cv2.calcHist([chan], [0], None, [bin_num], [0, max_val[i]])
        hist = cv2.normalize(hist, hist)
        feat.extend(hist)

    return np.asarray(feat)


def calculate_color_similarity_all(annos_1, annos_2, bin_num,
                                   method=cv2.HISTCMP_INTERSECT):
    """ Calculate similarity between all color histograms in
        annos_1 and annos_2 in which there are cropped images.

    @param annos_1: annotations in detection result with n bounding boxes
    @param annos_2: annotations in detection result with m bounding boxes

    @return D: n x m array with all distance

    """

    D = []
    number_1 = len(annos_1)
    number_2 = len(annos_2)

    for i in range(number_1):
        D_row = []
        for j in range(number_2):
            bb1 = tinycv.Rect([
                annos_1[i]['top'],
                annos_1[i]['bottom'],
                annos_1[i]['left'],
                annos_1[i]['right']])
            bb2 = tinycv.Rect([
                annos_2[j]['top'],
                annos_2[j]['bottom'],
                annos_2[j]['left'],
                annos_2[j]['right']])
            shift_x = min(bb1.w, bb2.w) / 2.0
            shift_y = min(bb1.h, bb2.h) / 2.0
            cent1 = [bb1.h / 2.0, bb1.w / 2.0]
            cent2 = [bb2.h / 2.0, bb2.w / 2.0]
            hist1 = calculate_color_hist(annos_1[i]['cropped_img'][
                int(cent1[0] - shift_y): int(cent1[0] + shift_y),
                int(cent1[1] - shift_x): int(cent1[1] + shift_x), :],
                bin_num)
            hist2 = calculate_color_hist(annos_2[j]['cropped_img'][
                int(cent2[0] - shift_y): int(cent2[0] + shift_y),
                int(cent2[1] - shift_x): int(cent2[1] + shift_x), :],
                bin_num)
            D_row.append(calculate_hist_similarity(hist1, hist2, method))
        D.append(D_row)

    return(np.array(D))


def calculate_hist_similarity(hist_1, hist_2, method=cv2.HISTCMP_INTERSECT):
    """ Calculate similarity between two histograms.

    """

    return cv2.compareHist(hist_1, hist_2, method)


def calculate_centroid_dist_all(json_data_1, json_data_2):
    """ Calculate centroid distance between all bounding boxes
    in json_data_1 and json_data_2

    @param json_data_1: annotations in detection result with n bounding boxes
    @param json_data_2: annotations in detection result with m bounding boxes

    @return D: n x m array with all distance

    """

    D = []

    number_1 = len(json_data_1)
    number_2 = len(json_data_2)

    for i in range(number_1):
        D_row = []
        for j in range(number_2):
            D_row.append(calculate_centroid_dist(
                [
                    json_data_1[i]['top'],
                    json_data_1[i]['bottom'],
                    json_data_1[i]['left'],
                    json_data_1[i]['right']
                ],
                [
                    json_data_2[j]['top'],
                    json_data_2[j]['bottom'],
                    json_data_2[j]['left'],
                    json_data_2[j]['right']
                ]
            ))
        D.append(D_row)
    return np.array(D)


def calculate_centroid_dist(bounding_box_1, bounding_box_2):
    """ Calculate centroid distance between two bounding boxes

    @param bounding_box_1: (top, bottom, left, right)
    @param bounding_box_2: (top, bottom, left, right)

    @return D: 2-norm centroid distance

    """
    bb1 = tinycv.Rect([
        bounding_box_1[0],
        bounding_box_1[1],
        bounding_box_1[2],
        bounding_box_1[3]])
    bb2 = tinycv.Rect([
        bounding_box_2[0],
        bounding_box_2[1],
        bounding_box_2[2],
        bounding_box_2[3]])

    centroid_1 = [int((bb1.r + bb1.l) / 2.0), int((bb1.b + bb1.t) / 2.0)]
    centroid_2 = [int((bb2.r + bb2.l) / 2.0), int((bb2.b + bb2.t) / 2.0)]
    square_dist_x = (centroid_1[0] - centroid_2[0])**2
    square_dist_y = (centroid_1[1] - centroid_2[1])**2

    D = (square_dist_x + square_dist_y) ** (1 / 2.0)

    return D


def calculate_IoU_all(json_data_1, json_data_2):
    """ Calculate IoU(intersection over union) between all bounding boxes
    in json_data_1 and json_data_2

    @param json_data_1: annotations in detection result with n bounding boxes
    @param json_data_2: annotations in detection result with m bounding boxes

    @return IoUs: n x m array with all IoUs

    """

    IoUs = []

    number_1 = len(json_data_1)
    number_2 = len(json_data_2)

    for i in range(number_1):
        IoU_row = []
        for j in range(number_2):
            IoU = calculate_IoU(
                [
                    json_data_1[i]['top'],
                    json_data_1[i]['bottom'],
                    json_data_1[i]['left'],
                    json_data_1[i]['right']
                ],
                [
                    json_data_2[j]['top'],
                    json_data_2[j]['bottom'],
                    json_data_2[j]['left'],
                    json_data_2[j]['right']
                ]
            )
            IoU_row.append(IoU)
        IoUs.append(IoU_row)
    return np.array(IoUs)


def calculate_IoU(bounding_box_1, bounding_box_2):
    """ Calculate IoU(intersection over union) between two bounding boxes

    @param bounding_box_1: (top, bottom, left, right)
    @param bounding_box_2: (top, bottom, left, right)

    @return IoU: interaction_area / union_area

    """
    top_1 = bounding_box_1[0]
    bottom_1 = bounding_box_1[1]
    left_1 = bounding_box_1[2]
    right_1 = bounding_box_1[3]
    top_2 = bounding_box_2[0]
    bottom_2 = bounding_box_2[1]
    left_2 = bounding_box_2[2]
    right_2 = bounding_box_2[3]
    area_1 = (right_1 - left_1 + 1) * (bottom_1 - top_1 + 1)
    area_2 = (right_2 - left_2 + 1) * (bottom_2 - top_2 + 1)

    max_1 = max(min(right_1, right_2) - max(left_1, left_2) + 1, 0)
    max_2 = max(min(bottom_1, bottom_2) - max(top_1, top_2) + 1, 0)
    interaction_area = max_1 * max_2
    iou = interaction_area / float(area_1 + area_2 - interaction_area)

    return iou


def calculate_overlap_ratio(
        bounding_box_1,
        bounding_box_2,
        denominator_type='union_area'):
    """ Calculate overlap ratio between two bounding boxes

    @param bounding_box_1: (top, bottom, left, right)
    @param bounding_box_2: (top, bottom, left, right)
    @param denominator_type: 'union_area', 'area_1' or 'area_2'

    @return overlap_ratio: interaction_area / denominator

    """

    top_1 = bounding_box_1[0]
    bottom_1 = bounding_box_1[1]
    left_1 = bounding_box_1[2]
    right_1 = bounding_box_1[3]
    top_2 = bounding_box_2[0]
    bottom_2 = bounding_box_2[1]
    left_2 = bounding_box_2[2]
    right_2 = bounding_box_2[3]

    max_i_1 = max(min(right_1, right_2) - max(left_1, left_2), 0)
    max_i_2 = max(min(bottom_1, bottom_2) - max(top_1, top_2), 0)
    interaction_area = max_i_1 * max_i_2

    denominator = 0.0
    if denominator_type == 'union_area':
        max_u_1 = max(max(right_1, right_2) - min(left_1, left_2), 0)
        max_u_2 = max(max(bottom_1, bottom_2) - min(top_1, top_2), 0)
        union_area = max_u_1 * max_u_2
        denominator = float(union_area)
    elif denominator_type == 'area_1':
        area_1 = max(right_1 - left_1, 0) * max(bottom_1 - top_1, 0)
        denominator = float(area_1)
    elif denominator_type == 'area_2':
        area_2 = max(right_2 - left_2, 0) * max(bottom_2 - top_2, 0)
        denominator = float(area_2)
    if denominator == 0:
        return 0
    else:
        over_lap_ratop = float(interaction_area) / float(denominator)
        return over_lap_ratop


def calculate_overlap_ratio_all(
        json_data_1,
        json_data_2,
        denominator_type='union_area'):
    """ Calculate overlap ratio between all bounding boxes
    in json_data_1 and json_data_2

    @param json_data_1: annotations in detection result with n bounding boxes
    @param json_data_2: annotations in detection result with m bounding boxes
    @param denominator_type: 'union_area', 'area_1' or 'area_2'

    @return overlap_ratio_all: n x m array with all overlap ratios

    """

    overlap_ratio_all = []

    number_1 = len(json_data_1)
    number_2 = len(json_data_2)

    for i in range(number_1):
        overlap_ratio_row = []
        for j in range(number_2):
            overlap_ratio = calculate_overlap_ratio(
                [
                    json_data_1[i]['top'],
                    json_data_1[i]['bottom'],
                    json_data_1[i]['left'],
                    json_data_1[i]['right']
                ],
                [
                    json_data_2[j]['top'],
                    json_data_2[j]['bottom'],
                    json_data_2[j]['left'],
                    json_data_2[j]['right']
                ], denominator_type)
            overlap_ratio_row.append(overlap_ratio)
        overlap_ratio_all.append(overlap_ratio_row)
    return np.array(overlap_ratio_all)


def is_lab_format(result_to_check, verbose=False, loose=False):
    return if_result_match_lab_format(
            result_to_check, verbose=verbose, loose=loose)


def if_result_match_lab_format(result_to_check, verbose=False, loose=False):
    """ Check if the result match lab format """

    if not isinstance(result_to_check, dict):
        if verbose:
            print("[dt42lab] ERROR: result is not a dictionary.")
        return False
    if loose:
        keys = ["annotations", "size"]
    else:
        keys = ["annotations", "size", "filename", "folder"]
    for key in keys:
        if key not in result_to_check.keys():
            if verbose:
                print(
                    "[dt42lab] ERROR: %s is not in the result checked." % key
                )
            return False
    return True


def conv_lab_anno_to_rect(lab_annotation):
    """ Convert lab annotation to Rect object """

    rect = tinycv.conv_bb_rect(
        [lab_annotation["top"], lab_annotation["bottom"],
         lab_annotation["left"], lab_annotation["right"]]
    )
    return rect


def match_by_overlap_ratio(
        detection_result_1,
        detection_result_2,
        overlap_ratio_th=0):
    """ One to one bounding boxes matching according to overlap ratio

    @param detection_result_1: annotations in detection result
    @param detection_result_2: annotations in detection result
    @param overlap_ratio_th: only overlap ratio > overlap_ratio_th matched

    @return match_result: {
        'match_index_1': list of bounding box index in detection_result_1
        'match_index_2': list of bounding box index in detection_result_2
        'overlap_ratio': list of overlap_ratio
    }

    """

    match_result = {
        'match_index_1': [],
        'match_index_2': [],
        'overlap_ratio': []
    }

    overlap_ratio_all = []
    number_1 = len(detection_result_1)

    number_2 = len(detection_result_2)
    if number_1 == 0 or number_2 == 0:
        return match_result

    overlap_ratio_all = calculate_overlap_ratio_all(
        detection_result_1, detection_result_2)

    # one to one match
    max_value = overlap_ratio_all.max()
    while max_value > overlap_ratio_th:
        max_index = overlap_ratio_all.argmax()
        index_1 = int(max_index / number_2)
        index_2 = int(max_index % number_2)
        match_result['match_index_1'].append(index_1)
        match_result['match_index_2'].append(index_2)
        match_result['overlap_ratio'].append(max_value)
        overlap_ratio_all[index_1, :] = 0
        overlap_ratio_all[:, index_2] = 0
        max_value = overlap_ratio_all.max()

    return match_result


def if_valid_anno_exist(lab_res):
    """ check if a valid annotation exist """

    # annotations should exist
    if "annotations" not in lab_res.keys():
        return False
    # annotations should be a list
    if not isinstance(lab_res["annotations"], list):
        return False
    # should contain at least one valid anno
    if len(lab_res["annotations"]) < 1:
        return False
    if not isinstance(lab_res["annotations"][0], dict):
        return False
    else:
        if "label" not in lab_res["annotations"][0].keys():
            return False
    return True


def split_detection(detection, cross_channel=True):
    """split bounding_box of a 4-channel-merged image.

     ------ ------  <--
    | img0 | img2 |   | height
     ------ ------  <--
    | img1 | img3 |
     ------ ------
    ^      ^
    |______|
      width

    :param detection: lab format detection results
    :param cross_channel: true to output detection results on merged image
                          false to output detection results on separated images
    """
    width = int(detection["size"]["width"] / 2)
    height = int(detection["size"]["height"] / 2)
    anno = detection["annotations"]

    # split vertically
    split_anno = []
    for i in range(len(anno)):
        bounding_box = [anno[i]["top"], anno[i]["bottom"],
                        anno[i]["left"], anno[i]["right"]]
        if bounding_box[0] >= height:
            split_anno.append(copy.deepcopy(anno[i]))
            split_anno[-1]["labinfo"]["channel_index"] = 1
            if not cross_channel:
                split_anno[-1]["top"] -= height
                split_anno[-1]["bottom"] -= height
        elif bounding_box[1] >= height:
            if bounding_box[1] - height < height - bounding_box[0]:
                split_anno.append(copy.deepcopy(anno[i]))
                split_anno[-1]["labinfo"]["channel_index"] = 0
                split_anno[-1]["bottom"] = height - 1
            else:
                split_anno.append(copy.deepcopy(anno[i]))
                split_anno[-1]["labinfo"]["channel_index"] = 1
                if cross_channel:
                    split_anno[-1]["top"] = height
                else:
                    split_anno[-1]["top"] = 0
                    split_anno[-1]["bottom"] -= height
        else:
            split_anno.append(copy.deepcopy(anno[i]))
            split_anno[-1]["labinfo"]["channel_index"] = 0

    # split horizontally
    anno = copy.deepcopy(split_anno)
    split_anno = []
    for i in range(len(anno)):
        bounding_box = [anno[i]["top"], anno[i]["bottom"],
                        anno[i]["left"], anno[i]["right"]]
        if bounding_box[2] >= width:
            split_anno.append(copy.deepcopy(anno[i]))
            split_anno[-1]["labinfo"]["channel_index"] += 2
            if not cross_channel:
                split_anno[-1]["left"] -= width
                split_anno[-1]["right"] -= width
        elif bounding_box[3] >= width:
            if bounding_box[3] - width < width - bounding_box[2]:
                split_anno.append(copy.deepcopy(anno[i]))
                split_anno[-1]["right"] = width - 1
            else:
                split_anno.append(copy.deepcopy(anno[i]))
                split_anno[-1]["labinfo"]["channel_index"] += 2
                if cross_channel:
                    split_anno[-1]["left"] = width
                else:
                    split_anno[-1]["left"] = 0
                    split_anno[-1]["right"] -= width
        else:
            split_anno.append(copy.deepcopy(anno[i]))

    # split annotations
    if cross_channel:
        results = copy.deepcopy(detection)
        results["annotations"] = split_anno
    else:
        results = []
        for i in range(4):
            results.append(copy.deepcopy(detection))
            results[-1]["annotations"] = []
            results[-1]["size"]["width"] = width
            results[-1]["size"]["height"] = height
        for res in split_anno:
            results[res["labinfo"]["channel_index"]]["annotations"].append(
                copy.deepcopy(res))
    return results


def img_comparator(tar_img, ref_img, dstack=True):
    """ check if two images exactly the same.

    @param dstack: true to auto turn one-channel gray image to three-channels
        gray image by stacking.
    @return diff_sum: sum of pixel-wise l1-norm difference between tar_img
        and ref_img.

    """

    if dstack is True and image.is_rgb(tar_img) is False:
        tar_img = np.dstack((tar_img, tar_img, tar_img))

    diff = tinycv.l1_norm_diff_cv2(ref_img, tar_img)
    diff_sum = sum(sum(diff))

    return diff_sum
