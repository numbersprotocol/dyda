'''
Updated 2018/01/18 by Tammy Yang

Functions of this module has been merged into dyda_utils.tools.
The file is kept for preserving the compatibility.
There should not be any new functions added into this file.
'''

import os
import sys
import json
import math
import csv
import numpy as np
from operator import mul
from dyda_utils import tools
from dyda_utils import image


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.generic):
            return np.asscalar(obj)
        return json.JSONEncoder.default(self, obj)


def rebin_2d(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape,
        by summing or averaging.

    Number of output dimensions must match number of input dimensions.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray


def cal_vector_length(array):
    """Calculate the length of an input array"""

    array = conv_to_np(array)
    mean = np.square(array).mean()
    return math.sqrt(mean)


def cal_standard_error(array):
    """Calculate standard error"""

    array = conv_to_np(array)
    return np.std(array)/math.sqrt(len(array))


def check_len(a, b):
    """Check if two arrays have the same length"""

    la = len(a)
    lb = len(b)
    if la == lb:
        return la
    print("[TOOLS] ERROR: length of a (%i) and b (%i) are different"
          % (la, lb))
    sys.exit(1)


def conv_to_np(array):
    """Convert list to np.ndarray"""

    if type(array) is list:
        return np.array(array)

    if is_np(array):
        return array

    print("[TOOLS] WARNING: the type of input array is not correct!")
    print(type(array))
    return array


def get_perc(data):
    """Convert the input data to percentage of the total sum

    @param data: input data array (1D)

    @return data in percentage-wise

    """

    data = conv_to_np(data)
    data = data.astype(float)
    return (data/np.sum(data))*100


def is_np(array):
    """Check if the input array is in type of np.ndarray"""

    if type(array) in [np.ndarray, np.int64, np.float64]:
        return True
    return False


def area(size):
    return reduce(mul, size)


def txt_to_list(file_path):
    """Read text file as list
       This is for meeting compatibility to auto_labeler of trainer
    """
    return read_txt(file_path)


def read_txt(file_path):
    """Read text file as list

    @param file_path: file to read

    @return list of text content

    """

    out_list = []
    with open(file_path, "r") as f:
        for row in f:
            out_list.append(row.strip())
    return out_list


def read_csv(fname, ftype=None):
    """Read CSV file as list

    @param fname: file to read

    Keyword arguments:
    ftype  -- convert data to the type (default: None)

    @return list of csv content

    """
    output = []
    with open(fname, 'rt') as csvfile:
        for row in csv.reader(csvfile, delimiter=','):
            if ftype is not None:
                row = map(ftype, row)
            output.append(list(row))
    return output


def write_csv(data, fname='output.csv'):
    """Write data to csv

    @param data: object to be written

    Keyword arguments:
    fname  -- output filename (default './output.csv')

    """
    f = open(fname, 'wb')
    wr = csv.writer(f, dialect='excel')
    wr.writerows(data)


def parse_json(fname, encoding=None):
    """Parse the input profile

    @param fname: input profile path

    @return data: a dictionary with user-defined data for training

    """
    if encoding is None:
        with open(fname) as data_file:
            data = json.load(data_file)
    else:
        with open(fname, encoding=encoding) as data_file:
            data = json.load(data_file)
    return data


def write_json(data, fname='./output.json'):
    """Write data to json

    @param data: object to be written

    Keyword arguments:
    fname  -- output filename (default './output.json')

    """
    with open(fname, 'w') as fp:
        json.dump(data, fp, cls=NumpyAwareJSONEncoder)


def read_csv_to_np(fname):
    """Read CSV file as numpy array

    Keyword arguments:
    fname  -- input filename

    @return numpy array

    """
    tools.check_exist(fname)
    content = read_csv(fname=fname, ftype=float)
    return conv_to_np(content)


def max_size(mat, value=0):
    """Find pos, h, w of the largest rectangle containing all `value`'s.
    For each row solve "Largest Rectangle in a Histrogram" problem [1]:
    [1]: http://blog.csdn.net/arbuckle/archive/2006/05/06/710988.aspx

    @param mat: input matrix

    Keyword arguments:
    value -- the value to be found in the rectangle

    @return (height, width), (start_y, start_x)
    """
    start_row = 0
    it = iter(mat)
    hist = [(el == value) for el in next(it, [])]
    max_size, start_pos = max_rectangle_size(hist)
    counter = 0
    for row in it:
        counter += 1
        hist = [(1+h) if el == value else 0 for h, el in zip(hist, row)]
        _max_size, _start = max_rectangle_size(hist)
        if area(_max_size) > area(max_size):
            max_size = _max_size
            start_pos = _start
            start_row = counter
    y = start_row - max_size[0] + 1
    if max_size[1] == len(hist):
        x = 0
    else:
        x = min(abs(start_pos - max_size[1] + 1), start_pos)
    return max_size, (y, x)


def max_rectangle_size(histogram):
    """Find height, width of the largest rectangle that fits entirely
    under the histogram. Algorithm is "Linear search using a stack of
    incomplete subproblems" [1].
    [1]: http://blog.csdn.net/arbuckle/archive/2006/05/06/710988.aspx
    """
    from collections import namedtuple
    Info = namedtuple('Info', 'start height')

    # Maintain a stack
    stack = []
    top = lambda: stack[-1]
    max_size = (0, 0)
    pos = 0
    for pos, height in enumerate(histogram):
        # Position where rectangle starts
        start = pos
        while True:
            # If the stack is empty, push
            if len(stack) == 0:
                stack.append(Info(start, height))
            # If the right bar is higher than the current top, push
            elif height > top().height:
                stack.append(Info(start, height))
            # Else, calculate the rectangle size
            elif stack and height < top().height:
                max_size = max(max_size, (top().height,
                               (pos - top().start)), key=area)
                start, _ = stack.pop()
                continue
            # Height == top().height goes here
            break

    pos += 1
    start_pos = 0
    for start, height in stack:
        _max_size = max(max_size, (height, (pos - start)), key=area)
        if area(_max_size) >= area(max_size):
            max_size = _max_size
            start_pos = start

    return max_size, start_pos


def _output_pred(input_path):
    """ Output prediction result based on dyda_utils spec https://goo.gl/So46Jw

    @param input_path: File path of the input

    """

    if not tools.check_exist(input_path):
        print('[dyda_utils] ERRPR: %s does not exist' % input_path)
        return

    input_file = os.path.basename(input_path)
    folder = os.path.dirname(input_path)
    input_size = image.get_img_info(input_path)[0]

    pred_info = {"filename": input_file, "folder": folder}
    pred_info["size"] = {"width": input_size[0], "height": input_size[1]}
    pred_info["sha256sum"] = tools.get_sha256(input_path)

    return pred_info


def output_pred_classification(input_path, conf, label, labinfo={}):
    """ Output classification result based on spec https://goo.gl/So46Jw

    @param input_path: File path of the input
    @param conf: Confidence score
    @param label: Label of the result

    Arguments:

    labinfo -- Additional results

    """

    pred_info = _output_pred(input_path)

    json_file = os.path.join(
        pred_info["folder"],
        pred_info["filename"].split('.')[0] + '.json'
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
    write_json(pred_info, fname=json_file)
