import os
import sys
import csv
import time
import json
import math
import shutil
import logging
import numpy as np
import datetime
from operator import mul
from functools import reduce


class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.generic):
            return np.asscalar(obj)
        return json.JSONEncoder.default(self, obj)


def is_empty_list(input_list, check_nest=True):
    """
        Return True only if the input nested list is a list and is empty.
        Check MR 80 for more details.
    """

    if isinstance(input_list, list):
        if len(input_list) == 0:
            return True
        else:
            if check_nest:
                # https://goo.gl/5I5NA8
                if isinstance(input_list[0], dict):
                    return not bool(input_list[0])
                else:
                    return all(map(is_empty_list, input_list))
            # if not check_nest, return False if len(input_list) > 0
            else:
                return False
    else:
        return False


def is_empty_dict(input_dict):
    """
        Return True only if the input_dict is a dict and is empty.
        Check MR 80 for more details.
    """

    if isinstance(input_dict, dict):
        return not bool(input_dict)
    else:
        return False


def check_cuda():
    """ Check the GPU and CUDA status """

    # FIXME: this might need to be improved in the future
    cuda_version_path = "/usr/local/cuda/version.txt"
    return check_exist(cuda_version_path)


def get_logger_level(level):
    """ Return logger level of logger """

    if isinstance(level, str):
        if level == 'debug' or level == 'vv':
            return logging.DEBUG
        elif level == 'info' or level == 'v':
            return logging.INFO
        else:
            return logging.WARNING
    elif isinstance(level, int):
        if level == 0:
            return logging.WARNING
        elif level == 1:
            return logging.INFO
        else:
            return logging.DEBUG
    else:
        return logging.WARNING


def replace_extension(file_path, new_ext):
    """ Replace extension of the given file_path

    @param file_path: input file path
    @param new_ext: extension to replace the original one

    @return file path with new extension

    """

    new_ext = new_ext if '.' in new_ext else '.' + new_ext
    return os.path.splitext(file_path)[0] + new_ext


def remove_extension(file_path, return_type='full'):
    """ Remove extension of the file_path

    @param file_path: input file path

    Keyword arguments:
    return_type -- the return type
                1. full (default): the fulle file path
                2. base-only: return only the basename

    @return file path with new extension

    """

    if return_type == 'full':
        name, ext = os.path.splitext(file_path)
        return name

    elif return_type == 'base-only':
        name, ext = os.path.splitext(os.path.basename(file_path))
        return name
    else:
        print('[dyda_utils] ERROR: wrong return_type for add_str_before_ext.')


def add_str_before_ext(file_path, str_to_add, return_type='full'):
    """ Add the given string to the file_path before the file extension

    @param file_path: input file path
    @param str_to_add: string to be added before extension of file_path

    Keyword arguments:
    return_type -- the return type
                1. full (default): the fulle file path
                2. base-only: return only the basename
                3. base-no-ext: return basename without extension

    @return new_name: file_path, basename or basename without extension

    """

    if return_type == 'full':
        name, ext = os.path.splitext(file_path)
        new_name = "{name}_{str_to_add}{ext}".format(
            name=name, str_to_add=str_to_add, ext=ext
        )
        return new_name
    elif return_type == 'base-only':
        name, ext = os.path.splitext(os.path.basename(file_path))
        new_name = "{name}_{str_to_add}{ext}".format(
            name=name, str_to_add=str_to_add, ext=ext
        )
        return new_name
    elif return_type == 'base-no-ext':
        name, ext = os.path.splitext(os.path.basename(file_path))
        new_name = "{name}_{str_to_add}".format(
            name=name, str_to_add=str_to_add
        )
        return new_name
    else:
        print('[dyda_utils] ERROR: wrong return_type for add_str_before_ext.')


def path_suffix(path, level=2):
    """Return the last parts of the path with a given level"""

    splits = path.split('/')
    suf = splits[-1]
    for i in range(2, level + 1):
        suf = os.path.join(splits[0 - i], suf)
    return suf


def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
    except OSError:
        pass


def find_folders(dir_path=None, keyword=None):
    """Find folders under a directory

    Keyword arguments:
    dir_path -- path of the directory to check (default: '.')
    keyword  -- keyword used to filter files (default: None)

    @return output: a list of folders found

    """
    if dir_path is None:
        dir_path = os.getcwd()
    dirs = [x[1] for x in os.walk(dir_path)][0]
    if keyword is not None:
        dirs = [d for d in dirs if d.find(keyword) >= 0]
    return dirs


def find_files(dir_path=None, keyword=None,
               suffix=('.json'), walkin=True, sort=False):
    """Find files under a directory

    Keyword arguments:
    dir_path -- path of the directory to check (default: '.')
    keyword  -- keyword used to filter files (default: None)
    suffix   -- file extensions to be selected (default: ('.json'))
    walkin   -- True to list recursively (default: True)
    sort     -- True to sort the list (default: False)

    @return output: a list of file paths found

    """
    if dir_path is None:
        dir_path = os.getcwd()

    output = []
    if walkin:
        for dirpath, dirnames, filenames in os.walk(dir_path):
            dirmatch = False
            if keyword is not None and dirpath.find(keyword) > 0:
                dirmatch = True
            if sort:
                filenames.sort()
            for f in filenames:
                # Update 2018/03/19 by Tammy
                # I have no idea why the logic was written in this wey
                # Since this is the core library, I will keep it as it is
                # for now and add an additional elif below
                # May need to revisit again in the future
                if keyword is not None and dirpath.find(keyword) < 0:
                    if not dirmatch:
                        continue
                # If keyword not found in dirpath and not in f
                elif keyword is not None and f.find(keyword) <= 0:
                    if not dirmatch:
                        continue
                if check_ext(f, suffix):
                    output.append(os.path.join(dirpath, f))
    else:
        for f in os.listdir(dir_path):
            # If keyword not found in dirpath and not in f
            if keyword is not None:
                if dir_path.find(keyword) <= 0 and f.find(keyword) <= 0:
                    continue
            if check_ext(f, suffix):
                output.append(os.path.join(dir_path, f))

    return output


def read_template(fname, temp_vars):
    """Read jinja template

    @param fname: Inpur file name
    @temp_vars: Variable dictionary to be used for the template

    @return: Rendered template

    """
    from jinja2 import FileSystemLoader, Environment
    templateLoader = FileSystemLoader(searchpath="/")
    templateEnv = Environment(loader=templateLoader)
    try:
        template = templateEnv.get_template(fname)
        return template.render(temp_vars)
    except BaseException:
        print("[dyda_utils] ERROR Exception:", sys.exc_info()[0])
        raise


def get_sha256(filepath):
    """Generate sha256 hash for the file"""

    import hashlib
    bufsize = 65536
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(bufsize)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def gen_md5(data):
    """Generate md5 hash for a data structure"""

    import hashlib
    import pickle
    _data = pickle.dumps(data)
    return hashlib.md5(_data).hexdigest()


def print_time(t0, s):
    """Print how much time has been spent

    @param t0: previous timestamp
    @param s: description of this step

    """
    print("[dyda_utils] INFO: %.5f seconds to %s" % ((time.time() - t0), s))
    return time.time()


def ptime(t0=None, s="execute"):
    """Set default for print_time

    Keyword arguments:
    t0 -- previous timestamp
    s  -- description of this step

    """
    if t0 is None:
        print("[dyda_utils] WARNING: Initial time is not set.")
        t0 = time.time()
    return print_time(t0, s)


def create_timestamp(datetime_obj=None):
    """Create lab-format timestampe"""

    if datetime_obj is None:
        datetime_obj = datetime.datetime.now()
    try:
        timestamp = datetime_obj.strftime("%Y%m%d%H%M%S%f")
        return timestamp

    except AttributeError:
        print("[dyda_utils] ERROR: Fail to get strftime of the input object")
        return None

    except:
        print("[dyda_utils] ERROR: Error occor in dyda_utils.create_timestamp")
        return None


def conv_to_date(raw_data, key):
    """Convert y,m,d assigned in json profile to date object"""

    from datetime import date
    return date(raw_data[key]["y"],
                raw_data[key]["m"], raw_data[key]["d"])


def get_combinations(input_list, n=2):
    """Get all combination of elements of input list

    @param input_list: input array

    Keyword arguments:
    n -- size of a combination (default: 2)

    """
    from itertools import combinations_with_replacement
    return combinations_with_replacement(input_list, n)


def check_exist(path, log=True):
    """Check if the path exists

    @param path: path to check

    """
    if os.path.exists(path):
        return True
    else:
        if log:
            print("[dyda_utils] WARNING: %s does not exist" % path)
        return False


def check_ext(file_name, extensions):
    """Check the file extension

    @param file_name: input file name
    @param extensions: string or list, extension(s) to check

    @return bool: True if it is matched

    """
    if file_name.endswith(extensions):
        return True
    return False


def check_yes(answer):
    """Check if the answer is yes"""
    if answer.lower() in ['y', 'yes']:
        return True
    return False


def dir_check(dirpath):
    check_dir(dirpath)


def check_dir(dirpath):
    """Check if a directory exists.
       create it if doean't"""

    if not os.path.exists(dirpath):
        print("[dyda_utils] INFO: Creating %s" % dirpath)
        os.makedirs(dirpath)


def check_parent(fpath):
    """Check if the parent directory exists.
       create it if doean't"""

    dirname = os.path.dirname(fpath)
    check_dir(dirname)


def move_file(fpath, newhome, ask=True):
    """Move a file

    @param fpath: the original path of the file
    @param newhome: new home (directory) of the file

    Keyword arguments:
    ask -- true to ask before moving (default: True)

    """
    dirname = os.path.dirname(fpath)
    check_dir(newhome)
    newpath = fpath.replace(dirname, newhome)
    if ask:
        show_str = "Do you want to move %s to %s?" % (fpath, newhome)
        yn = check_yes(raw_input(show_str))
    else:
        yn = True
    if yn:
        shutil.move(fpath, newpath)


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
    compression_pairs = [(d, c // d) for d, c in zip(new_shape,
                                                     ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1 * (i + 1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1 * (i + 1))
    return ndarray


def cal_vector_length(array):
    """Calculate the length of an input array"""

    array = conv_to_np(array)
    mean = np.square(array).mean()
    return math.sqrt(mean)


def cal_standard_error(array):
    """Calculate standard error"""

    array = conv_to_np(array)
    return np.std(array) / math.sqrt(len(array))


def check_len(a, b):
    """Check if two arrays have the same length"""

    la = len(a)
    lb = len(b)
    if la == lb:
        return la
    print("[dyda_utils] ERROR: length of a (%i) and b (%i) are different"
          % (la, lb))
    sys.exit(1)


def conv_to_np(array):
    """Convert list to np.ndarray"""

    if isinstance(array, list):
        return np.array(array)

    if is_np(array):
        return array

    print("[dyda_utils] WARNING: the type of input array is not correct!")
    print(type(array))
    return array


def get_perc(data):
    """Convert the input data to percentage of the total sum

    @param data: input data array (1D)

    @return data in percentage-wise

    """

    data = conv_to_np(data)
    data = data.astype(float)
    return (data / np.sum(data)) * 100


def is_np(array):
    """Check if the input array is in type of np.ndarray"""

    if type(array) in [np.ndarray, np.int64, np.float64]:
        return True
    return False


def area(size):
    return reduce(mul, size)


def list_to_txt(inlist, fpath="./list.txt"):
    """Write list to txt file"""

    with open(fpath, 'w') as f:
        for line in inlist:
            f.write("%s\n" % line)


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


def del_key_from_dic(input_dic, key):
    """Delete a key from the input dictionary"""
    try:
        del input_dic[key]
    except KeyError:
        print('[dyda_utils] %s is not found in the input dictionary,'
              ' do nothing.' % key)
        pass


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


def dump_json(data):
    """Dump dictionary data as json format

    @param data: object to be dumpped

    """
    return json.dumps(data, cls=NumpyAwareJSONEncoder)


def read_csv_to_np(fname):
    """Read CSV file as numpy array

    Keyword arguments:
    fname  -- input filename

    @return numpy array

    """
    check_exist(fname)
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
    _max_size, start_pos = max_rectangle_size(hist)
    counter = 0
    for row in it:
        counter += 1
        hist = [(1 + h) if el == value else 0 for h, el in zip(hist, row)]
        _max_size_rec, _start = max_rectangle_size(hist)
        if area(_max_size_rec) > area(_max_size):
            _max_size = _max_size_rec
            start_pos = _start
            start_row = counter
    y = start_row - _max_size[0] + 1
    if _max_size[1] == len(hist):
        x = 0
    else:
        x = min(abs(start_pos - _max_size[1] + 1), start_pos)
    return _max_size, (y, x)


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

    def top(): return stack[-1]
    _max_size = (0, 0)
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
                _max_size = max(_max_size, (top().height,
                                            (pos - top().start)), key=area)
                start, _ = stack.pop()
                continue
            # Height == top().height goes here
            break

    pos += 1
    start_pos = 0
    for start, height in stack:
        _max_size_stack = max(_max_size, (height, (pos - start)), key=area)
        if area(_max_size_stack) >= area(_max_size):
            _max_size = _max_size_stack
            start_pos = start

    return _max_size, start_pos


def voc_xml_to_dict(filename, force_label=None):
    """Read in data from a xml file and turn it to lab-format dict.

    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(filename)
    root = tree.getroot()

    try:
        folder = root.find('folder').text
    except BaseException:
        folder = None

    try:
        filename = root.find('filename').text
    except BaseException:
        filename = None

    try:
        for obj in root.iter('size'):
            width = int(obj.find('width').text)
            height = int(obj.find('height').text)
    except BaseException:
        width = None
        height = None

    json_data = {
        'folder': folder,
        'filename': filename,
        'annotations': [],
        'size': {'width': width, 'height': height}}

    anno = []
    for obj in root.iter('object'):
        if force_label is None:
            cls = obj.find('name').text
        else:
            cls = force_label
        bb = obj.find('bndbox')
        xmin = bb.find('xmin').text
        xmax = bb.find('xmax').text
        ymin = bb.find('ymin').text
        ymax = bb.find('ymax').text
        if cls == "people":
            cls = "person"

        json_data['annotations'].append({
            'label': cls,
            'type': 'ground_truth',
            'top': int(ymin),
            'bottom': int(ymax),
            'left': int(xmin),
            'right': int(xmax),
            'confidence': 1.0
        })

    return json_data


def delete_file(fpath):
    """ check if the path exist, then delete it """

    if os.path.exists(fpath):
        os.remove(fpath)
    else:
        print("[dyda_utils] The file %s does not exist" % fpath)


def gen_numerical_str(number):
    """ Generate numerical string with six digits """

    new_number = "%06d" % number
    return str(new_number)


def _record_time():
    """ generator that stores the time when call record_time"""
    t = time.time()
    yield
    while True:
        now = time.time()
        yield now - t
        t = now


def record_time(action=None, gen=_record_time()):
    """ Call this function two times, and it would count the time passing
        between the first call and second call.
        Example:
                record_time()
                code
                record_time() ==> output the time of code.

        If we specified two recorders(or more), we can let them record
        different time passing.
        Example:
               # action = 'start' means create recorder and count from here
               recorder1 = record_time(action='start')
               code1
               # gen = recorder1 means call recorder1 not others
               record_time(gen=recorder1) ==> output the time of code1

               # create another recorder
               recorder2 = record_time(action='start')
               code2
               record_time(gen=recorder2) ==> output the time of code2
               code3
               record_time(gen=recorder1) ==> output the time of code2 and 3
        """

    if action == 'start':
        gen = _record_time()
        next(gen)
        return gen
    else:
        return next(gen)
