"""tinycv - Computer vision utility functions
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
import numpy as np
from skimage import measure

from dyda_utils import image
from dyda_utils import data
from dyda_utils import tools
from dyda_utils import boxes
from dyda_utils import lab_tools


class Rect():

    def __init__(self, loc=None):

        self.t = 0
        self.b = 0
        self.l = 0
        self.r = 0
        self.w = 0
        self.h = 0
        if loc is not None:
            self.reset_loc(loc)

    def reset_loc(self, loc):

        self.t = loc[0]
        self.b = loc[1]
        self.l = loc[2]
        self.r = loc[3]
        self.w = loc[3] - loc[2]
        self.h = loc[1] - loc[0]


def rotate_ccw(_img, center=None, direction="ccw"):
    """
    Use numpy to rotate image instead of opencv
    Benchmark results see MR 88 of dt42-lab-lib
    """

    img = copy.deepcopy(_img)
    if image.is_rgb(img):
        if direction == 'ccw':
            img = np.transpose(img, (1, 0, 2))[::-1]
        elif direction == 'cw':
            img = np.transpose(img[::-1], (1, 0, 2))

    else:
        # case for gray image load by cv2.imread(img_path, 0)
        if direction == 'ccw':
            img = np.transpose(img, (1, 0))[::-1]
        elif direction == 'cw':
            img = np.transpose(img[::-1], (1, 0))
    return img


def rotate_ccw_opencv(_img, center=None, direction="ccw"):
    """
    Reference: https://goo.gl/GUQSHa
    Benchmark: < 2ms on gc1 for a 650x480 input
    center = [x, y] (in MATLAB format)
    """

    is_rgb = True
    if not image.is_rgb(_img):
        is_rgb = False
    if is_rgb:
        (rows, cols, ch) = _img.shape
    else:
        (rows, cols) = _img.shape

    size = max(rows, cols)
    img = image.auto_padding(_img)

    ycenter = int(size / 2)
    xcenter = int(size / 2)
    if direction == "ccw":
        angle = 90
    else:
        angle = 0 - 90
    M = cv2.getRotationMatrix2D((ycenter, xcenter), angle, 1)
    _dst = cv2.warpAffine(img, M, (size, size))

    min_length = min(rows, cols)
    # FIXME: https://gitlab.com/DT42/galaxy42/dt42-trainer/merge_requests/172
    # It is reported that the edge was shifted by one pixel.
    # However... the original math looks right... so I have no idea why...
    # Shift the space by one pixel can fix the issue temporarily
    # By shifting one pixel, if one rotates the image ccw->cw, then it is the
    # same as the original one.
    space = int((size - min_length) / 2) + 1

    if is_rgb:
        if cols > rows:
            dst = _dst[:, space:space + min_length, :]
        elif rows > cols:
            dst = _dst[space:space + min_length, :, :]

    else:
        if cols > rows:
            dst = _dst[:, space:space + min_length]
        elif rows > cols:
            dst = _dst[space:space + min_length, :]

    dst = image.resize_img(dst, (rows, cols))
    return dst


def rotate(img, angle, pivot='center', center=None):
    """
    Reference: https://goo.gl/GUQSHa
    Benchmark: < 2ms on gc1 for a 650x480 input
    center = [x, y] (in MATLAB format)
    """

    if image.is_rgb(img):
        (rows, cols, ch) = img.shape
    else:
        (rows, cols) = img.shape
    ycenter = 0
    xcenter = 0
    if center is not None and isinstance(center, list):
        (ycenter, xcenter) = center
    elif pivot == 'center':
        ycenter = cols / 2
        xcenter = rows / 2
    else:
        print('Using image center as pivot point')
        ycenter = cols / 2
        xcenter = rows / 2
    M = cv2.getRotationMatrix2D((ycenter, xcenter), angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def patch_rot_rec(img, angle, loc, pivot='center',
                  color=(0, 255, 0), line_width=6):
    """Patch a rotated rectangle to the image

    @param img: path of the image or the img tensor
    @param angle: rotation angle
    @param loc: position list, (top, bottom, left, right)

    Keyword parameters:
    pivot     : pivot point
    color     : color of the bounding box in turns of [B, G, R]
    line_width: width of the bounding box (default: 6)

    """

    if not isinstance(img, np.ndarray):
        tools.check_exist(img)
        img = image.read_img(img)

    (top, bottom, left, right) = loc
    img_rot = rotate(img, angle, pivot=pivot)
    patched = cv2.rectangle(img_rot, (left, top), (right, bottom),
                            color, line_width)
    img_rot_back = rotate(patched, 0 - angle, pivot=pivot)

    return img_rot_back


def rotate_and_patch(img, angle, loc, pivot='center',
                     color=(0, 255, 0), line_width=6):
    """Rotate the image and patch a rec on it

    @param img: path of the image or the img tensor
    @param angle: rotation angle
    @param loc: position list, (top, bottom, left, right)

    Keyword parameters:
    pivot     : pivot point
    color     : color of the bounding box in turns of [B, G, R]
    line_width: width of the bounding box (default: 6)

    """

    if not isinstance(img, np.ndarray):
        tools.check_exist(img)
        img = image.read_img(img)

    (top, bottom, left, right) = loc
    img_rot = rotate(img, angle, pivot=pivot)
    patched = cv2.rectangle(img_rot, (left, top), (right, bottom),
                            color, line_width)

    return patched


def scale_shift_param_polyfit(txt_path):
    """
    Get scale and shift parameters to transfer coordinate
    from image A to image B.

    @param txt_path: path of txt file in which each line contains
        corresponding points coordinate on source image A and
        destination image B in order of h_A, w_A, h_B, w_B
    @return scale_h, shift_h, scale_w, shift_w

    """
    data = [[] for i in range(4)]
    point_list = tools.txt_to_list(txt_path)
    for point in point_list:
        if not len(point.split(' ')) == 4:
            continue
        for idx in range(4):
            data[idx].append(int(point.split(' ')[idx]))
    scale_h, shift_h = np.polyfit(data[0], data[2], 1)
    scale_w, shift_w = np.polyfit(data[1], data[3], 1)

    return (scale_h, shift_h, scale_w, shift_w)


def scale_shift_param(P1, Q1, P2, Q2):
    """
    Get scale and shift parameters to transfer coordinate
    from image A to image B.
    h_q1 = scale_h * h_p1 + shift_h
    w_q1 = scale_w * w_p1 + shift_w
    h_q2 = scale_h * h_p2 + shift_h
    w_q2 = scale_w * w_p2 + shift_w

    @param P1: (h_p1, w_p1) coordinate of point P1 on image A
    @param Q1: (h_q1, w_q1) coordinate of point Q1 on image B
        corresponding to P1
    @param P2: (h_p2, w_p2) coordinate of point P2 on image A
    @param Q2: (h_q2, w_q2) coordinate of point Q2 on image B
        corresponding to P2

    @return scale_h, shift_h, scale_w, shift_w

    """
    (h_p1, w_p1) = P1
    (h_q1, w_q1) = Q1
    (h_p2, w_p2) = P2
    (h_q2, w_q2) = Q2

    a = np.array([[h_p1, 1], [h_p2, 1]])
    b = np.array([h_q1, h_q2])
    scale_h, shift_h = np.linalg.solve(a, b)

    a = np.array([[w_p1, 1], [w_p2, 1]])
    b = np.array([w_q1, w_q2])
    scale_w, shift_w = np.linalg.solve(a, b)

    return (scale_h, shift_h, scale_w, shift_w)


def image_calibration(imageA, imageB, ratio=0.75, reprojThresh=4.0):
    """
    Calibrate imageA and imageB by warping imageA to align imageB.

    @param imageA: image to be warped
    @param imageB: base image
    @param ratio, reprojThresh: parameters for feature matching
    @return resultA, resultB: images after calibration

    """

    heightA, widthA, channelA = imageA.shape
    heightB, widthB, channelB = imageB.shape

    # detect keypoints and extract local invariant descriptors from them
    (kpsA, featuresA) = sift_extraction(imageA)
    (kpsB, featuresB) = sift_extraction(imageB)

    # match features between the two images
    M = feature_matching(kpsA, kpsB,
                         featuresA, featuresB, ratio, reprojThresh)

    labelA = imageA
    labelA = np.ones((heightA, widthA, channelA))

    # if the match is None, then there aren't enough matched
    # keypoints to create a panorama
    if M is None:
        resultA = imageA
        labelA = np.ones((heightA, widthA, channelA))
    # otherwise, apply a perspective warp to stitch the images
    # together
    else:
        (matches, H, status) = M
        resultA = cv2.warpPerspective(imageA, H,
                                      (widthA + widthB, heightA))
        resultA = resultA[0:heightB, 0:widthB, :]
        labelA = np.ones((heightA, widthA, channelA))
        labelA = cv2.warpPerspective(labelA, H,
                                     (widthA + widthB, heightA))
        labelA = labelA[0:heightA, 0:widthA, :]
    resultA = np.where(labelA == 1, resultA, 0)
    resultB = imageB
    resultB = np.where(labelA == 1, resultB, 0)
    return (resultA, resultB)


def sift_extraction(img):
    """
    Detect keypoints and extract sift features from img.

    @param img: image to be extracted
    @return kps: keypoints
    @return features: sift features

    """

    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect and extract features from the image
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(img, None)

    # convert the keypoints from KeyPoint objects to NumPy
    # arrays
    kps = np.float32([kp.pt for kp in kps])

    # return a tuple of keypoints and features
    return (kps, features)


def feature_matching(kpsA, kpsB, featuresA, featuresB,
                     ratio, reprojThresh):
    """
    Match keypoints by features.

    @param kpsA, kpsB: keypoints to be matched
    @param featuresA, featuresB: features of keypoints
    @param ratio: keypoints matched if minimum distance <
        the second minimum distance * ratio
    @param reprojThresh: parameter of finding homography
    @return matches: list of matched index
    @return H: homograpy matrix
    @return status: status of each matched point

    """

    # compute the raw matches and initialize the list of actual
    # matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []

    # loop over the raw matches
    for m in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # computing a homography requires at least 4 matches
    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                         reprojThresh)

        # return the matches along with the homograpy matrix
        # and status of each matched point
        return (matches, H, status)

    # otherwise, no homograpy could be computed
    return None


def foreground_extraction_by_ccl(
        img,
        img_bg,
        calibration=1,
        re_width=400,
        diff_thre=15,
        pixel_num_min=500,
        kernel_size=3,
        iter_num=2,
        seg_margin=10):
    """
    Extract bounding box of foreground by connected components labeling(ccl).
    Only one bounding box with most different pixel number is output.
    If the different pixel number is less than pixel_num_min, [] is output.

    @param img: image to be extracted
    @param img_bg: background image
    @param calibration: do calibration before calculate difference if true
    @param diff_thre: pixelwise difference threshold
    @param pixel_num_min: minimum different pixel number
    @param kernel_size: kernel size for morphological opening
    @param iter_num: iteration number of erosion and dilation
    @param seg_margin: the margin added to bounding box
    @return bounding_box: bounding box of foreground

    """

    # resize image
    height, width, channels = img.shape
    re_height = int(re_width * height / width)
    img = cv2.resize(img, (re_width, re_height))
    img_bg = cv2.resize(img_bg, (re_width, re_height))
    ratio = width / re_width

    # calibration
    if calibration == 1:
        [img_bg, img] = image_calibration(img_bg, img)

    # l1 norm difference
    img_diff = l1_norm_diff_cv2(img, img_bg)

    # binarization
    img_diff_bn = np.ones((re_height, re_width), np.bool_)
    img_diff_bn = np.where(img_diff < diff_thre, 0, 1)
    print('max img diff', img_diff.max())

    # morphological opening
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_diff_bn = img_diff_bn.astype(np.uint8)
    img_diff_bn = cv2.erode(
        img_diff_bn,
        kernel,
        iter_num)
    img_diff_bn = cv2.dilate(
        img_diff_bn,
        kernel,
        iter_num)

    # connected components labeling
    cc_label = measure.label(img_diff_bn, background=0)
    cc_label = cc_label.astype(np.uint8)

    # calculate bounding box
    bounding_box = []
    max_num = 0
    label_idx = 1
    pixel_num = 1
    while pixel_num > 0:
        label = np.zeros((re_height, re_width), np.bool_)
        label = np.where(cc_label == label_idx, 1, 0)
        column_sum = label.sum(0)
        column_idx = np.where(column_sum > 0)
        row_sum = label.sum(1)
        row_idx = np.where(row_sum > 0)
        pixel_num = sum(sum(label))
        label_idx = label_idx + 1
        print('pixel_num:', pixel_num)
        if pixel_num > pixel_num_min and pixel_num > max_num:
            max_num = pixel_num
            top = max(0, row_idx[0][0] * ratio - seg_margin)
            bottom = min(height - 1, row_idx[0][-1] * ratio + seg_margin)
            left = max(0, column_idx[0][0] * ratio - seg_margin)
            right = min(width - 1, column_idx[0][-1] * ratio + seg_margin)
            bounding_box.append([
                int(top),
                int(bottom),
                int(left),
                int(right)])

    return bounding_box


def l1_norm_diff_cv2(
        imageA,
        imageB):
    """
    Calculate L1 norm difference between color vectors
    of each pixel in two images using opencv for absdiff.

    @param imageA, imageB: images to be calculated difference
    @return diff: l1 norm difference

    """
    image_sub = cv2.absdiff(imageB, imageA)
    if len(image_sub.shape) == 3:
        diff = image_sub.sum(axis=2)
    elif len(image_sub.shape) == 2:
        diff = image_sub
    return(diff)


def l1_norm_diff(
        imageA,
        imageB):
    """
    Calculate L1 norm difference between color vectors
    of each pixel in two images.

    @param imageA, imageB: images to be calculated difference
    @return diff: l1 norm difference

    """

    imageA = imageA.astype(float)
    imageB = imageB.astype(float)
    image_sub = np.subtract(imageB, imageA)
    image_sub = np.abs(image_sub)
    diff = image_sub.sum(axis=2)
    return(diff)


def resize_bounding_box_in_json(in_json, out_json, resize_ratio):
    """
    Resize the bounding box in a json file from a detector.
    The resize_ratio is (length in out_json) / (length in in_json).
    """

    with open(in_json, 'r') as rf:
        json_data = json.load(rf)
    for bi in range(len(json_data)):
        json_data[bi]['topleft']['x'] = int(
            float(json_data[bi]['topleft']['x']) * resize_ratio)
        json_data[bi]['topleft']['y'] = int(
            float(json_data[bi]['topleft']['y']) * resize_ratio)
        json_data[bi]['bottomright']['x'] = int(
            float(json_data[bi]['bottomright']['x']) * resize_ratio)
        json_data[bi]['bottomright']['y'] = int(
            float(json_data[bi]['bottomright']['y']) * resize_ratio)
    with open(out_json, 'w') as wf:
        json.dump(json_data, wf)


def img_radia_transform_return_info(img, seed, precision=2):
    """ Same RT transform as img_radia_transform but returns U, V, m, n """

    pi = round(math.pi, precision)
    (m, n) = seed
    shape = img.shape
    U = shape[1]
    V = shape[0]
    lens = len(shape)
    new_img = np.zeros(shape)
    for u in range(0, U):
        theta = 2 * pi * u / U
        for v in range(0, V):
            x = (math.floor(v * math.cos(theta)))
            y = (math.floor(v * math.sin(theta)))
            new_y = (int)(x + n)
            new_x = (int)(y + m)

            try:
                if lens == 3:
                    for i in range(0, 3):
                        if new_x < 0 or new_y < 0 or new_x >= U or new_y >= V:
                            new_img[v, u, i] = 128
                        else:
                            new_img[v, u, i] = img[new_y, new_x, i]
                else:
                    if new_x < 0 or new_y < 0 or new_x >= U or new_y >= V:
                        new_img[v, u] = 128
                    else:
                        new_img[v, u] = img[new_y, new_x]

            except IndexError:
                print('(v, u) = (%i, %i) out of boundary' % (u, v))

    return new_img.astype(np.uint8), (U, V, m, n)


def img_radia_transform(img, seed, precision=2):
    """
    Radia Transform of the image. Details see arXiv:1708.04347

    @param img: read image, not filename
    @seed     : center origin of the transformation

    Keyword arguments:
    precision  -- precision of the math.pi (default: 2)

    @return transformed image

    """

    pi = round(math.pi, precision)
    (m, n) = seed
    shape = img.shape
    U = shape[1]
    V = shape[0]
    lens = len(shape)
    new_img = np.zeros(shape)
    for u in range(0, U):
        theta = 2 * pi * u / U
        for v in range(0, V):
            x = (math.floor(v * math.cos(theta)))
            y = (math.floor(v * math.sin(theta)))
            new_y = (int)(x + n)
            new_x = (int)(y + m)

            try:
                if lens == 3:
                    for i in range(0, 3):
                        if new_x < 0 or new_y < 0 or new_x >= U or new_y >= V:
                            new_img[v, u, i] = 128
                        else:
                            new_img[v, u, i] = img[new_y, new_x, i]
                else:
                    if new_x < 0 or new_y < 0 or new_x >= U or new_y >= V:
                        new_img[v, u] = 128
                    else:
                        new_img[v, u] = img[new_y, new_x]

            except IndexError:
                print('(v, u) = (%i, %i) out of boundary' % (u, v))

    return new_img.astype(np.uint8)


def image_sharpen(img, C=0.5):
    """
    C is multiplicative coefficient and (0.3,1.5) is the reasonable range.
    The larger C is, the stronger sharpen effect will be.
    """

    img = img.astype(float)
    new_image = copy.deepcopy(img)

    for i in range(3):
        [gradY, gradX] = np.gradient(img[:, :, i])
        [sqgradXY, sqgradXX] = np.gradient(gradX)
        gradY_t = map(list, zip(*gradY))
        [sqgradYY_t, sqgradYX_t] = np.gradient(gradY_t)
        sqgradYX = map(list, zip(*sqgradYX_t))
        Laplacian = sqgradXX + sqgradYX
        new_image[:, :, i] = img[:, :, i] - C * Laplacian

    max_intensity = np.iinfo(np.uint8).max
    min_intensity = np.iinfo(np.uint8).min
    new_image = np.where(
        new_image > max_intensity,
        max_intensity,
        new_image)
    new_image = np.where(new_image < min_intensity,
                         min_intensity, new_image)

    new_image = np.array(new_image, dtype=np.uint8)

    return(new_image)


def image_brighten(img, phi=1, theta=1):

    img = img.astype(float)
    max_intensity = np.iinfo(np.uint8).max
    new_image = (max_intensity / phi) * (img / (max_intensity / theta))**0.5
    new_image = np.array(new_image, dtype=np.uint8)

    return(new_image)


def image_darken(img, phi=1, theta=1):

    img = img.astype(float)
    max_intensity = np.iinfo(np.uint8).max
    new_image = (max_intensity / phi) * (img / (max_intensity / theta))**2
    new_image = np.array(new_image, dtype=np.uint8)

    return(new_image)


def data_augmentation_detection(
        json_filename,
        output_folder,
        augmentation_type,
        save=True,
        out_suffix='jpg',
        prefix=''):
    """ Data augmentation for detection.

    @param json_filename: json file with detection result in
           lab_tools.output_pred_detection format
    @param output_folder: augmentation results will be saved in
           output_folder/image and output_folder/json
    @param augmentation_type: list of types including 'padding',
           'flip', 'flip_v', 'flip_h', 'blur', 'darken', 'brighten',
           'contrast'
    @param save: True to save output image and json
    @param out_suffix: suffix of output images
    @param prefix: prefix added to output images
    """

    output_img_folder = os.path.join(output_folder, 'image')
    tools.check_dir(output_img_folder)
    output_json_folder = os.path.join(output_folder, 'json')
    tools.check_dir(output_json_folder)

    data = tools.parse_json(json_filename)
    basename = os.path.basename(json_filename).split('.')[0]
    if not prefix == '':
        basename = prefix + '_' + basename
    image_filename = os.path.join(data['folder'],
                                  data['filename'])
    original_image = cv2.imread(image_filename)
    output_images = []
    output_jsons = []
    if 'padding' in augmentation_type:
        image_padded = image_padding(original_image)
        output_images.append(image_padded)
        ori_h, ori_w = original_image.shape[:2]
        s = max(ori_h, ori_w)
        h = int((s - ori_h) / 2)
        w = int((s - ori_w) / 2)
        output_jsons.append(
            lab_tools.shift_detection(
                copy.deepcopy(data), h, w, s, s))
    else:
        output_images.append(copy.deepcopy(original_image))
        output_jsons.append(copy.deepcopy(data))

    if 'flip' in augmentation_type or \
            ('flip_h' in augmentation_type and 'flip_v' in augmentation_type):
        _output_images = []
        _output_jsons = []
        for i, img in enumerate(output_images):
            for direction in ['h']:
                img_flipped = image.read_and_flip(
                    copy.deepcopy(img), direction)
                _output_images.append(img_flipped)
                _output_jsons.append(
                    lab_tools.flip_detection(
                        copy.deepcopy(
                            output_jsons[i]),
                        direction))
        output_images.extend(_output_images)
        output_jsons.extend(_output_jsons)
        _output_images = []
        _output_jsons = []
        for i, img in enumerate(output_images):
            for direction in ['v']:
                img_flipped = image.read_and_flip(
                    copy.deepcopy(img), direction)
                _output_images.append(img_flipped)
                _output_jsons.append(
                    lab_tools.flip_detection(
                        copy.deepcopy(
                            output_jsons[i]),
                        direction))
        output_images.extend(_output_images)
        output_jsons.extend(_output_jsons)
    elif 'flip_h' in augmentation_type:
        _output_images = []
        _output_jsons = []
        for i, img in enumerate(output_images):
            for direction in ['h']:
                img_flipped = image.read_and_flip(
                    copy.deepcopy(img), direction)
                _output_images.append(img_flipped)
                _output_jsons.append(
                    lab_tools.flip_detection(
                        copy.deepcopy(
                            output_jsons[i]),
                        direction))
        output_images.extend(_output_images)
        output_jsons.extend(_output_jsons)
    elif 'flip_v' in augmentation_type:
        _output_images = []
        _output_jsons = []
        for i, img in enumerate(output_images):
            for direction in ['v']:
                img_flipped = image.read_and_flip(
                    copy.deepcopy(img), direction)
                _output_images.append(img_flipped)
                _output_jsons.append(
                    lab_tools.flip_detection(
                        copy.deepcopy(
                            output_jsons[i]),
                        direction))
        output_images.extend(_output_images)
        output_jsons.extend(_output_jsons)

    if 'blur' in augmentation_type:
        kernel = np.ones((5, 5), np.float32) / 25
        _output_images = []
        _output_jsons = []
        for i, img in enumerate(output_images):
            image_new = cv2.filter2D(copy.deepcopy(img), -1, kernel)
            _output_images.append(image_new)
            _output_jsons.append(copy.deepcopy(output_jsons[i]))
        output_images.extend(_output_images)
        output_jsons.extend(_output_jsons)

    if 'brighten' in augmentation_type:
        _output_images = []
        _output_jsons = []
        for i, img in enumerate(output_images):
            image_new = image_brighten(copy.deepcopy(img))
            _output_images.append(image_new)
            _output_jsons.append(copy.deepcopy(output_jsons[i]))
        output_images.extend(_output_images)
        output_jsons.extend(_output_jsons)
    if 'darken' in augmentation_type:
        _output_images = []
        _output_jsons = []
        for i, img in enumerate(output_images):
            image_new = image_darken(copy.deepcopy(img))
            _output_images.append(image_new)
            _output_jsons.append(copy.deepcopy(output_jsons[i]))
        output_images.extend(_output_images)
        output_jsons.extend(_output_jsons)
    if 'contrast' in augmentation_type:
        _output_images = []
        _output_jsons = []
        for i, img in enumerate(output_images):
            image_new = image_increase_contrast(copy.deepcopy(img))
            _output_images.append(image_new)
            _output_jsons.append(copy.deepcopy(output_jsons[i]))
        output_images.extend(_output_images)
        output_jsons.extend(_output_jsons)

    if save:
        for i in range(0, len(output_images)):
            img = output_images[i]
            fname = copy.deepcopy(basename) + '_' + str(i) + '.' + out_suffix
            output_jsons[i]['folder'] = output_img_folder
            output_jsons[i]['filename'] = fname
            json_name = os.path.join(output_json_folder,
                                     fname + '.json')
            fname = os.path.join(output_img_folder, fname)
            print('[data_augmentation_detection] Save: ' + fname)
            image.save_img(img, fname=fname)
            tools.write_json(output_jsons[i], json_name)
    return output_images


def data_augmentation_new(image_filename, output_folder,
                          augmentation_type, save=True, out_suffix='jpg'):

    fname_lst = os.path.basename(image_filename).split('.')
    suffix = '.' + fname_lst[-1]
    output_images = []

    original_image = image.read_img(image_filename)

    if 'padding' in augmentation_type:
        image_padded = image_padding(original_image)
        output_images.append(image_padded)
    else:
        output_images.append(original_image)

    if 'flip' in augmentation_type:
        _output_images = []
        for _img in output_images:
            for direction in ['h', 'v']:
                img = image.read_and_flip(_img, direction=direction)
                _output_images.append(img)
        output_images = _output_images

    elif 'flip_h' in augmentation_type or 'flip_v' in augmentation_type:
        _output_images = []
        for _img in output_images:
            if 'flip_h' in augmentation_type:
                img = image.read_and_flip(_img, direction='h')
                _output_images.append(img)
            if 'flip_v' in augmentation_type:
                img = image.read_and_flip(_img, direction='h')
                _output_images.append(img)
        output_images = _output_images

    if 'blur' in augmentation_type:
        kernel = np.ones((5, 5), np.float32) / 25
        _output_images = []
        for _img in output_images:
            image_new = cv2.filter2D(_img, -1, kernel)
            _output_images.append(img)
        output_images = _output_images

    if 'brightness' in augmentation_type:
        _output_images = []
        for _img in output_images:
            image_new = image_brighten(_img)
            _output_images.append(image_new)
            image_new = image_darken(_img)
            _output_images.append(image_new)
            image_new = image_increase_contrast(_img)
            _output_images.append(image_new)
        output_images = _output_images

    if save:
        for i in range(0, len(output_images)):
            _fname = copy.deepcopy(fname_lst)
            _fname.insert(-1, str(i))
            _fname[-1] = out_suffix
            fname = '.'.join(_fname)
            fname = os.path.join(output_folder, fname)
            print(fname)
            image.save_img(img, fname=fname)
    return output_images


def data_augmentation(
        image_filename,
        output_folder,
        augmentation_type,
        save=True,
        out_suffix='jpg'):
    """ Data augmentation for classification.

    @param image_filename: input image to be augmented
    @param output_folder: augmentation results will be saved to
    @param augmentation_type: list of types including 'padding',
           'flip', 'flip_v', 'flip_h', 'blur', 'darken', 'brighten',
           'contrast'
    @param save: True to save output image and json
    @param out_suffix: suffix of output images
    """

    tools.check_dir(output_folder)
    basename = os.path.basename(image_filename).split('.')[0]

    original_image = cv2.imread(image_filename)
    output_images = []
    if 'padding' in augmentation_type:
        image_padded = image_padding(original_image)
        output_images.append(image_padded)
    else:
        output_images.append(copy.deepcopy(original_image))

    if 'flip' in augmentation_type or \
            ('flip_h' in augmentation_type and 'flip_v' in augmentation_type):
        _output_images = []
        for i, img in enumerate(output_images):
            for direction in ['h']:
                img_flipped = image.read_and_flip(
                    copy.deepcopy(img), direction)
                _output_images.append(img_flipped)
        output_images.extend(_output_images)
        _output_images = []
        for i, img in enumerate(output_images):
            for direction in ['v']:
                img_flipped = image.read_and_flip(
                    copy.deepcopy(img), direction)
                _output_images.append(img_flipped)
        output_images.extend(_output_images)
    elif 'flip_h' in augmentation_type:
        _output_images = []
        for i, img in enumerate(output_images):
            for direction in ['h']:
                img_flipped = image.read_and_flip(
                    copy.deepcopy(img), direction)
                _output_images.append(img_flipped)
        output_images.extend(_output_images)
    elif 'flip_v' in augmentation_type:
        _output_images = []
        for i, img in enumerate(output_images):
            for direction in ['v']:
                img_flipped = image.read_and_flip(
                    copy.deepcopy(img), direction)
                _output_images.append(img_flipped)
        output_images.extend(_output_images)

    if 'blur' in augmentation_type:
        kernel = np.ones((5, 5), np.float32) / 25
        _output_images = []
        for i, img in enumerate(output_images):
            image_new = cv2.filter2D(copy.deepcopy(img), -1, kernel)
            _output_images.append(image_new)
        output_images.extend(_output_images)

    if 'brighten' in augmentation_type:
        _output_images = []
        for i, img in enumerate(output_images):
            image_new = image_brighten(copy.deepcopy(img))
            _output_images.append(image_new)
        output_images.extend(_output_images)
    if 'darken' in augmentation_type:
        _output_images = []
        for i, img in enumerate(output_images):
            image_new = image_darken(copy.deepcopy(img))
            _output_images.append(image_new)
        output_images.extend(_output_images)
    if 'contrast' in augmentation_type:
        _output_images = []
        for i, img in enumerate(output_images):
            image_new = image_increase_contrast(copy.deepcopy(img))
            _output_images.append(image_new)
        output_images.extend(_output_images)

    if save:
        for i in range(0, len(output_images)):
            img = output_images[i]
            fname = copy.deepcopy(basename) + '_' + str(i) + '.' + out_suffix
            fname = os.path.join(output_folder, fname)
            print('[data_augmentation_classification] Save: ' + fname)
            image.save_img(img, fname=fname)
    return output_images


#    fn = os.path.basename(image_filename)
#    fn = fn.replace(suffix, '')
#    if 'padding' in augmentation_type:
#        original_image = cv2.imread(image_filename)
#        image_padded = image_padding(original_image)
#        image_padded_filename = output_folder + fn + '_padded'
#        cv2.imwrite(image_padded_filename + suffix, image_padded)
#        image_filename_list = [image_padded_filename]
#    else:
#        os.system('cp ' + image_filename + ' ' + output_folder)
#        image_filename_list = [output_folder + fn]
#
#    if 'flip' in augmentation_type:
#        for i in range(len(image_filename_list)):
#            image_filename = image_filename_list[i]
#            image_filename_full = image_filename + suffix
#            print(image_filename_full)
#            for direction in ['h', 'v']:
#                img_flipped = image.read_and_flip(
#                    copy.deepcopy(img), direction)
#                img, fname = image.read_and_flip_for_tinycv(
#                    image_filename_full, direction=direction, save=True
#                )
#                image_filename_list.append(fname)
#
#    if 'blur' in augmentation_type:
#        kernel = np.ones((5, 5), np.float32) / 25
#        for i in range(len(image_filename_list)):
#            image_filename = image_filename_list[i]
#            img = cv2.imread(image_filename + suffix)
#            image_new = cv2.filter2D(img, -1, kernel)
#            image_new_filename = image_filename + '_blur'
#            cv2.imwrite(image_new_filename + suffix, image_new)
#            image_filename_list.append(image_new_filename)
#
#    if 'brightness' in augmentation_type:
#        for i in range(len(image_filename_list)):
#            image_filename = image_filename_list[i]
#            img = cv2.imread(image_filename + suffix)
#            image_new = image_brighten(img)
#            image_new_filename = image_filename + '_brighten'
#            cv2.imwrite(image_new_filename + suffix, image_new)
#            image_filename_list.append(image_new_filename)
#            image_new = image_darken(img)
#            image_new_filename = image_filename + '_darken'
#            cv2.imwrite(image_new_filename + suffix, image_new)
#            image_filename_list.append(image_new_filename)
#            image_new = image_increase_contrast(img)
#            image_new_filename = image_filename + '_contrast'
#            cv2.imwrite(image_new_filename + suffix, image_new)
#            image_filename_list.append(image_new_filename)


def image_increase_contrast(img):
    # LAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels
    new_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return(new_image)


def txt_to_bounding_box(yolo_result):
    """parsing yolo result from txt to bounding_box.
    """
    bounding_box = np.array([], dtype=np.int).reshape(0, 4)
    with open(yolo_result) as data_file:
        data = data_file.readlines()
        for line in data:
            words = line.split()
            if words[0] == "person":
                bounding_box = np.row_stack([
                    bounding_box,
                    # top, bottom, left, right
                    [
                        int(words[3]),
                        int(words[3]) + int(words[5]),
                        int(words[2]),
                        int(words[2]) + int(words[4])
                    ]
                ])
    return bounding_box


def json_to_bounding_box(json_file_name):
    """parsing yolo result from json to bounding_box.
    """
    json_data = {}
    with open(json_file_name, 'r') as f:
        json_data = json.load(f)
    bounding_box = np.array([], dtype=np.int).reshape(0, 4)
    for ri in range(0, len(json_data)):
        if json_data[ri]["label"] == "person":
            bounding_box = np.row_stack(
                [bounding_box,
                    [int(json_data[ri]["topleft"]["y"]),
                     int(json_data[ri]["bottomright"]["y"]),
                     int(json_data[ri]["topleft"]["x"]),
                     int(json_data[ri]["bottomright"]["x"])]])
    return bounding_box


def padding_images(images, mode='center'):
    """Padding images to size
    """
    return [image_padding(np.array(img), mode) for img in images]


def merge_4_channel_images(images, size):
    """Paste 4 images into a single image.

     ------------------- --------------------
    | image top-left    | image top-right    |
     ------------------- --------------------
    | image button-left | image button-right |
     ------------------- --------------------
    """
    from PIL import Image
    to_image = Image.new('RGB', (size[0] * 2, size[1] * 2))
    for i in range(4):
        from_image = images[i]
        loc = ((int(i / 2) * size[0]), (i % 2) * size[1])
        to_image.paste(from_image, loc)
    return(np.array(to_image), size[0] * 2, size[1] * 2)


def merge_4_channel_images_opencv(images):
    """merge 4 images
    """
    image_top = np.concatenate((images[0], images[2]), axis=1)
    image_bottom = np.concatenate((images[1], images[3]), axis=1)
    out_image = np.concatenate((image_top, image_bottom), axis=0)
    return(out_image, int(image_top.shape[1] / 2), image_top.shape[0])


def split_bounding_box(bounding_box, width, height):
    """split bounding_box of a 4-channel-merged image.

     ------ ------  <--
    | img0 | img2 |   | height
     ------ ------  <--
    | img1 | img3 |
     ------ ------
    ^      ^
    |______|
      width

    :param width: width of a snapshot in a grid image
    :type width: int
    :param width: height of a snapshot in a grid image
    :type height: int
    :return: bounding box list, and channel index list of the bounding boxes
    :rtype: tuple
    """
    # Defensive programming: ensure bounding box values to be integers.
    #
    # If width or height is float, bounding box values will become float
    # after computations. This violates bounding box's definition.
    width = int(width)
    height = int(height)

    person_number = bounding_box.shape[0]
    channel_index = [0] * person_number

    # split vertically
    for i in range(person_number):
        if bounding_box[i][0] >= height:
            channel_index[i] = 1
        elif bounding_box[i][1] >= height:
            channel_index[i] = 0
            channel_index.append(1)
            bounding_box = np.row_stack(
                [bounding_box,
                    [height, bounding_box[i][1],
                     bounding_box[i][2], bounding_box[i][3]]])
            bounding_box[i][1] = height - 1

    # split horizontally
    person_number = bounding_box.shape[0]
    for i in range(person_number):
        if bounding_box[i][2] >= width:
            channel_index[i] = channel_index[i] + 2
        elif bounding_box[i][3] >= width:
            channel_index.append(channel_index[i] + 2)
            bounding_box = np.row_stack(
                [bounding_box,
                    [bounding_box[i][0], bounding_box[i][1],
                     width, bounding_box[i][3]]])
            bounding_box[i][3] = width - 1

    return (bounding_box, channel_index)


def image_padding(img, mode='center'):
    """Pad an image from non-square rectangle to square by 1.

    In mode 'center', the original image is put in the center
    after padding.

    In mode 'topleft', the original image in put in the top left corner
    after padding.
    """
    return image.auto_padding(img, mode)


def grouping(bounding_box, grouping_ratio):
    """group bounding_box when horizontally close and vertically overlap.
    """
    if bounding_box.shape[0] < 2:
        return bounding_box

    person_width = max(bounding_box[:, 3] - bounding_box[:, 2])
    change = 1
    while change == 1:
        person_number = bounding_box.shape[0]
        change = 0
        group = range(person_number)
        group = np.array(group)
        out_bounding_box = np.array([], dtype=np.int).reshape(0, 4)
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
                diff_rl = float(max(right_i, right_j) - min(left_i, left_j))
                diff_tb = min(bottom_i, bottom_j) - max(top_i, top_j)
                # group bounding_box when
                # 1) horizontally close enough:
                #       diff_rl < person_width * (2 + grouping_ratio)
                # 2) vertically overlap: diff_tb > 0.
                if (diff_rl < person_width * (2 + grouping_ratio)
                        and diff_tb) > 0:
                    group[j] = group[i]
                    change = 1
        for i in range(0, person_number):
            index = (group == i).nonzero()[0]
            if len(index) > 0:
                left = bounding_box[index[0]][2]
                top = bounding_box[index[0]][0]
                right = bounding_box[index[0]][3]
                bottom = bounding_box[index[0]][1]
                for j in range(1, len(index)):
                    left = min(bounding_box[index[j]][2], left)
                    top = min(bounding_box[index[j]][0], top)
                    right = max(bounding_box[index[j]][3], right)
                    bottom = max(bounding_box[index[j]][1], bottom)
                out_bounding_box = np.row_stack(
                    [out_bounding_box, [top, bottom, left, right]])
        bounding_box = out_bounding_box
    return out_bounding_box


def grouping_multi_channels(
        bounding_box, grouping_ratio, channel_index, channel_number):
    """group bounding_box on each channel separately.
    """
    print(bounding_box)
    print(channel_index)
    out_bounding_box = np.array([], dtype=np.int).reshape(0, 4)
    out_channel_index = []
    for ci in range(channel_number):
        index = [i for i, x in enumerate(channel_index) if x == ci]
        sub_bounding_box = grouping(bounding_box[index], grouping_ratio)
        print(ci)
        print(sub_bounding_box)
        out_bounding_box = np.row_stack([out_bounding_box, sub_bounding_box])
        for ni in range(sub_bounding_box.shape[0]):
            out_channel_index.append(ci)
    return(out_bounding_box, out_channel_index)


def tracking(bounding_box, bounding_box_track, r, fi, track_frame_number):
    """match bounding_box when overlap in current and referance image
    """
    person_number = bounding_box.shape[0]
    person_number_all = bounding_box_track.shape[0]
    frame_index = np.arange(bounding_box.shape[0])
    frame_index = (frame_index.transpose() + 1) * -1
    overlap = np.zeros((bounding_box.shape[0],
                        bounding_box_track.shape[0]), dtype=np.int)
    for i in range(0, person_number):
        right_i = bounding_box[i][3]
        left_i = bounding_box[i][2]
        bottom_i = bounding_box[i][1]
        top_i = bounding_box[i][0]
        area_i = (bottom_i - top_i) * (right_i - left_i)
        for j in range(0, person_number_all):
            right_j = bounding_box_track[j][3]
            left_j = bounding_box_track[j][2]
            bottom_j = bounding_box_track[j][1]
            top_j = bounding_box_track[j][0]
            area_j = (bottom_j - top_j) * (right_j - left_j)
            overlap_width = float(min(right_i, right_j) - max(left_i, left_j))
            overlap_height = float(min(bottom_i, bottom_j) - max(top_i, top_j))
            overlap_area = overlap_width * overlap_height
            # match bounding_box in current and reference image when
            # 1) horizontally overlap: overlap_width > 0
            # 2) vertically overlap: overlap_height > 0
            # 3) overlap area is large enough:
            #       overlap_area > min(area_i, area_j) * r
            # 4) time difference between current and reference image
            #    is small enough:
            #       fi - bounding_box_track[j][4] < track_frame_number
            if (overlap_width > 0 and
                    overlap_height > 0 and
                    overlap_area > min(area_i, area_j) * r and
                    fi - bounding_box_track[j][4] < track_frame_number):
                # Then we:
                overlap[i][j] = overlap_area / (
                    max(area_i, area_j) / min(area_i, area_j))
    while overlap.max() > 0:
        index = np.argwhere(overlap == overlap.max())
        i = index[0, 0]
        j = index[0, 1]
        frame_index[i] = j
        overlap[i, :] = 0
        overlap[:, j] = 0
    # new person
    index = np.argwhere(frame_index < 0)
    for k in index:
        i = k[0]
        # check
        right_i = bounding_box[i][3]
        left_i = bounding_box[i][2]
        bottom_i = bounding_box[i][1]
        top_i = bounding_box[i][0]
        area_i = (bottom_i - top_i) * (right_i - left_i)
        found = 0
        for j in range(0, person_number):
            if j != i:
                right_j = bounding_box[j][3]
                left_j = bounding_box[j][2]
                bottom_j = bounding_box[j][1]
                top_j = bounding_box[j][0]
                area_j = (bottom_j - top_j) * (right_j - left_j)
                overlap_width = float(
                    min(right_i, right_j) - max(left_i, left_j))
                overlap_height = float(
                    min(bottom_i, bottom_j) - max(top_i, top_j))
                overlap_area = overlap_width * overlap_height
                if (overlap_width > 0 and
                        overlap_height > 0 and
                        overlap_area > min(area_i, area_j) * 0.8):
                    frame_index[i] = frame_index[j]
                    found = 1
        if found == 0:
            bounding_box_track = np.row_stack(
                [bounding_box_track,
                    [bounding_box[i][0], bounding_box[i][1],
                     bounding_box[i][2], bounding_box[i][3], 0]])
            frame_index[i] = bounding_box_track.shape[0] - 1
    return (frame_index, bounding_box_track)


def sort_by_area(bounding_box,
                 is_multi_channel=False, channel_index=[]):
    """sort bounding_box by area from largest to smallest.
    """
    out_bounding_box = bounding_box
    person_number = bounding_box.shape[0]
    z = np.zeros((person_number, 1), int)
    bounding_box = np.c_[bounding_box, z]
    for i in range(0, person_number):
        bounding_box[i][4] = ((bounding_box[i][1] - bounding_box[i][0]) *
                              (bounding_box[i][3] - bounding_box[i][2]) * (-1))
    order = np.argsort(bounding_box[:, 4])
    out_bounding_box = out_bounding_box[order]
    if is_multi_channel:
        channel_index = [channel_index[x] for x in order]
        return (out_bounding_box, channel_index)
    else:
        return (out_bounding_box, [0] * person_number)


def sort_by_aspect_ratio(bounding_box,
                         is_multi_channel=False, channel_index=[]):
    """
    sort bounding_box by aspect_ratio(width/height)
    from largest to smallest.
    """
    out_bounding_box = bounding_box
    person_number = bounding_box.shape[0]
    z = np.zeros((person_number, 1), float)
    bounding_box = np.c_[bounding_box, z]
    for i in range(0, person_number):
        width = float(bounding_box[i][1] - bounding_box[i][0])
        height = float(bounding_box[i][3] - bounding_box[i][2])
        bounding_box[i][4] = width / height
    order = np.argsort(bounding_box[:, 4])
    out_bounding_box = out_bounding_box[order]
    if is_multi_channel:
        channel_index = [channel_index[x] for x in order]
        return (out_bounding_box, channel_index)
    else:
        return (out_bounding_box, [0] * person_number)


def sort_by_diff(
        frame_index, previous_frame_index, bounding_box,
        previous_bounding_box, img, previous_image):
    """
    sort bounding_box by difference in current image and previous image
    from largest to smallest.
    """
    max_diff = 0
    max_index = 0
    for i in range(0, bounding_box.shape[0]):
        index = (previous_frame_index == frame_index[i]).nonzero()[0]
        print(index)
        if len(index) > 0:
            for j in index:
                print(j)
                right_i = bounding_box[i][3]
                right_j = previous_bounding_box[j][3]
                left_i = bounding_box[i][2]
                left_j = previous_bounding_box[j][2]
                bottom_i = bounding_box[i][1]
                bottom_j = previous_bounding_box[j][1]
                top_i = bounding_box[i][0]
                top_j = previous_bounding_box[j][0]
                right = min(right_i, right_j)
                left = max(left_i, left_j)
                bottom = min(bottom_i, bottom_j)
                top = max(top_i, top_j)
                print([left, right, top, bottom])
                diff = (img[top:bottom, left:right, :] -
                        previous_image[top:bottom, left:right, :])
                diff = abs(diff).sum()
                if diff > max_diff:
                    max_diff = diff
                    max_index = i
    if max_index > 0:
        frame_index[[0, max_index]] = frame_index[[max_index, 0]]
        bounding_box[[0, max_index]] = bounding_box[[max_index, 0]]
    return (frame_index, bounding_box)


def check_boundary(loc, mergin, width, height):
    """check if bounding_box exceed image boundary after padding a mergin.
    """
    (top, bottom, left, right) = loc
    top = max(top - mergin, 0)
    left = max(left - mergin, 0)
    bottom = min(bottom + mergin, height - 1)
    right = min(right + mergin, width - 1)
    out = np.array([top, bottom, left, right])
    return out


def imwrite_seg(out_file_name, img, loc):
    """write image segmented according to loc.
    """
    (top, bottom, left, right) = loc
    cv2.imwrite(out_file_name, img[top:bottom, left:right, :])


def rgb_to_bgr(numpy_image):
    """Convert image color model from RGB to BGR in numpy array.

    OpenCV uses BGR by default because of historical reason. For more
    information, please refer to

    https://www.learnopencv.com/why-does-opencv-use-bgr-color-format/
    https://stackoverflow.com/questions/42406338
    """
    return numpy_image[:, :, [2, 1, 0]]


def _draw_bounding_box(img, loc, color, width, idx):
    """draw bounding box in one dimension according to loc.
    """

    top = loc[0] if loc[0] >= 0 else 0
    bottom = loc[1]
    left = loc[2] if loc[2] >= 0 else 0
    right = loc[3]

    img[top:bottom, left:np.minimum(left + width, right), idx:idx + 1] = color
    img[top:bottom, np.maximum(right - width, left):right, idx:idx + 1] = color
    img[top:np.minimum(top + width, bottom), left:right, idx:idx + 1] = color
    img[np.maximum(bottom - width, top):bottom,
        left:right, idx:idx + 1] = color

    return img


def draw_bounding_box(img, loc, color, width):
    """draw bounding box on image in three dimensions.
    """

    if not isinstance(color, list):
        color_list = [color, color, color]
    else:
        color_list = color
    for i in range(0, 3):
        img = _draw_bounding_box(img, loc, color_list[i], width, i)

    return img


def crop_img_rect_rgb(img, rect):
    """ Crop img based on the given Rect object """

    if not isinstance(img, np.ndarray):
        tools.check_exist(img)
        img = image.read_img(img)

    if image.is_rgb(img):
        return img[rect.t:rect.b, rect.l:rect.r, :]
    else:
        return img[rect.t:rect.b, rect.l:rect.r]


def crop_bb_img_lab(img, out_file_name, json_file, box_number='all',
                    thre=-1, skipped_labels=[], square_extend=True,
                    append_label=False, skip_null=True, space=0,
                    margin_multiplier=0.1):
    """Crop and save image to square according to bounding boxes.

    @param img: path of the image or the img tensor
    @param json_file: json created by dyda_utils.data

    Keyword parameters:
    box_number       : maximum number of boxes to save
    thre             : threshold to filter the results (should between 0-1)
    skipped_labels   : a list of labels to be skipped
    square_extend    : extend the cropped region to square shape
    append_label     : append labels to the output filenames
    skip_null        : skip labels of empty strings
    space            : cropped space
    margin_multiplier: only works if space < 0

    """

    if (thre > 1 or thre < 0) and thre != -1:
        print("WARNING: threshould shoule between 0-1, current valud %.2f."
              % thre)
    if isinstance(json_file, str):
        bb_info = data.parse_json(json_file)
    else:
        bb_info = json_file

    if not isinstance(img, np.ndarray):
        tools.check_exist(img)
        img = image.read_img(img)

    height = img.shape[0]
    width = img.shape[1]

    if box_number == 'all':
        box_number = len(bb_info["annotations"])
    else:
        box_number = min(len(bb_info["annotations"]), int(box_number))

    for i in range(0, box_number):
        conf = bb_info["annotations"][i]["confidence"]
        if bb_info["annotations"][i]["label"] in skipped_labels:
            continue
        if bb_info["annotations"][i]["label"] == "" and skip_null:
            continue
        if (thre > 0 and conf < thre):
            continue

        rect = conv_lab_anno_rect(bb_info["annotations"][i])

        margin = 0
        if space >= 0:
            margin = space
        else:
            if 0 <= margin_multiplier and 1 >= margin_multiplier:
                margin = int(min(rect.h, rect.w) * margin_multiplier)
            else:
                print("WARNING: margin_multiplier range should be between 0-1")

        if margin > 0:
            rect = extend_rect(
                rect, margin, margin, max_h=height, max_w=width
            )

        loc = conv_rect_bb(rect)
        if square_extend:
            loc = boxes.square_extend(loc, width, height)
        final_out_name = out_file_name + '_' + str(i) + '.png'
        if append_label:
            final_out_name = final_out_name[:-4] + \
                '_' + bb_info["annotations"][i]["label"] + '.png'
        imwrite_seg(final_out_name, img, loc)


def patch_bb_by_key(json_file, color=[0, 0, 255], line_width=6, keys=['label'],
                    save=False, space=40, output_path=""):
    """Show detection results by patching a rectangle and key value
    to the image.
    @param json_file: json created by dyda_utils.data

    Keyword parameters:
    color     : color of the bounding box in turns of [B, G, R]
    save      : save the file as ${json_file}.jpg
    space     : space between bb and text
    thre      : threshold to filter the results (should between 0-1)

    """
    if isinstance(json_file, dict):
        results = json_file
    else:
        results = data.parse_json(json_file)
    img_filename = os.path.join(
        results['folder'],
        results['filename'])
    img_array = image.read_img(img_filename)
    img = img_array

    for i in range(0, len(results["annotations"])):
        loc = (results["annotations"][i]["top"],
               results["annotations"][i]["bottom"],
               results["annotations"][i]["left"],
               results["annotations"][i]["right"])

        img = draw_bounding_box(img_array, loc, color, line_width)
        text = ''

        for key in keys:
            if key not in results["annotations"][i].keys():
                continue
            value = results["annotations"][i][key]
            if isinstance(value, (float)):
                value = "{0:.2f}".format(value)
            elif isinstance(value, (int)):
                value = str(value)
            text += value + ' '
        img = patch_text(img, text, color=color,
                         loc=(results["annotations"][i]["left"] + space,
                              results["annotations"][i]["top"] + space))

    if save:
        if output_path == "":
            output_path = "./patched.jpg"
        image.save_img(img, fname=output_path)
        print('[patch_bb_by_key] Save:' + output_path)

    return img


def patch_bb_trainer(img_array, results, color=[0, 0, 255], patch_label=True,
                     patch_perc=True, line_width=6, save=False, label_loc="up",
                     space=40, thre=-1, output_path="./trainer.jpg"):
    """Open an image and patch a rectangle to it

    @param img: path of the image or the img tensor
    @param json_file: json created by dyda_utils.data

    Keyword parameters:
    color     : color of the bounding box in turns of [B, G, R]
    patch_label: True to patch label to the lt corner of the bb
    patch_perc: True to patch percentage (4 decimals)
    line_width: width of the bounding box (default: 6)
    save      : save the file as ${json_file}.jpg
    label_loc : up => top-left, down => bottom-left
    space     : space between bb and text
    thre      : threshold to filter the results (should between 0-1)

    """

    if (thre > 1 or thre < 0) and thre != -1:
        print("WARNING: threshould shoule between 0-1, current valud %.2f."
              % thre)

    img = img_array
    for i in range(0, len(results["annotations"])):
        conf = results["annotations"][i]["confidence"]
        if (thre > 0 and conf < thre):
            continue
        loc = (int(results["annotations"][i]["top"]),
               int(results["annotations"][i]["bottom"]),
               int(results["annotations"][i]["left"]),
               int(results["annotations"][i]["right"]))

        img = draw_bounding_box(img_array, loc, color, line_width)
        if patch_label:
            text = results["annotations"][i]["label"]
            if patch_perc:
                if isinstance(conf, str):
                    text = text + conf
                else:
                    text = text + ' ' + "{0:.2f}".format(conf)
            if label_loc == "down":
                img = patch_text(img, text, color=color,
                                 loc=(loc[2] + space, loc[1] + space))
            else:
                img = patch_text(img, text, color=color,
                                 loc=(loc[2] + space, loc[0] + space))

    if save:
        output = output_path
        image.save_img(img, fname=output)

    return img


def patch_bb_img_lab(img, json_file, color=[0, 0, 255], patch_label=True,
                     patch_perc=True, line_width=6, save=False,
                     space=40, thre=-1):
    """Open an image and patch a rectangle to it

    @param img: path of the image or the img tensor
    @param json_file: json created by dyda_utils.data

    Keyword parameters:
    color     : color of the bounding box in turns of [B, G, R]
    patch_label: True to patch label to the lt corner of the bb
    patch_perc: True to patch percentage (4 decimals)
    line_width: width of the bounding box (default: 6)
    save      : save the file as ${json_file}.jpg
    space     : space between bb and text
    thre      : threshold to filter the results (should between 0-1)

    """

    results = data.parse_json(json_file)

    if not isinstance(img, np.ndarray):
        tools.check_exist(img)
        img_array = image.read_img(img)
    else:
        img_array = img

    output = json_file + ".jpg"

    img = patch_bb_trainer(img_array, results, color=color,
                           patch_label=patch_label, patch_perc=patch_perc,
                           line_width=line_width, save=save,
                           space=space, thre=thre, output_path=output)
    return img


def patch_bb_img(img, loc, color=[0, 0, 255], line_width=6):
    """Open an image and patch a rectangle to it

    @param img: path of the image or the img tensor
    @param loc: position list, (top, bottom, left, right)

    Keyword parameters:
    color     : color of the bounding box in turns of [B, G, R]
    line_width: width of the bounding box (default: 6)

    """

    if not isinstance(img, np.ndarray):
        tools.check_exist(img)
        img = image.read_img(img)

    return draw_bounding_box(img, loc, color, line_width)


def patch_text(img, text, loc=(40, 40), color=(255, 255, 255),
               fontscale=1, thickness=2):
    """Patch the text to the image

    @param img: Image tensor (numpy array)
    @param text: Text to be patched

    Keyword parameters:
    loc       : location vector (x, y)
    color     : color vector (B, G, R)
    fontscale : font size
    thickness : thickness of the text

    """

    return cv2.putText(img, text, loc, cv2.FONT_HERSHEY_SIMPLEX,
                       fontscale, color, thickness, cv2.LINE_AA)


def find_bb_center(loc):
    """Find the center of the given bounding box"""

    (top, bottom, left, right) = loc
    y = int((top + bottom) / 2)
    x = int((left + right) / 2)
    return (x, y)


def find_bb_center_rect(rect):
    """Find the center of the  object defined in tinycv"""
    y = int((rect.t + rect.b) / 2)
    x = int((rect.l + rect.r) / 2)
    return (x, y)


def find_offset_rect(rect_ref, rect_tar):
    """
    Find the offset of centers of two given rect
    rect_tar_center + offset = rect_ref_center
    return: offset_x, offset_y
    """

    ref_center = find_bb_center_rect(rect_ref)
    tar_center = find_bb_center_rect(rect_tar)
    return (ref_center[0] - tar_center[0],
            ref_center[1] - tar_center[1])


def check_boundary_limit(boundary, limit):
    """ Check if the boundary exist 0 or width/height """

    if boundary < 0:
        return 0
    if boundary > limit:
        return limit
    return boundary


def shift_rect(rect, delta_x, delta_y):
    """ Shift the Rect by delta_x and delta_y """

    _new_t = check_boundary_limit(rect.t + delta_y, rect.h)
    _new_b = check_boundary_limit(rect.b + delta_y, rect.h)
    _new_l = check_boundary_limit(rect.l + delta_x, rect.w)
    _new_r = check_boundary_limit(rect.r + delta_x, rect.w)

    return Rect([_new_t, _new_b, _new_l, _new_r])


def extend_rect(rect, delta_x, delta_y, max_w=-1, max_h=-1):
    """ Extend the Rect by delta_x and delta_y """

    _new_t = check_boundary_limit(rect.t - delta_y, max(rect.h, max_h))
    _new_b = check_boundary_limit(rect.b + delta_y, max(rect.h, max_h))
    _new_l = check_boundary_limit(rect.l - delta_x, max(rect.w, max_w))
    _new_r = check_boundary_limit(rect.r + delta_x, max(rect.w, max_w))

    return Rect([_new_t, _new_b, _new_l, _new_r])


def conv_lab_anno_rect(anno):
    """ Convert lab annotation to Rect object """
    loc = (anno["top"], anno["bottom"], anno["left"], anno["right"])
    return Rect(loc)


def conv_bb_rect(loc):
    """ Convert bounding box loc to Rect object """
    return Rect(loc)


def conv_rect_bb(rect):
    """ Convert  object to lab loc definition (t, b, l, r) """
    return (rect.t, rect.b, rect.l, rect.r)


def find_rect_union(recs):
    """
    Find union from a list of Rect object defined in tinycv
    Benchmark: < 0.05 ms to find union of three recs on gc2
    """

    t_list = []
    b_list = []
    l_list = []
    r_list = []
    for rec in recs:
        t_list.append(rec.t)
        b_list.append(rec.b)
        l_list.append(rec.l)
        r_list.append(rec.r)
    return Rect([min(t_list), max(b_list), min(l_list), max(r_list)])


def patch_rect_img(img, rect, color=[0, 0, 255], line_width=6):
    """Open an image and patch the defined  objec to it

    @param img: path of the image or the img tensor
    @param loc: position list, (top, bottom, left, right)

    Keyword parameters:
    color     : color of the bounding box in turns of [B, G, R]
    line_width: width of the bounding box (default: 6)

    """

    if not isinstance(img, np.ndarray):
        tools.check_exist(img)
        img = image.read_img(img)

    return draw_bounding_box(img, conv_rect_bb(rect), color, line_width)


def find_true_slice(boolean_list):
    """Find true slice in the given boolean_list

    @param boolean_list: list of True and False
    @return true_slice: boolean_list[true_slice[i][0]:true_slice[i][1]] = True
    """

    flag_change = list(np.convolve(boolean_list, [1, -1], 'same'))
    true_start = [
        i for i, x in enumerate(flag_change) if x == 1]
    true_end = [
        i for i, x in enumerate(flag_change) if x == -1]
    if len(true_end) == len(true_start) - 1:
        true_end.append(len(boolean_list))

    true_slice = []
    for i in range(len(true_start)):
        true_slice.append([true_start[i], true_end[i]])

    return(true_slice)


def segmentation_by_projection(binary_img, proj_mode,
                               length_ratio_thre,
                               percentile_thre,
                               target=0):
    """Calculate segmentation indices by projection.

       @param binary_img: binarized image with only 0 for background
            and 1 for foreground.
       @param proj_mode: 'h' for horizontal projection and
            'v' for vertical projection.
       @param length_ratio_thre: segmentation is kept
            when length_ratio > length_ratio_thre.
       @param percentile_thre: image is segmented when
            projection_ratio > (percentile_thre)-th percentile
            of all projection ratio.

       @return indices_seg: segmentation indices
    """
    if target == 1:
        binary_img = binary_img.astype(np.int)
        binary_img = abs(binary_img - 1)

    if proj_mode == 'h':
        proj_idx = 1
        total_length = binary_img.shape[0]
    elif proj_mode == 'v':
        proj_idx = 0
        total_length = binary_img.shape[1]

    # Extract low brightness region
    proj_sum = binary_img.sum(proj_idx)
    proj_length = binary_img.shape[proj_idx]
    proj_ratio = proj_sum / proj_length
    if percentile_thre > 1:
        proj_thre = np.percentile(proj_ratio, percentile_thre)
    else:
        proj_thre = percentile_thre
    low_ratio_flag = proj_ratio < proj_thre
    indices_seg = find_true_slice(low_ratio_flag)

    # Only length ratio > length_ratio_thre kept
    length = [x[1] - x[0] for x in indices_seg]
    length_ratio = [x / total_length for x in length]
    for i in range(len(indices_seg) - 1, -1, -1):
        if length_ratio[i] < length_ratio_thre:
            indices_seg.pop(i)

    return indices_seg


def projcetion_feature(label):
    """Extract feature from horizontal projection and vertical projection.

       @param label: binary image with only 0 and 1.
       @return feature: feature from horizontal projection and vertical
           projection

    """
    row_sum = list(label.sum(1))
    column_sum = list(label.sum(0))
    feature = row_sum + column_sum
    return feature


def rotate_and_extract_box(img, angle_start=-45, angle_end=46):
    """Rotate and extract one box in dark background.

       @param img: color image with one box in dark background.
       @param angle_start, angle_end: find the angle between angle_start
            and angle_end to make the box straight.

       @return img_box: color image with box only
    """

    # rgb to gray
    img_g = cv2.cvtColor(
        image.resize_img(img, (100, None)), cv2.COLOR_RGB2GRAY)

    # image binarization global
    thre = int(np.mean(img_g.mean(axis=1)))
    ret, img_b = cv2.threshold(
        img_g, thre - 10, 1, cv2.THRESH_BINARY)

    # find angle
    min_height = np.inf
    for angle in range(angle_start, angle_end, 3):
        img_r = rotate(img_b, angle, pivot='center')
        proj = segmentation_by_projection(img_r, 'h',
                                          0, 80, target=1)
        length = [x[1] - x[0] for x in proj]
        idx = length.index(max(length))
        height = proj[idx][1] - proj[idx][0]
        if height < min_height:
            min_height = height
            min_angle = [angle]
        elif height == min_height:
            min_angle.append(angle)
    min_angle = min_angle[int(np.round(len(min_angle) / 2.0))]

    min_height = np.inf
    angle_start = min_angle - 3
    angle_end = min_angle + 3
    for angle in np.arange(angle_start, angle_end, 0.5):
        img_r = rotate(img_b, angle, pivot='center')
        proj = segmentation_by_projection(img_r, 'h',
                                          0, 80, target=1)
        length = [x[1] - x[0] for x in proj]
        idx = length.index(max(length))
        height = proj[idx][1] - proj[idx][0]
        if height < min_height:
            min_height = height
            min_angle = [angle]
        elif height == min_height:
            min_angle.append(angle)

    # rotation
    min_angle = min_angle[int(np.round(len(min_angle) / 2.0))]
    img_r = rotate(img, min_angle)

    # rgb to gray
    img_g = cv2.cvtColor(
        img_r, cv2.COLOR_RGB2GRAY)

    # image binarization global
    thre = int(np.mean(img_g.mean(axis=1)))
    ret, img_b = cv2.threshold(
        img_g, thre - 10, 1, cv2.THRESH_BINARY)
    cv2.imwrite('bin.png', img_b * 255)

    # find boundary
    proj_h = segmentation_by_projection(img_b, 'h',
                                        0, 85, target=1)
    length = [x[1] - x[0] for x in proj_h]
    idx_h = length.index(max(length))
    proj_v = segmentation_by_projection(
        img_b[proj_h[idx_h][0]:proj_h[idx_h][1], :], 'v', 0, 0.5, target=1)
    length = [x[1] - x[0] for x in proj_v]
    idx_v = length.index(max(length))
    proj_h2 = segmentation_by_projection(
        img_b[proj_h[idx_h][0]:proj_h[idx_h][1],
              proj_v[idx_v][0]:proj_v[idx_v][1]],
        'h', 0, 0.5, target=1)
    top = proj_h[idx_h][0] + proj_h2[0][0]
    bottom = proj_h[idx_h][0] + proj_h2[0][1]
    img_box = img_r[top:bottom, proj_v[idx_v][0]:proj_v[idx_v][1], :]
    info_box = {
        "rot_angle": min_angle,
        "annotations": [{
            "bottom": bottom,
            "top": top,
            "right": proj_v[idx_v][1],
            "left": proj_v[idx_v][0]
        }]
    }
    return(img_box, info_box)
