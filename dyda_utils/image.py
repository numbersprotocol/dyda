import os
import sys
import subprocess
import numpy as np
import logging
import cv2
from PIL import Image
from dyda_utils import tools


def get_cv2_color_map(color_map="COLORMAP_JET"):
        """return cv2 color map
           color_map definitiion should match http://bit.ly/color_map
        """

        try:
            return getattr(cv2, color_map)
        except:
            self.logger.warning(
                "Unable to get %s, return COLORMAP_JET" % color_map
            )
            return cv2.COLORMAP_JET


def check_images_in_folder(folder, keyword=None, output="fail.json"):
    """Check if the images in the folder is good or not"""
    image_paths = find_images(dir_path=folder, keyword=keyword)
    fail_paths = []

    for img_path in image_paths:
        img = read_img(img_path, log=False)
        if img is None or img.shape[0] == 0:
            fail_paths.append(img_path)

    return fail_paths


def convert_all_to_png(folder, size=None, keyword=None,
                       outd=None, suffix='png'):
    """Convert the image to png
    @param folder: image folder to look for

    Keyword arguments:
    suffix -- suffix of the image to be converted
    size   --size of the output image
    keyword -- keyword of the image files to be converted
    outd   -- output folder for the converted file

    """

    image_paths = find_images(dir_path=folder, keyword=keyword)
    fail_paths = []

    if outd is None:
        outd = folder
    else:
        tools.check_dir(outd)

    for img_path in image_paths:
        img = read_img(img_path, log=False)
        if img is None or img.shape[0] == 0:
            print('dyda_utils: Error: %s cannot be read' % img_path)
            continue
        _fname = os.path.basename(img_path).split('.')
        _fname[-1] = suffix
        fname = '.'.join(_fname)
        fname = os.path.join(outd, fname)
        save_img(img, fname=fname)


def get_images_in_list(list_file, suffix=('.bmp', '.jpg', 'png', '.JPEG')):
    """Find images under a directory

    @param list_file: text file contains a list of image files

    Keyword arguments:
    suffix -- suffix of the image to be checked

    @return img_files: a list of image files
    """

    img_files = []
    with open(list_file, "r") as img_list:
        for img_row in img_list:
            img_file = img_row.split('\n')[0]
            if tools.check_ext(img_file, suffix):
                img_files.append(img_file)
    return img_files


def find_images(dir_path=None, walkin=True, keyword=None):
    """Find images under a directory

    Keyword arguments:
    dir_path -- path of the directory to check (default: '.')
    keyword  -- keyword used to filter images (default: None)
    walkin   -- True to list recursively (default: True)

    @return output: a list of images found

    """

    if dir_path is not None and os.path.isfile(dir_path):
        return [dir_path]
    return tools.find_files(dir_path=dir_path, keyword=keyword, walkin=walkin,
                            suffix=('.jpg', '.png', '.JPEG', '.bmp', '.gif'))


def get_images(path):
    """Find images from the given path"""

    if os.path.isfile(path):
        if tools.check_ext(path, ('.jpg', '.png', '.JPEG', '.bmp', '.gif')):
            return [path]
    elif os.path.isdir(path):
        return find_images(path)


def get_img_info(img_path):
    """Find image size and pixel array

    @param img_path: path of the input image

    @return image.size: tuple, size of the image
    @return pix: pixel of the image

    """
    im = Image.open(img_path)
    pix = im.load()
    return im.size, pix


def is_valid(img):
    """Check if the image is valid of not
       Return False if np.sum is not valid
    """

    if np.sum(img) is None or np.sum(img) == 0:
        return False
    return True


def is_rgb(img):
    """Check if the image is rgb or gray scale"""

    if len(img.shape) <= 2:
        return False
    if img.shape[2] < 3:
        return False
    return True


def read_and_gray(fimg, size=None, save=False):
    """Read and convert images to gray

    @param fimg: input image file name

    Keyword arguments:
    save      -- True to save the image
    size -- tuple of new size in (height, width)

    @return img

    """
    img = read_img(fimg, size=size)
    if img is None:
        return img
    img_gray = conv_gray(img)
    if save:
        dirname = os.path.dirname(fimg)
        _fname = os.path.basename(fimg).split('.')
        _fname.insert(-1, '_gray.')
        fname = ''.join(_fname)
        fname = os.path.join(dirname, fname)
        save_img(img_gray, fname=fname)
    return img_gray


def conv_color(img, save=False, order="bgr"):
    """Convert the image to gray scale

    @param img: image array

    Keyword arguments:
    save  -- True to save the image
    order -- order in color space (rgb or bgr)

    """
    if order == "bgr":
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif order == "rgb":
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        self.logger.error("Order in color space should be rgb or bgr")
        return False
    if save:
        save_img(color, 'color.png')
    return color


def conv_gray(img, save=False):
    """Convert the image to gray scale

    @param img: image array

    Keyword arguments:
    save -- True to save the image

    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if save:
        save_img(gray, 'gray.png')
    return gray


def intensity(img, save=False):
    """Get the pixel intensity

    @param img: image array

    Keyword arguments:
    save -- True to save the image

    """
    if is_rgb(img):
        intensity = conv_gray(img)
    else:
        intensity = img
    intensity = intensity.astype(float)
    intensity *= (1.0 / intensity.max())
    if save:
        from dyda_utils import plot
        plot.Plot().plot_matrix(intensity, fname='intensity_cm.png',
                                show_text=False, show_axis=False, norm=False)
    return intensity


def save_img(img, fname='cv2.jpg'):
    """Write images

    @param img: image array

    Keyword arguments:
    fname -- save file name

    """
    cv2.imwrite(fname, img)
    return 0


def read_and_flip_for_tinycv(fimg, direction='h', save=False):
    """Flip images

    @param fimg: input image file name or the image matrix

    Keyword arguments:
    direction -- 'h' for horizontal or 'v' for vertical
    save      -- True to save the image

    @return img

    """
    if type(fimg) is str:
        img = read_img(fimg)
        if img is None:
            print('ERROR: Cannot read %s' % fimg)
            return img
    else:
        img = fimg

    code = 1 if direction == 'h' else 0
    img_flip = cv2.flip(img, code)

    fname = ""

    if save:
        if type(fimg) is not str:
            save_img('./flip_' + direction + '.png')
        else:
            dirname = os.path.dirname(fimg)
            _fname = os.path.basename(fimg).split('.')
            _fname.insert(-1, '_flip_' + direction + '.')
            fname = ''.join(_fname)
            fname = os.path.join(dirname, fname)
            save_img(img_flip, fname=fname)

    return img_flip, fname


def read_and_flip(fimg, direction='h', save=False):
    """Flip images

    @param fimg: input image file name

    Keyword arguments:
    direction -- 'h' for horizontal or 'v' for vertical
    save      -- True to save the image

    @return img

    """

    img, fname = read_and_flip_for_tinycv(fimg, direction=direction, save=save)
    return img


def resize_img(img, size=(None, None), force_cpu=True):
    """ Resize the image
    @param img: input image array to be resized

    Keyword arguments:
    size      -- tuple of new size (default None)
    force_cpu -- True to force using CPU

    @return imgs: dictionary of the croped images

    """

    height = img.shape[0]
    width = img.shape[1]
    if size[0] is None and size[1] is None:
        size = (width, height)
    elif size[0] is None:
        size = (int(size[1] / height * width), size[1])
    elif size[1] is None:
        size = (size[0], int(size[0] / width * height))

    if force_cpu:
        return cv2.resize(img, size)

    # define rule to determine if the image size if "large"
    # use gpuwrapper if cuda is installed or force_gpu=True
    large_size = False
    if height > 2000 or width > 2000:
        large_size = True

    if tools.check_cuda() and large_size:
        print("[dyda_utils] Using GPU to resize")
        from dyda_utils.cv2cuda import gpuwrapper
        return gpuwrapper.cudaResizeWrapper(img, size)
    else:
        return cv2.resize(img, size)


def read_and_random_crop(fimg, size=None, ratio=0.7, save=False):
    """Read images and do random crops

    @param fimg: input image file name

    Keyword arguments:
    size -- tuple of new size (default None)
    ratio -- used to determin the croped size (default 0.7)

    @return imgs: dictionary of the croped images

    """
    img = read_img(fimg)
    if img is None:
        return img
    nrow = len(img)
    ncol = len(img[0])
    imgs = {}
    imgs['crop_img_lt'] = img[0:int(nrow * ratio),
                              0:int(ncol * ratio)]
    imgs['crop_img_lb'] = img[int(nrow * (1 - ratio)):nrow,
                              0:int(ncol * ratio)]
    imgs['crop_img_rt'] = img[0:int(nrow * ratio),
                              int(ncol * (1 - ratio)):ncol]
    imgs['crop_img_rb'] = img[int(nrow * (1 - ratio)):nrow,
                              int(ncol * (1 - ratio)):ncol]
    for corner in imgs:
        if size is not None:
            imgs[corner] = resize_img(imgs[corner], size)
        if save:
            dirname = os.path.dirname(fimg)
            dirname = os.path.join(dirname, 'crops')
            tools.check_dir(dirname)

            _fname = os.path.basename(fimg).split('.')
            _fname.insert(-1, '_' + corner + '.')

            fname = ''.join(_fname)
            fname = os.path.join(dirname, fname)
            save_img(imgs[corner], fname=fname)
    return imgs


def read_gif(fimg):
    """Access image pixels

    @param fimg: input image file name

    """
    img = np.array(Image.open(fimg).convert('RGB'))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def read_img(fimg, size=None, log=True):
    """Access image pixels

    @param fimg: input image file name

    Keyword arguments:
    size -- tuple of new size in (height, width)
    log  -- True to print log if the action fails

    """
    if not tools.check_exist(fimg):
        if log:
            print("[IMAGE] Error %s does not exist" % fimg)
        sys.exit(1)

    if tools.check_ext(fimg, 'gif'):
        img = read_gif(fimg)

    else:
        img = cv2.imread(fimg)

    if img is None:
        if log:
            print("[IMAGE] Error reading file %s" % fimg)
        return img

    if size is not None:
        img = resize_img(img, size)

    return img


def auto_padding(img, mode='center', value=127):
    """Padding images
    @param img: image to be padded

    Keyword arguments:
    value  -- padded value
    mode   -- padding mode, three modes are supported
              top-left, center, bottom-right

    """
    ori_h, ori_w = img.shape[:2]
    s = max(ori_h, ori_w)
    if mode == 'top-left':
        padded_img = padding(img, top=0, bottom=(s - ori_h),
                             right=(s - ori_w), left=0, value=value)
    elif mode == 'center':
        h = int((s - ori_h) / 2)
        w = int((s - ori_w) / 2)
        padded_img = padding(img, top=h, bottom=(s - ori_h - h),
                             right=(s - ori_w - w), left=w, value=value)
    elif mode == 'bottom-right':
        padded_img = padding(img, top=(s - ori_h), bottom=0,
                             right=0, left=(s - ori_w), value=value)

    return padded_img


def padding(img, top=0, bottom=0, right=0, left=0, value=127):
    """Padding images
    @param img: image to be padded

    Keyword arguments:
    top    -- padding size to the top border
    bottom -- padding size to the bottom border
    right  -- padding size to the right border
    left   -- padding size to the left border
    value  -- padded value

    """

    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=[value, value, value])
    return padded_img


def get_jpeg_quality(img_path):
    """Get the jpeg quality using identify tool"""

    try:
        q = subprocess.check_output("identify -verbose %s | grep Quality"
                                    % img_path, shell=True)
        q = q.replace(' ', '').split('\n')[0].split(':')[1]
        return int(q)
    except subprocess.CalledProcessError:
        return None


def select(img, imin, imax, default=0, inv=False):
    """Select only values in a given range
       and apply default value to the rest

    @param img: image array
    @param imin: lower limit
    @param imax: upper limit

    Keyword arguments:
    default -- the default value to be applied (default: 0)
    inv     -- invert the selection, to select values NOT
               in the region (default: False)

    """
    if inv:
        cp_img = np.where(img < imax and img > imin, default, img)
    else:
        cp_img = np.where(img > imax, default, img)
        cp_img = np.where(cp_img < imin, default, cp_img)
    return cp_img


def satuation(img, save=False):
    """Get the image satuation

    @param img: image array

    Keyword arguments:
    save  -- True to save the image (default: False)

    """
    if not is_rgb(img):
        print('ERROR: Cannot support grayscale images')
        sys.exit(0)
    np.seterr(divide='ignore')
    sat = 1 - np.divide(3, (img.sum(axis=2) * img.min(axis=2)))
    sat[np.isneginf(sat)] = 0
    if save:
        from dyda_utils import plot
        plot.Plot().plot_matrix(sat, fname='sat_cm.png',
                                show_text=False, show_axis=False, norm=False)
    return sat


def find_boundary(img, thre=0, findmax=True):
    mean = np.array(img).mean(axis=1)
    selected = [i for i in range(0, len(mean)) if mean[i] > thre]
    start = selected[0]
    end = selected[-1]
    if findmax:
        return max(start, len(img) - end)
    return min(start, len(img) - end)


def create_blank_img(height=480, width=640, color=(0, 0, 0)):
    """ Create a blank image for testing """

    blank_image = np.zeros((height, width, 3), np.uint8)
    blank_image[:, :] = color
    return blank_image


def is_black(img_array):
    """ Detect if the input image array is black of not """
    if np.sum(img_array) == 0:
        return True
    else:
        return False


def crop_black_bars(img, fname=None, thre=1):
    """Crop symmetric black bars"""

    if is_rgb:
        _gray = conv_gray(img)
    else:
        _gray = img
    cut1 = find_boundary(_gray, thre=thre)
    cut2 = find_boundary(_gray.T, thre=thre)

    if cut1 > 0:
        img = img[cut1:-cut1]
    if cut2 > 0:
        img = img[:, cut2:-cut2]

    if fname is not None:
        logging.info('Saving croped file as %s.' % fname)
        save_img(img, fname)

    return img


def laplacian(img, save=False):
    """Laplacian transformation"""

    if is_rgb(img):
        img = conv_gray(img)
    la = cv2.Laplacian(img, cv2.CV_64F)
    if save:
        save_img(la, 'laplacian.png')
    return la


def cal_side_means(img, thre=0.15):
    """Calculate the mean of four sides"""

    upper = int(thre * img.shape[0])
    lower = int((1 - thre) * img.shape[0])
    up = np.mean(img[:upper])
    down = np.mean(img[lower:])
    left = np.mean(img[:upper, :thre * img.shape[1]]) + \
        np.mean(img[lower:, :thre * img.shape[1]])
    left /= 2
    right = np.mean(img[:upper, (1 - thre) * img.shape[1]:]) + \
        np.mean(img[lower:, (1 - thre) * img.shape[1]:])
    right /= 2
    logging.debug("up: %.2f, down: %.2f, left: %.2f, right: %.2f"
                  % (up, down, left, right))

    return (up + down + left + right) / 4.0


def sobel(img, axis=0, save=False):
    """Sobel transformation"""

    if axis == 0:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    elif axis == 1:
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    if save:
        save_img(sobel, 'sobel.png')
    return sobel


def LBP(img, save=False, parms=None, subtract=False, method='uniform'):
    """Get the LBP image
    (reference: http://goo.gl/aeADZd)

    @param img: image array

    Keyword arguments:
    save    -- True to save the image
    parms     -- [points, radius] (default: None)
    subtract -- True to subtract values to pts (default: False)

    """
    from skimage.feature import local_binary_pattern

    if is_rgb(img):
        img = conv_gray(img)
    if parms is None:
        pts = int(img.shape[0] * img.shape[1] * 0.0003)
        radius = min(img.shape[0], img.shape[1]) * 0.015
    else:
        pts = parms[0]
        radius = parms[1]
    lbp = local_binary_pattern(img, pts, radius, method=method)
    if subtract:
        lbp = np.abs(lbp - pts)
    if save:
        from dyda_utils import plot
        plot.Plot().plot_matrix(lbp, fname='lbp_cm.png', show_text=False,
                                show_axis=False, norm=False)
    return lbp


def substract_bkg_files(fimg, fbkgs, fname=None):
    """Substract image background

    @param img: file name of the input forward image
    @param bkgs: a list of file names of the background images

    Keyword arguments:
    fname -- specify to output the substracted image

    """

    bkgs = []
    for fbkg in fbkgs:
        bkgs.append(read_img(fbkg))
    img = read_img(fimg)
    return substract_bkg(img, bkgs, fname=fname)


def substract_bkg(img, bkgs, fname=None):
    """Substract image background

    @param img: input forward image in np array
    @param bkgs: a list of background image in np arrays

    Keyword arguments:
    fname -- specify to output the substracted image

    """

    backsub = cv2.BackgroundSubtractorMOG2()
    fgmask = None
    for bkg in bkgs:
        fgmask = backsub.apply(bkg)
    fgmask = backsub.apply(img)
    if fname is not None and type(fname) is str:
        save_img(fgmask, fname=fname)
    return cv2.bitwise_and(img, img, mask=fgmask)


def get_houghlines(img):
    """Get lines from hough transform"""

    if is_rgb(img):
        img = conv_gray(img)

    edges = cv2.Canny(img, 100, 200)
    return cv2.HoughLines(edges, 1, np.pi / 180, 200)


def draw_houghlines(img, lines, save=False):
    """Draw lines found by hough transform"""

    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if save:
        save_img(img, 'houghlines.png')
    return lines


def morph_opening(img, hr=0.05, wr=0.1, save=False):
    """Apply Morphological opening transform

    @param img: image array

    Keyword arguments:
    hr   -- ratio to the height, for closing window (default: 0.1)
    wr   -- ratio to the width, for closing window (default: 0.2)
    save -- True to save the image

    """
    h = int(img.shape[0] * hr)
    w = int(img.shape[1] * wr)
    kernel = np.ones((h, w), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    if save:
        from dyda_utils import plot
        plot.Plot().plot_matrix(opening, fname='opening_cm.png',
                                show_text=False, show_axis=False, norm=False)
    return opening


def morph_dilation(img, rs=0.01, save=False):
    """Apply Morphological dilation transform

    @param img: image array

    Keyword arguments:
    shape -- width of the kernel
    save -- True to save the image

    """
    shape = int(min(img.shape[0], img.shape[1]) * rs)
    kernel = np.ones((shape, shape), np.uint8)
    dil = cv2.dilate(img, kernel, iterations=1)
    if save:
        from dyda_utils import plot
        plot.Plot().plot_matrix(dil, fname='dil_cm.png',
                                show_text=False, show_axis=False, norm=False)
    return dil


def morph_closing(img, hr=0.1, wr=0.2, save=False):
    """Apply Morphological closing transform

    @param img: image array

    Keyword arguments:
    hr   -- ratio to the height, for closing window (default: 0.1)
    wr   -- ratio to the width, for closing window (default: 0.2)
    save -- True to save the image

    """
    h = int(img.shape[0] * hr)
    w = int(img.shape[1] * wr)
    kernel = np.ones((h, w), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    if save:
        from dyda_utils import plot
        plot.Plot().plot_matrix(closing, fname='closing_cm.png',
                                show_text=False, show_axis=False, norm=False)
    return closing


def max_s(img, save=False):
    """Get maxS, more details see http://goo.gl/d3GQ3T

    @param img: image array

    Keyword arguments:
    save     -- True to save the image (default: False)

    """
    intensity = intensity(img, save=save)
    max_s = np.where(intensity > 0.5, 2 * (0.5 - intensity), 2 * intensity)
    if save:
        from dyda_utils import plot
        plot.Plot().plot_matrix(max_s, fname='max_s_cm.png', show_text=False,
                                show_axis=False, norm=False)
    return max_s


def tilde_s(img, save=False, nan_to_num=True):
    """Get tilde S, more details see http://goo.gl/d3GQ3T

    @param img: image array

    Keyword arguments:
    save     -- True to save the image (default: False)
    nan_to_num -- True to convert inf to numbers (default: True)

    """
    sat = satuation(img, save=save)
    max_s = max_s(img, save=save)
    tilde_s = sat / max_s
    if nan_to_num:
        tilde_s = np.nan_to_num(tilde_s)
    if save:
        save_img(tilde_s, 'tilde_s_cm.png')
    return tilde_s


def cal_d(diff_tilde_s, diff_int, left=True):
    """Get D, more details see http://goo.gl/d3GQ3T

    @param diff_tilde_s: difference of tilde S matrix
    @param diff_int: difference of the intensity matrix

    Keyword arguments:
    left  -- True to make D_L, False for D_R

    """
    if left:
        tilde_s = np.insert(diff_tilde_s, 0,
                            diff_tilde_s.T[0], axis=1)
        intensity = np.insert(diff_int, 0,
                              diff_int.T[0], axis=1)

    else:
        tilde_s = np.insert(diff_tilde_s, diff_tilde_s.shape[1],
                            diff_tilde_s.T[-1], axis=1)
        intensity = np.insert(diff_int, diff_int.shape[1],
                              diff_int.T[-1], axis=1)
    return (1 + tilde_s) * intensity


def overlay_text(img, save=False, T_H=1):
    """Get D, more details see http://goo.gl/d3GQ3T

    @param img: image array

    Keyword arguments:
    T_H   -- threshold used for the transition map
    save -- True to save the image

    """
    tilde_s = tilde_s(img, save=save)
    intensity = intensity(img, save=save)
    diff_tilde_s = np.diff(tilde_s)
    diff_int = np.absolute(np.diff(intensity))
    D_L = cal_d(diff_tilde_s, diff_int) + 1
    D_R = cal_d(diff_tilde_s, diff_int, left=False)
    T = np.where(D_R > D_L, 1, 0)
    if save:
        from dyda_utils import plot
        plot.Plot().plot_matrix(T, fname='T_cm.png', show_text=False,
                                show_axis=False, norm=False)
    return T


def linked_map_boundary(img, save=False, T_H=1, r=0.04):
    """Get linked_map_boundary

    @param img: image array

    Keyword arguments:
    r      -- ratio for setting threshold (default: 0.04)
    T_H    -- threshold used for the transition map
              (used by overlay_text)
    save -- True to save the image

    """
    T = overlay_text(img, save=save, T_H=T_H)
    thre = int(T.shape[1] * r)
    for rth in range(0, T.shape[0]):
        non_zero = np.nonzero(T[rth])[0]
        for i in range(0, len(non_zero) - 1):
            s = non_zero[i]
            e = non_zero[i + 1]
            if e - s < thre:
                T[rth][s:e + 1] = 255
    if save:
        save_img(T, 'lmb.png')
    return T


def contours(img, save=False):
    """Get contours"""

    if is_rgb(img):
        img = conv_gray(img)
    contours, hier = cv2.findContours(img, cv2.RETR_LIST,
                                      cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contours(img, contours, amin=-1, amax=-1,
                  save=False, rect=False, whratio=-1.0,
                  color=(255, 0, 255), width=2, bcut=0.3, bwidth=0.1):
    """Draw contours

    @param img: input image array
    @param contours: contours to be drawn

    Keyword arguments:
    amin   -- min of the area to be selected
    amax   -- max of the area to be selected
    rect   -- True to draw boundingRec (default: False)
    save   -- True to save the image (default: False)
    color  -- Line color (default: (255, 0, 255))
    width  -- Line width, -1 to fill (default: 2)
    bwidth -- boundary selection width, set it to 0 if no boundary
              selection should be applied (default: 0.1)
    bcut   -- boundary selection ratio, set it to 0 if no boundary
              selection should be applied (default: 0.3)

    """
    areas = []
    [h0, w0] = img.shape
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if amin > 0 and area < amin:
            continue
        if amax > 0 and area > amax:
            continue
        if rect:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if whratio > 0:
                if w / h < whratio and h / w < whratio:
                    continue
            if bcut > 0 and bwidth > 0:
                if w > h:
                    if (y <= h0 * bcut and y + h >= h0 * (bcut + bwidth)):
                        continue
                    if (y + h >= h0 * (1 - bcut) and y <=
                            h0 * (1 - (bcut + bwidth))):
                        continue
                    if (y >= h0 * bcut and y + h <= h0 * (1 - bcut)):
                        continue
                if h > w:
                    if (w <= w0 * bcut and x + w >= w0 * (bcut + bwidth)):
                        continue
                    if (x + w >= w0 * (1 - bcut) and x <=
                            w0 * (1 - (bcut + bwidth))):
                        continue
                    if (x >= w0 * bcut and x + w <= w0 * (1 - bcut)):
                        continue
            cv2.rectangle(img, (x, y), (x + w, y + h), color, width)
            areas.append([x, y, w, h])

        else:
            cv2.drawContours(img, [cnt], 0, color, width)
            areas.append(area)
    if save:
        save_img(img, 'contours.png')
    return img, areas


def check_cnt_std(img, cnt, thre=0.01):
    """Check if std of contour points is within a threshold"""

    std_x = np.std(cnt.T[0][0])
    w = img.shape[1]
    std_y = np.std(cnt.T[1][0])
    h = img.shape[0]
    if std_x <= thre * w or std_y <= thre * h:
        return False
    return True


def mask_image(input_img, top=0, bottom=0, left=0, right=0,
               mask_color=0, save=False, dup_new=False):
    """Mask a rect region in image with a given mask_color value"""

    if dup_new:
        img = copy.deepcopy(input_img)
    else:
        img = input_img
    if len(img.shape) == 2:
        img[top:bottom, left:right] = mask_color
    elif (len(img.shape) == 3):
        img[top:bottom, left:right, :] = mask_color
    if save:
        save_img(img, fname="masked_img.png")
    return img


def mask_image_rect(input_img, rect, save=False, dup_new=False, mask_color=0):
    """Mask a rect in image with a given mask_color value
       @param input_img: input image
       @param rect: input rect object to specify the boundary

       Keyword arguments:
       save       - True to save the masked image
       dup_new    - True to deepcopy the input_img
       mask_color - Mask value, default=0
    """

    img = mask_image(input_img, top=rect.t, bottom=rect.b,
                     left=rect.l, right=rect.r, mask_color=mask_color,
                     save=save, dup_new=dup_new)
    return img
