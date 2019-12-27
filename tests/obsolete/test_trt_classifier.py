import ctypes
import sys
import unittest
import time

# from pympler import tracker

from dyda_utils import lab_tools
from dyda_utils import image
from dyda.components.trt_classifier import ClassifierTensorRT
from dt42lab.utility import dict_comparator

def test_main_process():

    dyda_config = {
	'ClassifierTensorRT': {
            'model_file': '/home/nvidia/mobilenet_v1_1p0_224.engine',
            'label_file': '/home/nvidia/tf_to_trt_image_classification/data/imagenet_labels_1001.txt'
        }
    }

    classifier_trt = ClassifierTensorRT(dyda_config_path=dyda_config)

    input_path = '/home/nvidia/tf_to_trt_image_classification/data/images/golden_retriever.jpg'
    classifier_trt.input_data = [image.read_img(input_path)]
    # tr = tracker.SummaryTracker()
    for i in range(int(sys.argv[1])):
        print('Run #{}'.format(i))
        s = time.time()
        classifier_trt.run()
        tar_data = classifier_trt.results[0]
        print(tar_data)
        print('classifier_trt.run() takes: {} ms'.format((time.time() - s) * 1000))
        # tr.print_diff()
    # tr.print_diff()


if __name__ == '__main__':
    test_main_process()
