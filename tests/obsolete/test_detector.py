import unittest
from dyda_utils import lab_tools
from dyda_utils import image
from dyda.components.yolo_detector import DetectorYOLO
from dyda.components.detector import FaceDetector
from dt42lab.utility import dict_comparator


class TestDetectorYOLO(unittest.TestCase):
    def test_main_process(self):

        # pull test data from gitlab
        config_url = ("https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/"
                      "dbd906035055b99da05308aa927b9f4a/DetectorYOLO.config")
        dyda_config = lab_tools.pull_json_from_gitlab(config_url)

        ref_url = ("https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/"
                   "6ce02b726d96edaf4cb4e4108fb9ba9f/DetectorYOLO_output.json")
        ref_data = lab_tools.pull_json_from_gitlab(ref_url)

        detector_yolo = DetectorYOLO(dyda_config_path=dyda_config)

        input_path = '/home/shared/DT42/test_data/test_detector/dog.jpg'
        detector_yolo.input_data = [image.read_img(input_path)]
        detector_yolo.run()

        tar_data = detector_yolo.results[0]
        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])

        # Since the filename depending on time stamp, it wont match anyway
        for mis_match in report['mismatch_val']:
            for tar_key in mis_match['tar_key']:
                if tar_key == 'filename':
                    continue
                else:
                    self.assertEqual(report['mismatch_val'], [])


class TestFaceDetector(unittest.TestCase):
    def test_main_process(self):

        # test the case with using HOG
        dyda_config = {"FaceDetector": {"model": "hog"}}

        ref_data = {'filename': '', 'size': {'width': 960, 'height': 544},
                    'annotations': [{'left': 414, 'right': 637, 'top': 192,
                                     'labinfo': {}, 'bottom': 415,
                                     'type': 'detection', 'id': 0,
                                     'label': 'face', 'confidence': -1.0}],
                    'folder': ''}

        face_detector = FaceDetector(dyda_config_path=dyda_config)

        input_url = 'https://gitlab.com/DT42/galaxy42/dt42-dyda/uploads/'\
                    'a73f92e9b271420d48f70481f56944c4/2019-03-15-111527.jpg'

        input_data = lab_tools.pull_img_from_gitlab(input_url)
        face_detector.input_data = input_data.copy()
        face_detector.run()

        tar_data = face_detector.results

        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])

        # Since the filename depending on time stamp, it wont match anyway
        for mis_match in report['mismatch_val']:
            for tar_key in mis_match['tar_key']:
                if tar_key == 'filename':
                    continue
                else:
                    self.assertEqual(report['mismatch_val'], [])

        # test the case with list input
        dyda_config = {"FaceDetector": {"model": "hog"}}

        face_detector = FaceDetector(dyda_config_path=dyda_config)

        input_data = lab_tools.pull_img_from_gitlab(input_url)
        face_detector.input_data = [input_data.copy()]
        face_detector.run()

        tar_data = face_detector.results

        report = dict_comparator.get_diff([ref_data], tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])

        # Since the filename depending on time stamp, it wont match anyway
        for mis_match in report['mismatch_val']:
            for tar_key in mis_match['tar_key']:
                if tar_key == 'filename':
                    continue
                else:
                    self.assertEqual(report['mismatch_val'], [])

        # test the case with using CNN
        dyda_config = {"FaceDetector": {"model": "cnn"}}

        ref_data = {'annotations': [{'top': 200, 'confidence': -1.0,
                                     'labinfo': {}, 'type': 'detection',
                                     'right': 627, 'bottom': 404,
                                     'label': 'face', 'left': 424, 'id': 0}],
                    'filename': '', 'folder': '',
                    'size': {'width': 960, 'height': 544}}

        face_detector = FaceDetector(dyda_config_path=dyda_config)

        face_detector.input_data = input_data.copy()
        face_detector.run()

        tar_data = face_detector.results

        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])

        # Since the filename depending on time stamp, it wont match anyway
        for mis_match in report['mismatch_val']:
            for tar_key in mis_match['tar_key']:
                if tar_key == 'filename':
                    continue
                else:
                    self.assertEqual(report['mismatch_val'], [])

        # test the case with specify upsample_times
        dyda_config = {"FaceDetector": {"model": "cnn",
                                           "upsample_times": 3}}

        ref_data = {'folder': '', 'filename': '',
                    'annotations': [{'right': 613, 'top': 165, 'left': 394,
                                     'label': 'face', 'bottom': 384,
                                     'id': 0, 'confidence': -1.0,
                                     'labinfo': {}, 'type': 'detection'}],
                    'size': {'width': 960, 'height': 544}}

        face_detector = FaceDetector(dyda_config_path=dyda_config)

        face_detector.input_data = input_data.copy()
        face_detector.run()
        tar_data = face_detector.results

        report = dict_comparator.get_diff(ref_data, tar_data)
        self.assertEqual(report['extra_field'], [])
        self.assertEqual(report['missing_field'], [])

        # Since the filename depending on time stamp, it wont match anyway
        for mis_match in report['mismatch_val']:
            for tar_key in mis_match['tar_key']:
                if tar_key == 'filename':
                    continue
                else:
                    self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
