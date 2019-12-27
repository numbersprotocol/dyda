import cv2
import unittest
from dyda_utils import tools
from dyda_utils import tinycv
from dyda_utils import lab_tools
from dt42lab.utility import dict_comparator
from dyda.components.determinator import DeterminatorMotion


input_data = [{"folder": "",
               "filename": "00000156",
               "size": {"width": 1280, "height": 720},
               "annotations": [{
                   "type": "detection",
                   "id": 0,
                   "label": "car",
                   "confidence": 0.6722624897956848,
                   "right": 500, "top": 100,
                   "left": 300, "bottom": 300,
                   "track_id": 1},
                   {"type": "detection",
                    "id": 1,
                    "label": "car",
                    "confidence": 0.6198570728302002,
                    "right": 700, "top": 200,
                    "left": 500, "bottom": 400,
                    "track_id": 2},
                   {"type": "detection",
                    "id": 1,
                    "label": "car",
                    "confidence": 0.6198570728302002,
                    "right": 800, "top": 300,
                    "left": 700, "bottom": 500,
                    "track_id": 3}
               ]
               },
              {"folder": "",
               "filename": "00000157",
               "size": {"width": 1280, "height": 720},
               "annotations": [{"type": "detection",
                                "id": 1,
                                "label": "car",
                                "confidence": 0.6741642355918884,
                                "right": 600, "top": 200,
                                "left": 400, "bottom": 400,
                                "track_id": 1},
                               {"type": "detection",
                                "id": 3,
                                "label": "car",
                                "confidence": 0.7654070258140564,
                                "right": 500, "top": 0,
                                "left": 300, "bottom": 200,
                                "track_id": 2},
                               {"type": "detection",
                                "id": 4,
                                "label": "car",
                                "confidence": 0.6039748191833496,
                                "right": 400, "top": 100,
                                "left": 300, "bottom": 400,
                                "track_id": 4}
                               ]
               },

              {"folder": "",
               "filename": "00000157",
               "size": {"width": 1280, "height": 720},
               "annotations": [{"type": "detection",
                                "id": 1,
                                "label": "car",
                                "confidence": 0.6741642355918884,
                                "right": 700, "top": 100,
                                "left": 500, "bottom": 300,
                                "track_id": 1},
                               {"type": "detection",
                                "id": 3,
                                "label": "car",
                                "confidence": 0.7654070258140564,
                                "right": 400, "top": 200,
                                "left": 300, "bottom": 400,
                                "track_id": 3},
                               {"type": "detection",
                                "id": 4,
                                "label": "car",
                                "confidence": 0.6039748191833496,
                                "right": 100, "top": 400,
                                "left": 0, "bottom": 700,
                                "track_id": 4}
                               ]
               }

              ]

output_DeterminatorMotion = [{"folder": "",
                              "filename": "00000156",
                              "size": {"width": 1280,
                                       "height": 720},
                              "annotations": [{"type": "detection",
                                               "id": 0,
                                               "label": "car",
                                               "confidence": 0.6722624897956848,
                                               "right": 500,
                                               "top": 100,
                                               "left": 300,
                                               "bottom": 300,
                                               "motion_angle": None,
                                               "motion_distance": None,
                                               "track_id": 1},
                                              {"type": "detection",
                                               "id": 1,
                                               "label": "car",
                                               "confidence": 0.6198570728302002,
                                               "right": 700,
                                               "top": 200,
                                               "left": 500,
                                               "bottom": 400,
                                               "motion_angle": None,
                                               "motion_distance": None,
                                               "track_id": 2},
                                              {"type": "detection",
                                               "id": 1,
                                               "label": "car",
                                               "confidence": 0.6198570728302002,
                                               "right": 800,
                                               "top": 300,
                                               "left": 700,
                                               "bottom": 500,
                                               "motion_angle": None,
                                               "motion_distance": None,
                                               "track_id": 3}]},
                             {"folder": "",
                              "filename": "00000157",
                              "size": {"width": 1280,
                                       "height": 720},
                              "annotations": [{"type": "detection",
                                               "id": 1,
                                               "label": "car",
                                               "confidence": 0.6741642355918884,
                                               "right": 600,
                                               "top": 200,
                                               "left": 400,
                                               "bottom": 400,
                                               "motion_angle": -45.0,
                                               "motion_distance": 100 * (2**0.5),
                                               "track_id": 1},
                                              {"type": "detection",
                                               "id": 3,
                                               "label": "car",
                                               "confidence": 0.7654070258140564,
                                               "right": 500,
                                               "top": 0,
                                               "left": 300,
                                               "bottom": 200,
                                               "motion_angle": 135.0,
                                               "motion_distance": 200 * (2**0.5),
                                               "track_id": 2},
                                              {"type": "detection",
                                               "id": 4,
                                               "label": "car",
                                               "confidence": 0.6039748191833496,
                                               "right": 400,
                                               "top": 100,
                                               "left": 300,
                                               "bottom": 400,
                                               "motion_angle": None,
                                               "motion_distance": None,
                                               "track_id": 4}]},
                             {"folder": "",
                              "filename": "00000157",
                              "size": {"width": 1280,
                                       "height": 720},
                              "annotations": [{"type": "detection",
                                               "id": 1,
                                               "label": "car",
                                               "confidence": 0.6741642355918884,
                                               "right": 700,
                                               "top": 100,
                                               "left": 500,
                                               "bottom": 300,
                                               "motion_angle": 45.0,
                                               "motion_distance": 100 * (2**0.5),
                                               "track_id": 1},
                                              {"type": "detection",
                                               "id": 3,
                                               "label": "car",
                                               "confidence": 0.7654070258140564,
                                               "right": 400,
                                               "top": 200,
                                               "left": 300,
                                               "bottom": 400,
                                               "motion_angle": None,
                                               "motion_distance": None,
                                               "track_id": 3},
                                              {"type": "detection",
                                               "id": 4,
                                               "label": "car",
                                               "confidence": 0.6039748191833496,
                                               "right": 100,
                                               "top": 400,
                                               "left": 0,
                                               "bottom": 700,
                                               "motion_angle": -135.0,
                                               "motion_distance": 300 * (2**0.5),
                                               "track_id": 4}]}]


class TestDeterminatorMotion_simple(unittest.TestCase):
    """ Test simple case. """

    def test_main_process(self):
        """ Main process of unit test. """

        # initialization
        comp = DeterminatorMotion()

        # run component
        for i in range(len(input_data)):
            comp.reset()
            comp.input_data = input_data[i]
            comp.run()

            # compare output_data with reference
            ref_data = output_DeterminatorMotion[i]
            tar_data = comp.results
            report = dict_comparator.get_diff(ref_data, tar_data)
            self.assertEqual(report['extra_field'], [])
            self.assertEqual(report['missing_field'], [])
            self.assertEqual(report['mismatch_val'], [])


if __name__ == '__main__':
    unittest.main()
