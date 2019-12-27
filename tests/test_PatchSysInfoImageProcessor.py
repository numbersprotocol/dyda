import unittest
from dyda_utils import image
from dyda_utils import payload
from dyda.components.image_processor import PatchSysInfoImageProcessor


class TestPatchSysInfoImageProcessor(unittest.TestCase):
    """ Test case of vertical and counterclcokwise. """

    def test_main_process(self):
        """ Main process of unit test. """

        img = image.create_blank_img(width=500, height=500)
        patcher = PatchSysInfoImageProcessor()
        patcher.patch_meta_roi = True
        patcher.keys_to_patch = ["counter"]
        patcher.external_metadata = {
            "roi": [
                    {
                        "top": 0,
                        "right": 400,
                        "left": 240,
                        "overlap_threshold": 0.5,
                        "bottom": 316
                    }
                ]
        }
        results = {"counter": 10}
        patcher.input_data = [img, results]
        patcher.run()

        output_img = image.resize_img(patcher.output_data, size=(10, 10))
        jpgbytes = payload.encode_np_array(output_img)
        jpgstr = payload.stringify_jpg(jpgbytes)

        ref_data_str = (
            "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEB"
            "AMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAg"
            "ICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgo"
            "KCgoKCgoKCgoKCgr/wAARCAAKAAoDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAA"
            "AAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE"
            "1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRk"
            "dISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKW"
            "mp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3"
            "+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDB"
            "AcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNO"
            "El8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl"
            "6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU"
            "1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD8ATLKYhAZGKKxZ"
            "UJ4BOMnHqcD8hTa6SzsbFvCk9y1nEZBpSuJDGNwb7VIuc+uABn0GKwtVRI9UuY40C"
            "qs7hVAwANx4r3s1ymeBwlDESqc3tIQltsnzJK93eyhptppZWMoT5pNW2Z//9k="
        )

        self.assertEqual(jpgstr, ref_data_str)

if __name__ == '__main__':
    unittest.main()
