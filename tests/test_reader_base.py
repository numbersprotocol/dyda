import unittest

from dyda.core.data_reader_base import DataReaderBase


class TestReaderBase(unittest.TestCase):
    def test_run(self):
        reader_base = DataReaderBase()
        reader_base.run()

    def test_get_metadata(self):
        reader_base = DataReaderBase()
        metadata_input = {}
        reader_base.get_metadata(metadata_input)


if __name__ == '__main__':
    unittest.main()
