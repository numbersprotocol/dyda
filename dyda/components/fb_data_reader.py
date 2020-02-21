import os
import re
from fbjson2table.func_lib import parse_fb_json
from dyda.core import data_reader_base


class FbYourPostsJsonReader(data_reader_base.DataReaderBase):
    """Read in your_posts*.json from the facebook data folder.

        input: PATH_OF_FOLDER, or LIST_OF_PATH_OF_FOLDER

        output: JSON_LIKE_DICT, or LIST_OF_JSON_LIKE_DICT

    """

    def __init__(self, dyda_config_path="", param=None):
        super(FbYourPostsJsonReader, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function called by the external code """
        self.uniform_input()
        self.reset_output()

        for input_path in self.input_data:
            self.output_data.append(
                self.read_your_posts_json(input_path))
        self.uniform_output()

    def read_your_posts_json(self, path):

        posts_json_list = []
        for root, dirs, files in os.walk(path, topdown=True):
            for f in files:
                filepath = os.path.join(root, f)
                if (os.path.splitext(f)[1] == '.json') & \
                   ("posts" in root) & \
                        bool(re.match('your_posts*', f)):
                    temp_dict = {
                        "filename": os.path.split(root)[-1] + '__' + f,
                        "filecontent": parse_fb_json(filepath)}
                    posts_json_list.append(temp_dict)
        return posts_json_list
