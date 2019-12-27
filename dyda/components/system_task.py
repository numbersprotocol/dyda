import os
import tempfile
import shutil
from dyda_utils import tools
from dyda.core import system_task_base


class CreateSymbolicLinkTask(system_task_base.SystemTaskBase):
    """ Generate training data from labels and image paths """

    def __init__(self, dyda_config_path='', param=None):
        super(CreateSymbolicLinkTask, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.results = ""

    def main_process(self):
        """ main_process of CreateSymbolicLinkTask """

        if "output_folder" in self.param.keys():
            tmpdir = self.param["output_folder"]
        else:
            tmpdir = tempfile.mkdtemp()
        self.results = tmpdir
        tools.check_dir(tmpdir)
        self.logger.warning("Training data is linked to %s" % tmpdir)

        image_paths = self.input_data[0]
        labels = self.input_data[1]

        if len(image_paths) != len(labels):
            self.terminate_flag = True
            self.logger.error("Lengths of labels and data do not match")
            return False

        for label in set(labels):
            if not isinstance(label, str):
                label = str(label)
            label_folder = os.path.join(tmpdir, label)
            tools.check_dir(label_folder)

        for i in range(0, len(image_paths)):
            label = labels[i]
            if not isinstance(label, str):
                label = str(label)
            input_path = image_paths[i]
            fname = os.path.basename(input_path)
            output_path = os.path.join(tmpdir, label, fname)
            os.symlink(input_path, output_path)

        return True

    def reset_results(self):
        """ reset_results for CreateSymbolicLinkTask"""
        self.results = ""


class RemoveFolder(system_task_base.SystemTaskBase):
    """ Remove a given folder """

    def __init__(self, dyda_config_path=''):
        """ __init__ of RemoveFolder """

        super(RemoveFolder, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

    def main_process(self):
        """ define main_process of dyda component """

        if isinstance(self.input_data, str):
            self.remove(self.input_data)
        elif isinstance(self.input_data, list):
            for directory in self.input_data:
                self.remove(directory)
        if "add_folder_to_rm" in self.param.keys():
            for directory in self.param["add_folder_to_rm"]:
                self.remove(directory)
        return True

    def remove(self, folder_to_rm):
        """ This can only removes the folder under /tmp """
        if folder_to_rm[:4] != "/tmp":
            self.logger.warning("%s is not under /tmp, pass" % folder_to_rm)
            return
        if tools.check_exist(folder_to_rm):
            self.logger.warning("Removing %s" % folder_to_rm)
            shutil.rmtree(folder_to_rm)
        else:
            self.logger.info("%s does not exist, pass" % folder_to_rm)

    def reset_results(self):
        self.results = {}
