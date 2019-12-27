import os
import subprocess
from dyda.core import tool_base


class TFCkptToPbTool(tool_base.ToolBase):
    """ depends on tensorflow 1.12.0

        use tensorflow object_detection API to export ckpt file to
        pb file

        input: {'ckpt_dir': $DIR_OF_CKPT,
                'config_path': $PATH_OF_CONFIG}

        $DIR_OF_CKPT: directory which contains .ckpt files

        $PATH_OF_CONFIG: the path of .config, you should use the same
                         config file as training the model

        this component usually uses the results of LearnerTFDetector
        as input, and it would automatically set things fine

        example usage: /dyda/pipeline/configs/learner_mobilenet_ssd.config

        note: no matter snapshot or not, this componet will automatically
              make a folder named 'model' under snapshot_folder, and
              put intermediates like frozen_inference_graph.pb in it.
    """

    def __init__(self, dyda_config_path=''):
        """ __init__ of TFCkptToPbTool """

        super(TFCkptToPbTool, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

    def main_process(self):
        """ main process """
        self.check_snapshot_folder()
        ckpt_dir = self.input_data['ckpt_dir']

        model_dir = os.path.join(self.snapshot_folder,
                                 'model')

        if not os.path.exists(model_dir):
            print("[dyda_utils] INFO: Creating %s" % model_dir)
            os.makedirs(model_dir)

        # search and use the last checkpoint in ckpt_dir
        files = []
        for (dirpath, dirnames, filenames) in os.walk(ckpt_dir):
            files.extend(filenames)
            break
        number_of_checkpoint = 0
        for f in files:
            filename, file_extension = os.path.splitext(f)
            if file_extension == '.index':
                number = int(filename.split('-')[1])
                number_of_checkpoint = max(number_of_checkpoint, number)

        ckpt = "model.ckpt-" + str(number_of_checkpoint)

        # run tensorflow object_detection API to export ckpt file
        # to pb file
        # note: you should set $MODEL in terminal, $MODEL is
        #       the path of https://github.com/tensorflow/models,
        #       you clone at where of local
        cmd = ("python3 $MODEL/research/object_detection/" +
               "export_inference_graph.py " +
               "--input_type image_tensor " +
               "--pipeline_config_path " + self.input_data['config_path'] +
               " --trained_checkpoint_prefix " +
               os.path.join(ckpt_dir, ckpt) +
               " --output_directory " + model_dir)
        self.logger.info("Running %s " % cmd)
        output = subprocess.check_output(["bash", "-c", cmd])
