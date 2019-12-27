import os
import sys
import subprocess
import tempfile
from shutil import copyfile
from dyda_utils import tools
from dyda_utils import image
from dyda.core import learner_base


class LearnerYOLO(learner_base.LearnerBase):
    """ Use darknet to retrain a yolo model """

    def __init__(self, dyda_config_path=''):
        """ __init__ of LearnerTFClassifier """

        super(LearnerYOLO, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)
        self.classes = self.param["labels"]
        self.overwrite = True
        if "overwrite" in self.param.keys():
            self.overwrite = self.param["overwrite"]
        if "output_folder" in self.param.keys():
            tmpdir = self.param["output_folder"]
        else:
            tmpdir = tempfile.mkdtemp()
        self.output_file = self.param["output_path"]
        self.label_folder = os.path.join(tmpdir, "labels")
        self.img_folder = os.path.join(tmpdir, "JPEGImages")
        for folder in [tmpdir, self.label_folder, self.img_folder]:
            tools.check_dir(folder)
        self.darknet_path = self.param["darknet_path"]

    def convert(self, size, X, Y):
        """ convert size, X, Y to YOLO format """

        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (X[0] + X[1]) / 2.0
        y = (Y[0] + Y[1]) / 2.0
        w = X[1] - X[0]
        h = Y[1] - Y[0]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    def main_process(self):
        """ main process """

        if not os.path.isdir(self.input_data):
            self.logger.error("%s is not a valid folder" % self.input_data)
            self.terminate_flag = True

        if self.overwrite:
            output = open(self.output_file, 'w')

        else:
            if tools.check_exist(self.output_file):
                output = open(self.output_file, 'a')
            else:
                output = open(self.output_file, 'w')
        print("LearnerYOLO: creating %s" % self.output_file)

        check_keys = ["folder", "filename", "annotations"]
        for json_file in tools.find_files(self.input_data, walkin=False):
            try:
                json_content = tools.parse_json(json_file)
            except BaseException:
                self.logger.error("Fail to open %s" % json_file)
                continue
            for key in check_keys:
                if key not in json_content.keys():
                    self.logger.error(
                        "%s is not found in %s" % (key, json_file)
                    )
                    continue
            folder = json_content["folder"]
            filename = json_content["filename"]
            # FIXME
            # folder = folder.replace("results", "labeled_data")
            # folder = folder.replace("_tmp", "")

            in_img_path = os.path.join(folder, filename)
            out_img_path = os.path.join(self.img_folder, filename)
            o_file_path = os.path.join(
                self.label_folder, tools.remove_extension(filename) + '.txt'
            )
            o_file = open(o_file_path, 'w')

            annos = json_content["annotations"]
            size, pix = image.get_img_info(in_img_path)

            h = float(size[0])
            w = float(size[1])

            for anno in annos:
                X = []
                Y = []
                cls = anno["label"]
                if cls not in self.classes:
                    self.logger.debug("%s is not in the selected class" % cls)
                    continue
                cls_id = self.classes.index(cls)
                X = [anno["left"], anno["right"]]
                Y = [anno["top"], anno["bottom"]]
                bb = self.convert((w, h), X, Y)
                o_file.write(
                    str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
                )

            o_file.close()
            self.logger.info("Link %s to %s" % (in_img_path, out_img_path))
            os.symlink(in_img_path, out_img_path)

            output.write(out_img_path + '\n')
        output.close()

        # FIXME darknet env has to be well prepared and fix the classes now
        train_path = os.path.join(self.darknet_path, "train.txt")
        if train_path != self.output_file:
            copyfile(self.output_file, train_path)
        os.chdir(self.darknet_path)
        cmd = ("./darknet detector train cfg/dt42.data"
               " cfg/tiny-yolo-voc-dt42.cfg darknet.weights.13 -gpus 1")
        self.logger.info("Running %s " % cmd)
        output = subprocess.check_output(["bash", "-c", cmd])
        self.results = {
            "root_directory": self.darknet_path,
            "weight_file": "backup_dt42/yolo-voc-dt42_final.weights",
            "data_file": "cfg/dt42.data",
            "names_file": "data/dt42.names",
            "cfg_file": "cfg/tiny-yolo-voc-dt42.cfg"
        }


class LearnerTFClassifier(learner_base.LearnerBase):
    """ Use retrain.py to retrain a model """

    def __init__(self, dyda_config_path=''):
        """ __init__ of LearnerTFClassifier """

        super(LearnerTFClassifier, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

        self.reset_results()
        self.weight_files = ["output_graph.pb", "output_labels.txt"]

    def reset_results(self):
        self.results = {"labels": []}

    def main_process(self):
        """ define main_process of dyda component """

        # input_data should be full path of the data directory
        if not os.path.isdir(self.input_data):
            self.logger.error("%s is not a valid folder" % self.input_data)
            self.terminate_flag = True
        labels = tools.find_folders(self.input_data)
        self.results["labels"] = labels
        self.logger.info("Training for labels: %s." % ' '.join(labels))
        cmd = "python3 " + self.param["retrain_script_path"]
        cmd = cmd + " --learning_rate=" + str(self.param["learning_rate"])
        cmd = cmd + " --testing_percentage=" + str(self.param["test_perc"])
        cmd = cmd + " --validation_percentage=" + str(self.param["val_perc"])
        cmd = cmd + " --train_batch_size=" + str(self.param["train_batch"])
        cmd = cmd + " --validation_batch_size=" + str(self.param["val_batch"])
        if self.param["aug_lip_left_right"]:
            cmd = cmd + " --flip_left_right True"
        if "aug_random_brightness" in self.param.keys():
            cmd = cmd + " --random_brightness=" + \
                str(self.param["aug_random_brightness"])
        if "aug_random_scale" in self.param.keys():
            cmd = cmd + " --random_scale=" + \
                str(self.param["aug_random_scale"])
        cmd = cmd + " --eval_step_interval=" + str(self.param["eval_step"])
        cmd = cmd + " --how_many_training_steps=" + \
            str(self.param["train_steps"])
        cmd = cmd + " --architecture=" + str(self.param["architecture"])
        cmd = cmd + " --image_dir " + self.input_data
        self.logger.info("Running %s " % cmd)
        output = subprocess.check_output(["bash", "-c", cmd])

        # FIXME: get results directly from retrain.py
        training_results = tools.parse_json("./output.json")
        self.results["training_results"] = training_results
        self.cp_weights()

    def cp_weights(self):
        """ copy weight files to snapshot_folder """

        print("Copying weights...")
        cp_folder = os.path.join(self.snapshot_folder, "weights")
        tools.check_dir(cp_folder)
        for fname in self.weight_files:
            ori_path = os.path.join("/tmp", fname)
            if tools.check_exist(ori_path, log=False):
                destination = os.path.join(cp_folder, fname)
                try:
                    copyfile(ori_path, destination)
                except BaseException:
                    self.logger.error("Fail to copy %s" % ori_path)
            else:
                self.logger.error(
                    "%s does not exist, exit dyda." % ori_path
                )
                sys.exit(0)

        # FIXME: if there are more TF classifiers in the future
        classifier = "ClassifierInceptionv3"
        if self.param["architecture"].find('mobilenet') >= 0:
            classifier = "ClassifierMobileNet"
        if classifier in self.config.keys():
            new_config = self.config[classifier]
            new_config["model_file"] = os.path.join(
                cp_folder, "output_graph.pb"
            )
            new_config["label_file"] = os.path.join(
                cp_folder, "output_labels.txt"
            )
            new_config_path = os.path.join(cp_folder, "dyda.config.learner")
            tools.write_json({classifier: new_config}, new_config_path)


class LearnerTFDetector(learner_base.LearnerBase):
    """ depends on tensorflow 1.12.0

        use tensorflow object_detection API to retrain
        detection model

        @param model_path: the path of tf detection used to retrain
                           the model name is like 'model.ckpt(-$NUMBER)',
                           and usually 'model.ckpt' if we use the model
                           that others trained.

               config_path: the path of .config file, usually named
                            'pipeline.config' and contained in pretrain
                            model folder.

               num_clones: how many clones to deploy to deploy per worker.
                           usually used to control how many CPU/GPU we
                           want to use.

        input: {'train_record': $PATH_OF_TRAIN_RECORD,
                'eval_record': $PATH_OF_EVAL_RECORD,
                'label_map': $PATH_OF_LABEL_MAP',
                'n_classes': $NUMBER_OF_CLASSES}

        $PATH_OF_TRAIN_RECORD: the path of tfrecord used to train
                               model.
        $PATH_OF_TRAIN_RECORD: the path of tfrecord used to evaluation
                               model.
        $PATH_OF_LABEL_MAP: the path of label map.
        $NUMBER_OF_CLASSES: the amount of classes

        this component usually uses the results of LabToTFRecordConverter
        as input, and it would automatically set things fine

        example usage: /dyda/pipeline/configs/learner_mobilenet_ssd.config

        note: no matter snapshot or not, this componet will automatically
              make a folder named 'training' under snapshot_folder, and
              put intermediates like model.ckpt in it.
    """

    def __init__(self, dyda_config_path=''):
        """ __init__ of LearnerTFDetector """

        super(LearnerTFDetector, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)
        self.model_path = self.param['model_path']
        self.config_path = self.param['config_path']
        self.num_clones = 1
        if "num_clones" in self.param.keys():
            self.num_clones = self.param['num_clones']

    def main_process(self):
        """ main process """

        self.check_snapshot_folder()
        training_dir = os.path.join(self.snapshot_folder,
                                    'training')

        if not os.path.exists(training_dir):
            print("[dt42lab] INFO: Creating %s" % training_dir)
            os.makedirs(training_dir)

        num_classes = self.input_data['n_classes']
        model_ckpt_path = self.model_path
        lines_to_write = []

        # auto generate the config file that with correct file path
        # that we feed in self.input and dyda_config
        with open(self.config_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if 'num_classes:' in line:
                    lines_to_write.append('    num_classes: ' +
                                          str(num_classes) + '\n')
                elif 'fine_tune_checkpoint:' in line:
                    lines_to_write.append('  fine_tune_checkpoint: ' +
                                          '"' + model_ckpt_path + '"' + '\n')
                elif 'input_path:' in line:
                    if 'train_input_reader' in lines_to_write[i - 2]:
                        lines_to_write.append(
                            '    input_path: ' + '"' +
                            self.input_data['train_record'] + '"' + '\n')
                    elif 'eval_input_reader' in lines_to_write[i - 2]:
                        lines_to_write.append(
                            '    input_path: ' + '"' +
                            self.input_data['eval_record'] + '"' + '\n')
                elif 'label_map_path:' in line:
                    lines_to_write.append(
                        '  label_map_path: ' +
                        '"' +
                        self.input_data['label_map'] +
                        '"' +
                        '\n')

                else:
                    lines_to_write.append(line)
        tf_pipeline_config_path = os.path.join(training_dir,
                                               'configured_pipeline.config')
        with open(tf_pipeline_config_path, 'w') as f:
            f.writelines(lines_to_write)

        # run tensorflow object_detection API to retrain model
        # note: you should set $MODEL in terminal, $MODEL is
        #       the path of https://github.com/tensorflow/models,
        #       you clone at where of local
        cmd = ("python3 $MODEL/research/object_detection/legacy/train.py "
               "--logtostderr "
               "--num_clones=" + str(self.num_clones) + " "
               "--train_dir=" + training_dir + " "
               "--pipeline_config_path=" + tf_pipeline_config_path)
        self.logger.info("Running %s " % cmd)
        output = subprocess.check_output(["bash", "-c", cmd])
        self.results = {
            "ckpt_dir": training_dir,
            "config_path": tf_pipeline_config_path
        }
