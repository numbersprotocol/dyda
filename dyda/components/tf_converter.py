import os
import subprocess
import tensorflow as tf
from object_detection.utils import dataset_util
from dyda_utils import lab_tools
from dyda_utils import image
from dyda_utils import tools
from dyda.core import data_converter_base


class LabToTFRecordConverter(data_converter_base.ConverterBase):
    """ depends on tensorflow 1.12.0

        use lab-format json files to generate tfrecord files

        @param classes: the list of label we have interest in

        input: {"train_json_dir": $DIR_OF_TRAIN_JSON,
                "eval_json_dir": $DIR_OF_EVAL_JSON
               }

        $DIR_OF_TRAIN_JSON: directory which contains lab-format json
                            used to train model.
        $DIR_OF_EVAL_JSON: directory which contains lab-format json
                            used to evaluation model.

        example usage: /dyda/pipeline/configs/learner_mobilenet_ssd.config

        note: no matter snapshot or not, this componet will automatically
              make a file named 'label_map.pbtxt' and a folder named
              'tfrecords' under snapshot_folder, and put 'train.record',
              'eval.record' in 'tfrecords'.
    """

    def __init__(self, dyda_config_path='', param=None):
        super(LabToTFRecordConverter, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.classes = self.param['classes']

    def main_process(self):

        self.check_snapshot_folder()
        tfrecord_dir = os.path.join(self.snapshot_folder,
                                    'tfrecords')
        if not os.path.exists(tfrecord_dir):
            print("[dyda_utils] INFO: Creating %s" % tfrecord_dir)
            os.makedirs(tfrecord_dir)

        train_dir = self.input_data['train_json_dir']
        if not os.path.isdir(train_dir):
            self.terminate_flag = True
            self.logger.error("{} is not a folder".format(train_dir))

        eval_dir = self.input_data['eval_json_dir']
        if not os.path.isdir(eval_dir):
            self.terminate_flag = True
            self.logger.error("{} is not a folder".format(eval_dir))

        path_train_record = os.path.join(tfrecord_dir, "train.record")
        self.create_tf_record(train_dir, path_train_record)

        path_eval_record = os.path.join(tfrecord_dir, "eval.record")
        self.create_tf_record(eval_dir, path_eval_record)

        # automatically generate label_map.pbtxt
        # note the id should start from 1, 0 is reserved by tensorflow
        path_label_map = os.path.join(self.snapshot_folder,
                                      'label_map.pbtxt')
        with open(path_label_map, 'w') as f:
            lines = []

            for i, item_class in enumerate(self.classes):
                if i != 0:
                    lines.append('\n')

                lines.append('item {\n')
                lines.append('  id: ' +
                             str(self.class_to_int(item_class)) +
                             '\n')
                lines.append('  name: ' +
                             "'" + "{}".format(item_class) + "'" + '\n')
                lines.append('}')
            f.writelines(lines)

        self.results = {"train_record": path_train_record,
                        "eval_record": path_eval_record,
                        "label_map": path_label_map,
                        "n_classes": len(self.classes)}

    def create_tf_record(self, json_dir, tfrecord_path):

        writer = tf.python_io.TFRecordWriter(tfrecord_path)

        check_keys = ["folder", "filename", "annotations"]
        for json_file in tools.find_files(json_dir, walkin=False):
            try:
                json_content = tools.parse_json(json_file)
            except BaseException:
                print("Fail to open %s" % json_file)
                continue
            for key in check_keys:
                if key not in json_content.keys():
                    print(
                        "%s is not found in %s" % (key, json_file)
                    )
                    continue

            tf_example = self.lab_format_to_tf_example(json_content)
            writer.write(tf_example.SerializeToString())
        writer.close()

    def class_to_int(self, text):
        return self.classes.index(text) + 1

    def lab_format_to_tf_example(self, json_content):
        # turn the lab-format directory into the format accepted by tfrecord

        in_img_path = os.path.join(json_content["folder"],
                                   json_content["filename"])

        with tf.gfile.GFile(in_img_path, 'rb') as fid:
            encoded_img = fid.read()

        size, pix = image.get_img_info(in_img_path)
        width = int(size[0])
        height = int(size[1])

        filename = json_content['filename'].encode('utf8')
        _, file_extension = os.path.splitext(json_content['filename'])

        # get image format. e.g. '.jpg'
        file_extension = file_extension.replace('.', '')
        image_format = '{}'.format(file_extension).encode()

        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for anno in json_content['annotations']:
            if anno['label'] not in self.classes:
                continue
            xmins.append(anno['left'] / width)
            xmaxs.append(anno['right'] / width)
            ymins.append(anno['top'] / height)
            ymaxs.append(anno['bottom'] / height)
            classes_text.append(anno['label'].encode('utf8'))
            classes.append(self.class_to_int(anno['label']))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_img),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(
                classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(
                classes),
        }))
        return tf_example
