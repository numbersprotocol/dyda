import os
import sys
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from dyda.core import data_reader_base
from dt42lab.core import tools
from dt42lab.core import data

import json
def restore_ordered_json(json_path):
    json_data = tools.parse_json(json_path)
    with open(json_path, 'w') as outfile:
        outfile.write(json.dumps(json_data,
                      indent=4, sort_keys=True))

class LabelMeDataReader(data_reader_base.DataReaderBase):

    def __init__(self):
        super(LabelMeDataReader, self).__init__()
        self.save_flag = True

        self.img_path = []
        self.svfolder_path = []
        self.lab_flag = False

        self.query_flag = True

    def get_convert_annotation(self, xml_path):
        in_file = open(xml_path)
        tree = ET.parse(in_file)
        root = tree.getroot()

        out_meta = self.init_out_meta()
        for obj in root.iter('object'):
            x_list = []
            y_list = []
            cls = obj.find('name').text
            for obj_pt in obj.find('polygon').findall('pt'):
                x_list.append(float(obj_pt.find('x').text))
                y_list.append(float(obj_pt.find('y').text))
            if (len(x_list) >= 2 and len(y_list) >= 2):
                x_list = sorted(set(x_list))
                y_list = sorted(set(y_list))
                annotations = {
                    'left': int(x_list[0]),
                    'top': int(y_list[0]),
                    'right': int(x_list[1]),
                    'bottom': int(y_list[1]),
                    'confidence': 1,
                    'label': cls,
                    'type': 'GroundTruth'}
                out_meta['annotations'].append(annotations)

        return out_meta

    def init_out_meta(self):
        img_sz = self.input_data.shape
        init_meta = {}
        init_meta['size'] = {'width': img_sz[1], 'height': img_sz[0]}
        init_meta['data_type'] = 'image'
        init_meta['folder'] = os.path.dirname(self.img_path)
        init_meta['filename'] = os.path.basename(self.img_path)
        init_meta['annotations'] = list()
        return init_meta

    def find_xml_path(self, img_path):
        tmp_path = img_path.replace('Images', 'Annotations')
        xml_path = tools.replace_extension(tmp_path, 'xml')
        err_msg = "\n[ERROR] Folder is not in Labelme input format\n" +\
                  xml_path + " does not exists!"
        assert os.path.exists(xml_path), err_msg
        return xml_path

    def pre_process(self):
        pass

    def main_process(self):
        self.img_path = os.path.abspath(self.img_path)
        self.read_data(self.img_path, 'image')
        self.input_data = self.output_data[-1]

        xml_path = self.find_xml_path(self.img_path)
        meta = self.get_convert_annotation(xml_path)
        self.results = meta
        meta_len = len(meta['annotations'])
        self.output_data = [self.input_data]*meta_len

    def post_process(self):
        if self.svfolder_path:
            if not os.path.isdir(self.svfolder_path):
                tools.check_dir(self.svfolder_path)
            elif self.lab_flag:
                while(self.query_flag):
                    print('\nThere already exists a folder on '
                          'save folder path...')
                    ans = input("choosing action\n"
                                "1:Don't save, 2:overwrite? ")
                    if ans == '1':
                        print('Exit the post_process...')
                        self.save_flag = False
                        self.query_flag = False
                        break
                    elif ans == '2':
                        shutil.rmtree(self.svfolder_path)
                        tools.check_dir(self.svfolder_path)
                        self.query_flag = False
                        break
            else:
                self.save_flag = True

            if self.save_flag:
                basename = os.path.basename(self.img_path)
                json_name = tools.replace_extension(basename, 'json')
                o_json_path = os.path.join(self.svfolder_path, json_name)
                data.write_json(self.results, o_json_path)
                restore_ordered_json(o_json_path)
        else:
            print('Please set saving folder first !')
            sys.exit(0)
