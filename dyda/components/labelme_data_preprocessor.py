import os
import sys
import numpy as np
from dyda.core import data_reader_base
from dt42lab.core import tools
from dt42lab.core import data
from dt42lab.core import image
from dt42lab.core import boxes
from dt42lab.core import tinycv


class LabelMeDataPreProcessor(data_reader_base.DataReaderBase):
    '''
    input the result of labelme_data_reader
    output the cropped images as out_data
    output the cropped json information as out_meta_data
    '''

    def __init__(self):
        super(LabelMeDataPreProcessor, self).__init__()
        self.metadata_path = None
        self.svfolder_path = None
        self.save_flag = True
        self.reader_meta = []
        self.output_data = np.empty([1])
        self.input_data = np.empty([1])

    def get_crop_imgs(self):
        '''
        cropping image in to sqrt shape
        return a list of cropped imgs
        '''
        sqrt_annotation = boxes.square_extend_in_json(self.reader_meta)
        crop_list = list()
        for i in range(len(sqrt_annotation['annotations'])):
            ymin = sqrt_annotation['annotations'][i]['top']
            ymax = sqrt_annotation['annotations'][i]['bottom']
            xmin = sqrt_annotation['annotations'][i]['left']
            xmax = sqrt_annotation['annotations'][i]['right']
            crop_img = self.input_data[i][ymin:ymax, xmin:xmax, :]
            crop_list.append(crop_img)
        return crop_list

    def detection2classify_metadata(self):
        crop_meta = list()
        for i in range(len(self.output_data)):
            out_meta = dict(self.reader_meta)
            img_sz = self.output_data[i].shape
            out_meta['folder'] = None
            out_meta['filename'] = None
            out_meta['size'] = {'width': img_sz[1], 'height': img_sz[0]}
            out_meta['annotations'] = list()
            annotations = {}
            annotations['top'] = 0
            annotations['bottom'] = img_sz[0]
            annotations['left'] = 0
            annotations['right'] = img_sz[1]
            # FIXME: dangerous to only rely on order of annotations
            #        and output_data lists
            _label = self.reader_meta['annotations'][i]['label']
            annotations['label'] = _label
            out_meta['annotations'] = annotations
            crop_meta.append(out_meta)
        return crop_meta

    def filled_crops_filenames(self, crops_meta):
        i = 0
        for out_meta in crops_meta:
            basename = self.reader_meta['filename']
            post_ext = str(i) + '_crop'
            name = tools.add_str_before_ext(basename, post_ext,
                                            return_type='base-only')
            out_meta['filename'] = name
            i += 1
        return crops_meta

    def pre_process(self):
        pass

    def main_process(self):
        self.reader_meta = (
            self.input_data[-1][-1]["results"]
            )
        self.output_data = self.get_crop_imgs()
        crops_meta = self.detection2classify_metadata()
        crops_meta = self.filled_crops_filenames(crops_meta)
        self.results['additional_info'] = crops_meta
        self.results['data_type'] = 'image'

    def post_process(self):
        if self.svfolder_path:
            if not os.path.isdir(self.svfolder_path):
                tools.check_dir(self.svfolder_path)
            if self.save_flag:
                i = 0
                for out_meta in self.results['additional_info']:
                    name = out_meta['filename']
                    out_meta['folder'] = os.path.abspath(self.svfolder_path)
                    json_name = tools.replace_extension(name, 'json')
                    json_path = os.path.join(self.svfolder_path, json_name)
                    img_path = os.path.join(self.svfolder_path, name)
                    data.write_json(out_meta, json_path)
                    image.save_img(self.output_data[i], img_path)
                    i += 1
        else:
            print('Please set saving folder first !')
            sys.exit(0)
