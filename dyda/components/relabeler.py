import cv2
import numpy as np
from dyda_utils import image
from dyda.core import data_converter_base
from dyda.components import det_drawer


class Relabeler(data_converter_base.ConverterBase):
    """docstring for Relabeler"""
    def __init__(self):
        super(Relabeler, self).__init__()
        '''
        input_data = {'labelid': labelid, 'bbox': [ixy, xy],
                         'relabel': bool, 'img_path': img_path}}
        output_data: same format of input_data, change the label
        '''

        # TODO:
        # now determinator does not fit the latest version of dyda_base
        # self.deter = determinator.DeterminatorConfidenceThreshold()
        self.config_data = {}
        self.init_img = []
        self.draw_tool = det_drawer.DetDrawer()
        self.disp_sz = (320, 240)
        self.disp_scale = 1.0
        self.use_key = True
        self.fig_name = ''
        self.key = ''

        self.action = 0

    def change_labelid(self):
        idx = self.output_data["labelid"]
        class_num = self.config_data["class_number"]
        self.output_data["labelid"] = (idx + 1) % class_num
        self.update_disp_img()

    def rm_label(self):
        '''
        batch_relabeler will not save the label
        if labelid > class_num
        '''
        class_num = self.config_data["class_number"]
        self.output_data["labelid"] = class_num + 1
        self.update_disp_img()

    def update_disp_img(self):
        '''
        redraw the label result and disp on screen
        '''
        img = self.init_img.copy()
        data = [img, self.output_data]
        self.draw_tool.input_data = data
        self.draw_tool.run()
        draw_img = self.draw_tool.output_data
        display_img = cv2.resize(draw_img, self.disp_sz)
        cv2.imshow(self.fig_name, display_img)

    def check_mouse_inroi(self, m_x, m_y):
        xmin, ymin = self.output_data["bbox"][0]
        xmax, ymax = self.output_data["bbox"][1]

        if m_x * self.disp_scale[0] < xmax and \
           m_y * self.disp_scale[1] < ymax and \
           m_x * self.disp_scale[0] > xmin and \
           m_y * self.disp_scale[1] > ymin:
            return True
        else:
            return False
        return False

    def mousce_action(self, event, x, y, flags, param):
        if self.check_mouse_inroi(x, y):
            if event == cv2.EVENT_LBUTTONUP:
                self.set_change_flag()
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.set_remove_flag()

    def key_action(self):
        if self.key == ord('r'):
            self.set_change_flag()
        elif self.key == ord('q'):
            self.set_remove_flag()
        return False

    def set_change_flag(self):
        self.action = 1

    def set_remove_flag(self):
        self.action = 2

    def set_config(self, input_data, config_data, use_key):
        self.input_data = input_data
        self.config_data = config_data
        self.draw_tool.config_data = config_data
        self.use_key = use_key
        self.output_data = self.input_data
        img_path = self.input_data['img_path']
        self.init_img = image.read_img(img_path)

        img_sz = self.init_img.shape
        img_sz = np.multiply([img_sz[1], img_sz[0]], 1.0)
        self.disp_scale = np.divide(img_sz, self.disp_sz)

    def pre_process(self):
        if self.use_key:
            self.key_action()

    def main_process(self):
        self.results['convert_type'] = 'change_label'
        if self.action == 1:
            self.change_labelid()
        elif self.action == 2:
            self.rm_label()

        self.action = 0

    def post_process(self):
        pass
