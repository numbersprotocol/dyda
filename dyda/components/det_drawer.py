import cv2
from dyda.core import drawer_base
from dyda_utils import tinycv

class DetDrawer(drawer_base.DrawerBase):
    """docstring for DetDrawer"""
    """drawing tool for detection label"""
    def __init__(self):
        super(DetDrawer, self).__init__()


    def draw(self, label_data, img):
        ix, iy = label_data["bbox"][0]
        x, y = label_data["bbox"][1]
        labelid = label_data["labelid"]
        class_num = self.config_data["class_number"]
        if labelid < class_num:
            label_name = self.get_labelname(labelid)
            settings = self.get_drawing_setting(labelid)
            font, fontScale, lineType, color = settings
            # loc == (top, bottom, left, right)
            if "annot" in label_data:
                if "id" in label_data["annot"]:
                    label_name = label_name + " track id:" +\
                    str(label_data["annot"]["id"])
            loc = (iy, y, ix, x)
            tinycv.patch_bb_img(img, loc, color, lineType)
            tinycv.patch_text(img, label_name, (ix, iy - 10), color,
                        fontScale, lineType)
        self.output_data = img

    def main_process(self):
        img = self.input_data[0]
        label_data = self.input_data[1]
        self.draw(label_data, img)
        self.results['draw_type'] = 'Detection'
