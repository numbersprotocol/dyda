import cv2
from dyda.core import dyda_base


class DrawerBase(dyda_base.TrainerBase):

    def __init__(self):
        """ __init__ of DrawLabelBase """

        super(DrawerBase, self).__init__()
        self.config_data = []
        self.results['draw_type'] = ''

    def get_labelname(self, labelid):
        name_dict = self.config_data["class_info"]
        for key, val in name_dict.items():
            if labelid == val:
                return key

    def get_drawing_setting(self, labelid):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        lineType = 5
        idx = labelid + 1
        class_num = self.config_data["class_number"]
        color_setting = ((idx/4)%2, (idx/2)%2, idx%2)
        color = [255*x for x in color_setting]
        return font, fontScale, lineType, color

    def reset_output_data(self):
        """ Reset output_data """
        self.output_data = []

    def reset_results(self):
        """ Reset results """
        pass
