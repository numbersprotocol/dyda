from berrynet.engine.movidius_engine import MovidiusMobileNetSSDEngine
from dyda_utils import lab_tools
from dyda.core import detector_base


class DetectorMovidiusMobileNetSSD(detector_base.DetectorBase):
    """ Modified from Movidius engine in BerryNet """

    def __init__(self, dyda_config_path=''):
        """ __init__ of ClassifierInceptionv3

        Trainer Variables:
            input_data: a list of image array
            results: defined by lab_tools.output_pred_detection (dict)

        Arguments:
            testing -- True to call set_testing_params. Anything written in
                       dyda.config will be overwritten.
        """

        super(DetectorMovidiusMobileNetSSD, self).__init__(
            dyda_config_path=dyda_config_path
        )

        self.set_param(self.class_name)
        self.check_param_keys()
        param = self.param

        try:
            model = param['net_weights']
            labels = param['net_labels']
            self.engine = MovidiusMobileNetSSDEngine(model, labels)
        except Exception as error:
            print(error)
            print("[movidius_detector] ERROR: loading Movidius model fails")
            raise

    def check_param_keys(self):
        """
        Check if any default key is missing in the self.param.
        """

        # TODO: remove unused keys
        default_keys = [
            "net_weights", "net_labels", "thresh"
        ]
        for _key in default_keys:
            if _key not in self.param.keys():
                print("[movidius_detector] ERROR: %s missing in self.param" % _key)
                self.terminate_flag = True
            else:
                continue
        print("[movidius_detector] INFO: keys of self.param are checked")

    def main_process(self, thresh=.1):
        # TODO: Check whether the FIXME below affects this detector or not.
        # FIXME: the first version of DetectorYOLO accepts a list of image
        #       paths as input_data, this should be fixed (issue #30)
        thresh = self.param["thresh"]
        annos = None

        def nparray_to_image(img):
            data = img.ctypes.data_as(POINTER(c_ubyte))
            image = self.ndarray_image(
                data, img.ctypes.shape, img.ctypes.strides
            )
            return image

        if not isinstance(self.input_data, list):
            self.input_data = [self.input_data]
            package = True
        else:
            package = False

        for img_array in self.input_data:
            image_data = self.engine.process_input(img_array)
            output = self.engine.inference(image_data)
            model_outputs = self.engine.process_output(output)

            # TODO: Write another process_output method to
            #       generate dyda-preferred format instead of
            #       doing redundant format conversions.
            annos = []
            for anno in model_outputs['annotations']:
                annos.append([anno['label'],
                              anno['confidence'],
                              [
                                  anno['top'],
                                  anno['bottom'],
                                  anno['left'],
                                  anno['right'],
                              ]])
            res = lab_tools.output_pred_detection(self.metadata[0], annos)
            self.results.append(res)

        if package:
            self.results = self.results[0]
        return annos
