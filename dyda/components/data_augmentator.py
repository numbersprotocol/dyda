import copy
from dt42lab.core import tinycv
from dyda.core import data_augmentator_base


class RTConverterSingleSeed(data_augmentator_base.AugmentatorBase):
    """ Radial Transform of input image data """

    def __init__(self, dyda_config_path='', param=None):
        super(RTConverterSingleSeed, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)
        self.output_data = []
        self.results = []

    def main_process(self):
        """ main_process of RTConverter """

        seed = self.param["seed"]
        for data_matrix in self.input_data:
            _seed = self.get_seed(seed, data_matrix)
            new_img, (U, V, m, n) = tinycv.img_radia_transform_return_info(
                data_matrix, _seed, precision=self.param["precision"]
            )
            self.output_data.append(new_img)
            self.results.append({
                "U": U,
                "V": V,
                "m": m,
                "n": n
            })

    def get_seed(self, seed, data_matrix):
        """ get dynamic seeds from input seed info and data_matrix """
        if isinstance(seed, str):
            if seed == "center":
                shape = data_matrix.shape
                m = int(shape[0] / 2)
                n = int(shape[1] / 2)
                return (m, n)
            else:
                self.logger.error("seed method is not yet supported %s"
                                  % seed)
                self.terminate_flag = True
        else:
            return seed

    def reset_results(self):
        """ reset_results of RTConverter"""
        self.results = []


class RTConverterMultipleSeeds(RTConverterSingleSeed):
    """
        Radial Transform of input image data, accept multiple seeds
        and output_data is lists of lists, each child list contains all
        converted matrix from nth input using all seeds.
    """

    def __init__(self, dyda_config_path='', param=None):
        super(RTConverterMultipleSeeds, self).__init__(
            dyda_config_path=dyda_config_path
        )

    def main_process(self):
        """ main_process of RTConverter """

        self.output_data = []
        self.results = {"length": 0, "seeds": []}
        seeds = self.param["seeds"]
        self.check_input_size()

        for seed in seeds:
            _seed = self.get_seed(seed, self.input_data)
            new_img, (U, V, m, n) = tinycv.img_radia_transform_return_info(
                self.input_data, _seed, precision=self.param["precision"]
            )
            self.output_data.append(new_img)
            self.results["seeds"].append({
                "U": U,
                "V": V,
                "m": m,
                "n": n
            })
        self.results["length"] = len(self.results["seeds"])


class DataDuplicator(data_augmentator_base.AugmentatorBase):
    """
        Data Suplicator is a simple component to duplicate the input data
        into N copies, where N comes from "length" key of the metadata
    """

    def __init__(self, dyda_config_path='', param=None):
        super(DataDuplicator, self).__init__(
            dyda_config_path=dyda_config_path
        )

    def main_process(self):
        """ main_process of DataDuplicator """

        self.output_data = []
        data_to_dup = []
        ncopies = 1
        if isinstance(self.input_data, list):
            if len(self.input_data) == 2:
                data_to_dup = copy.deepcopy(self.input_data[0])
                if isinstance(self.input_data[1], dict):
                    try:
                        ncopies = int(self.input_data[1]["length"])
                    except:
                        try:
                            ncopies = int(self.param["ncopies"])
                        except:
                            self.logger.error(
                                "Cannot get ncopies from input_data or param"
                            )
                            self.terminate_flag = True
                            return False
                else:
                    self.logger.error(
                        "Wrong input, the second object type should be dict"
                    )
                    self.terminate_flag = True
                    return False
            elif len(self.input_data) == 1:
                try:
                    ncopies = int(self.param["ncopies"])
                except:
                    self.logger.error(
                        "Cannot get ncopies from dyda config"
                    )
                    self.terminate_flag = True
                    return False
            else:
                self.logger.error(
                    "Wrong input_data size, should be 1 or 2"
                )
                self.terminate_flag = True
                return False
        else:
            data_to_dup = copy.deepcopy(self.input_data)

        for i in range(0, ncopies):
            self.output_data.append(data_to_dup)
