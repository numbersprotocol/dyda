import copy
import numpy as np
import pandas as pd
from dyda.core import data_selector_base


class MetaInfoSelector(data_selector_base.DataSelectorBase):
    """Select metadata information or pre-defined parameters based on
       channel_index, channel_name, or node_name

       @param key_look_for: string, should be channel_index,
                      channel_name or node_name. Default: channel_index.
       @param meta_info_dic: dictionaty of pre-defined metadata, the key should
                         be pre-defined. Default: empty dict.

       example:
          external_metadata = {"channel_id": "1"}
          key_look_for = "channel_id"
          self.meta_info_dic = {
              "0": {"parm": [16, 5, 3, 2.4]},
              "1": {"parm": [20, 1, 3, 8.5]}
          }
          => after running the component:
             self.results = {"parm": [20, 1, 3, 8.5]}
    """

    def __init__(self, dyda_config_path=""):
        """ __init__ of MetaInfoSelector """

        super(MetaInfoSelector, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

        self.key = None
        self.key_look_for = "channel_index"
        if "key_look_for" in self.param.keys():
            self.key_look_for = self.param["key_look_for"]
        self.meta_info_dic = None
        self.meta_info = {}
        if "meta_info_dic" in self.param.keys():
            self.meta_info_dic = self.param["meta_info_dic"]

    def main_process(self):
        """ define main_process of dyda component """
        try:
            self.key = self.external_metadata[self.key_look_for]
        except KeyError:
            self.logger.info("Fail to get key value, terminate")
            self.terminate_flag = True
            return False
        try:
            self.meta_info = self.meta_info_dic[self.key]
        except KeyError:
            self.logger.info("Fail to get meta info, terminate.")
            self.terminate_flag = True
            return False

        self.results = self.meta_info
        return True


class RandomDataSelector(data_selector_base.DataSelectorBase):
    """Randomly select the data in DataFrame by given feature

       @param random_by: randomly select data based on this parameter
       @param how_many: how many unique values in random_by want to select
       @split: if equals "yes", return the data in independent DataFrame

       example: random_by: id, how_many: 32
       >> randomly select 32 ids and their associated data in the DataFrame

       note: if how_many is bigger than the total unique values in the
             specified feature, will return the total DataFrame
    """

    def __init__(self, dyda_config_path=""):
        """ __init__ of RandomDataSelector """

        super(RandomDataSelector, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.set_param(self.class_name)

        self.random_by = self.param["random_by"]
        if "how_many" in self.param.keys():
            self.how_many = self.param["how_many"]
        else:
            self.how_many = 8

        if "split" in self.param.keys():
            self.split = self.param["split"]
        else:
            self.split = False

    def main_process(self):
        """ define main_process of dyda component """
        self.pack_input_as_list()

        for input_df in self.input_data:
            uniques = pd.unique(input_df[self.random_by])
            if self.how_many >= len(uniques):
                selected = uniques
            else:
                selected = np.random.choice(uniques,
                                            self.how_many,
                                            replace=False)

            output_dfs = []
            if self.split:
                for i in selected:
                    mask = (input_df[self.random_by] == i)
                    output_dfs.append(input_df[mask])
            else:
                mask = input_df[self.random_by].isin(selected)
                output_dfs.append(input_df[mask])

            self.output_data.append(output_dfs)
        self.unpack_single_output()
