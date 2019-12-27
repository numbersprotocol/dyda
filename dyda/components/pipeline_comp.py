from dyda.core import dyda_base


class OutputGeneratorFromMeta(dyda_base.TrainerBase):

    def __init__(self, dyda_config_path=''):
        """ __init__ of OutputGeneratorFromMeta

        TrackerBase.input_data
            component/key pairs defined in pipeline config
        TrackerBase.results
            lab-format results using values specified in comp_key_pairs

        """

        super(OutputGeneratorFromMeta, self).__init__(
            dyda_config_path=dyda_config_path
        )
        self.discard_extra = False
        if "discard_extra_anno" in self.param.keys():
            self.discard_extra = self.param["discard_extra_anno"]
        self.ignore_len_diff = False
        if "ignore_anno_len_diff" in self.param.keys():
            self.ignore_len_diff = self.param["ignore_anno_len_diff"]

    def main_process(self):
        """
        main_process function will be called in the run function after
        pre_process and before post_process. The main logical computation
        of data should happen here.
        """

        # return default results when pipeline status is 0
        if self.pipeline_status == 0:
            self.results = {
                "folder": None,
                "filename": self.metadata[0],
                "size": {
                    "width": None,
                    "height": None
                },
                "annotations": []
            }
            return

        classes = [pair[0] for pair in self.input_data]
        pipeline_classes = [meta["comp_name"] for meta in self.metadata[1:]]
        counter = 0

        len_check = self.check_meta_length(
            include_comp=classes, use_comp_name=True
        )
        if len_check is False:
            self.logger.error(
                "Length of component results do not match."
            )
            self.terminate_flag = True
            return False

        self.results = []
        for i in range(0, len_check):
            self.results.append({})

        for pair in self.input_data:
            cls = pair[0]
            key = pair[1]

            if cls == "external":
                if key == "data":
                    results = self.external_data
                elif key == "metadata":
                    results = self.external_metadata
                else:
                    self.logger.error(
                        "Wrong type of external input setting"
                    )
                    self.terminate_flag = True
                    return False

                if isinstance(results, list) and \
                        len(self.results) == len(results):
                    for i in range(0, len(results)):
                        self.results[i]["external_" + key] = results[i]

                elif isinstance(results, str):
                    self.results[0]["external_" + key] = results
                else:
                    self.logger.warning(
                        "Length of results and external %s does not match,"
                        " add everything to results[0]" % key
                    )
                    self.results[0]["external_" + key] = results
                continue

            if cls not in pipeline_classes:
                self.logger.error(
                    "%s does not exist in current pipeline!" % cls
                )
                self.terminate_flag = True
                return False

            for meta_sess in self.metadata[1:]:
                if meta_sess["comp_name"] == cls:
                    results = meta_sess["results"]
                    self.get_key_value(results, key, pair)
                    break
                else:
                    continue

        if self.unpack_single_list:
            self.unpack_single_results()

    def get_key_value(self, results, key, pair):
        """ Get values from the key specified in the class-key pair """

        if isinstance(results, dict):
            if key == "annotations":
                self.get_anno_value(results, key, pair, 0)
            else:
                self.results[0][key] = results[key]

        elif isinstance(results, list):
            for i in range(0, len(results)):
                if key == "annotations":
                    self.get_anno_value(results[i], key, pair, i)
                else:
                    self.results[i][key] = results[i][key]

    def get_anno_value(self, results, key, pair, ith):
        """ Get values for annotation results, key is always annotations"""

        if len(pair) == 2:
            self.results[ith][key] = results[key]
        elif len(pair) == 3:
            if not isinstance(pair[2], list):
                self.logger.warning(
                    "Wrong type of third item of input pair, use all."
                )
                self.results[ith][key] = results[key]
            else:
                if key not in self.results[ith].keys():
                    self.results[ith][key] = []
                    for j in range(0, len(results[key])):
                        self.results[ith][key].append({})
                len_existing = len(self.results[ith][key])
                len_this = len(results[key])
                if len_existing != len_this:
                    self.logger.warning(
                        "[%s] Length of results does not match, %i v.s. %i"
                        % (pair[0], len_existing, len_this)
                    )
                for sub_key in pair[2]:
                    if len_existing == len_this:
                        self.assign_anno_values(
                            len_existing, key, sub_key, ith, results
                        )
                    elif len_existing > len_this:
                        if self.ignore_len_diff:
                            self.assign_anno_values(
                                len_this, key, sub_key, ith, results
                            )
                        else:
                            self.terminate_flag = True
                            self.logger.error(
                                "ignore_len_diff is True, raise"
                            )
                            return False
                    else:
                        if self.discard_extra:
                            self.assign_anno_values(
                                len_existing, key, sub_key, ith, results
                            )
                        else:
                            if self.ignore_len_diff:
                                if len(self.results[ith][key]) < len_this:
                                    for l in range(len_existing, len_this):
                                        self.results[ith][key].append({})
                                self.assign_anno_values(
                                    len_this, key, sub_key, ith, results
                                )
                            else:
                                self.terminate_flag = True
                                self.logger.error(
                                    "ignore_len_diff is True, raise"
                                )
                                return False
        else:
            self.logger.error(
                "Wrong pair settings, check pipeline config"
            )
            self.terminate_flag = True
            return False

    def assign_anno_values(self, len_annos, key, sub_key, ith, results):
        """ Assign anno value, key is always annotations """

        for j in range(0, len_annos):
            try:
                self.results[ith][key][j][sub_key] = results[key][j][sub_key]
            except KeyError:
                self.logger.warning(
                    "Fail to get value of %s, keep old."
                    % sub_key
                )
