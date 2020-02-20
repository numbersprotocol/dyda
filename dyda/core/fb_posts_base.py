import pandas as pd
from dyda.core import dyda_base


class FbPostsBase(dyda_base.TrainerBase):
    """ Base class of detector """

    def __init__(self, dyda_config_path=''):
        """ Init function of DetectorBase """
        super(FbPostsBase, self).__init__(
            dyda_config_path=dyda_config_path
        )
        # Note: results is a list of output from
        #       lab_tools.output_pred_detection
        self.results = []
        self.define_columns = ['post_id',
                               'timestamp',
                               'update_timestamp',
                               'title',
                               'post',
                               'external_context_url',
                               'external_context_source',
                               'external_context_name',
                               'event_name',
                               'event_start_timestamp',
                               'event_end_timestamp',
                               'dyda_column']

    def reset_results(self):
        self.results = []

    def reset_output_data(self):
        """ Reset output_data """
        self.output_data = []

    def return_output_data_sample(self):
        return pd.DataFrame(columns=self.define_columns)

    def return_results_sample(self):
        return {
            "data_owner": "",
            "filename": "",
            "folder": "",
            "timestamp": "",
            "sha256sum": "",
            "annotations": {}}

    def check_columns(self, to_check):
        """
        check the to_check is DataFrame with defined columns or
        a list of DataFrame with defined columns
        """

        if isinstance(to_check, list):
            is_df = [isinstance(i, pd.DataFrame) for i in to_check]
            if all(is_df):
                if all([set(i.columns) == set(self.define_columns)
                        for i in to_check]):
                    return True
                else:
                    return False
            else:
                return False
        elif isinstance(to_check, pd.DataFrame):
            if set(to_check.columns) == set(self.define_columns):
                return True
            else:
                return False
        else:
            return False
