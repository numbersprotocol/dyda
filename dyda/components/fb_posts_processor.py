import fbjson2table as fj2t
import json
from fbjson2table.table_class import TempDFs
import requests
import pandas as pd
import numpy as np
from dyda.core import fb_posts_base


class FbYourPostsJsonConverter(fb_posts_base.FbPostsBase):
    """Extract wanted data from input json,
       and convert it into DataFrame

       input: JSON_LIKE_DICT, or JSON_LIKE_DICT

       output: pd.DataFrame, or list of pd.DataFrame

    """

    def __init__(self, dyda_config_path="", param=None):
        super(FbYourPostsJsonConverter, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

    def main_process(self):
        """ Main function called by the external code """
        self.uniform_input()
        self.reset_output()

        wanted_columns = ['timestamp',
                          'data_update_timestamp',
                          'title',
                          'data_post',
                          'external_context_url',
                          'external_context_source',
                          'external_context_name',
                          'event_name',
                          'event_start_timestamp',
                          'event_end_timestamp']

        for input_json in self.input_data:
            temp_dfs = TempDFs(input_json, table_prefix='your_posts')
            df, top_id = temp_dfs.temp_to_wanted_df(
                wanted_columns=wanted_columns)
            df.rename(columns={
                top_id: 'post_id',
                "data_update_timestamp": "update_timestamp",
                "data_post": "post"}, inplace=True)
            df.loc[:, 'dyda_column'] = np.nan
            self.output_data.append(df)

        self.uniform_output()


class FBPostsDataSelector(fb_posts_base.FbPostsBase):
    """ Get FB post data in the specified period

        input: pd.DataFrame, or list of pd.DataFrame

        output: pd.DataFrame, or list of pd.DataFrame

        @param post_start_timestamp: the minimum timestamp
            of selected df

        @param post_end_timestamp: the maximum timestamp
            of selected df
    """

    def __init__(self, param=None,
                 dyda_config_path=""):

        super(FBPostsDataSelector, self).__init__(
            dyda_config_path=dyda_config_path
        )

        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.post_start_timestamp = None
        if "post_start_timestamp" in self.param.keys():
            self.post_start_timestamp = self.param["post_start_timestamp"]

        self.post_end_timestamp = None
        if "post_end_timestamp" in self.param.keys():
            self.post_end_timestamp = self.param["post_end_timestamp"]

    def main_process(self):
        """ main process of dyda component """
        self.reset_output()
        self.uniform_input()

        for input_df in self.input_data:
            df = input_df.copy(deep=True)

            if not self.check_columns(df):
                self.logger.error("input df is not defined df")
                self.terminate_flag = True
                break

            if self.post_start_timestamp is not None:
                df = df[df['timestamp'] >= self.post_start_timestamp]
            if self.post_end_timestamp is not None:
                df = df[df['timestamp'] <= self.post_end_timestamp]
            self.output_data.append(df)


class FbPostSentimentAnalyzer(fb_posts_base.FbPostsBase):
    """ Calculate the sentiment score of post

        input: pd.DataFrame, or a list of pd.DataFrame

        output: pd.DataFrame, or a list of pd.DataFrame,
               with sentiment score of each post in "dyda_column."
                And we store the score of whole DataFrame in results.

        @param sentiment_api_key: the api key of sentiment api

        @param norm_min: the minimum value of normalized sentiment score

        @param norm_max: the maximum value of normalized sentiment score

        @param cutoff_low_scale: the fraction of cutoff low
    """

    def __init__(self, param=None,
                 dyda_config_path=''):
        super(FbPostSentimentAnalyzer, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.sentiment_api_url = \
            'https://api.deepai.org/api/sentiment-analysis'

        if "sentiment_api_key" in self.param.keys():
            self.sentiment_api_key = self.param["sentiment_api_key"]
        else:
            self.logger.error("sentiment_key not specified")
            sys.exit(1)

        self.sentiment_api_headers = {
            'api-key': self.sentiment_api_key,
        }

        self.norm_min = 0
        if "norm_min" in self.param.keys():
            try:
                self.norm_min = int(self.param["norm_min"])
            except ValueError as e:
                self.logger.warning("fail to set norm_min, keep 0")
                print(e)
                pass

        self.norm_max = 10
        if "norm_max" in self.param.keys():
            try:
                self.norm_max = int(self.param["norm_max"])
            except ValueError as e:
                self.logger.warning("fail to set norm_max, keep 10")
                print(e)
                pass

        self.cutoff_low_scale = 0.7
        if "cutoff_low_scale" in self.param.keys():
            try:
                self.cutoff_low_scale = float(self.param["cutoff_low_scale"])
            except ValueError as e:
                self.logger.warning("fail to set cutoff_low_scale, keep 0.8")
                print(e)
                pass

    def main_process(self):
        """ main process of dyda component """
        self.reset_output()
        self.uniform_input()

        for input_df in self.input_data:
            df = input_df.copy(deep=True)
            if self.check_columns(df):
                df['dyda_column'] = \
                    df['post'].apply(self.post_to_sentiment_score)
            else:
                self.logger.error("input df is not defined df")
                self.terminate_flag = True
                break

            self.output_data.append(df)
            sentiment_scores = df['dyda_column'].tolist()
            total_score = np.nansum(sentiment_scores)
            self.results.append({
                "analysis": {
                    "posts_sentiment_score": self.normalize_score(total_score),
                    "posts_sentiment_score_orig": total_score
                }})

        self.uniform_output()

    def sentiment_request(self, content):
        data = {
            'text': content
        }
        res = requests.post(
            url=self.sentiment_api_url,
            headers=self.sentiment_api_headers,
            data=data
        )
        try:
            res.raise_for_status()
            return json.loads(res.text)
        except requests.exceptions.HTTPError as error:
            logger.error(error.response.text)
            return {}

    def map_result(self, result):
        if result == 'Positive':
            return 1
        elif result == 'Negative':
            return -1
        else:
            return 0

    def convert_result(self, sentiment_res):
        output = sentiment_res.get('output', None)
        score_list = [self.map_result(r) for r in output]
        return score_list

    def post_to_sentiment_score(self, post):
        if post is not np.nan:
            sentiment_res = self.sentiment_request(post)
            score_list = self.convert_result(sentiment_res)
            score_sum = sum(score_list)

            return score_sum
        else:
            return np.nan

    def normalize_score(self, score):
        """ normalize score to norm_min to norm_max """

        norm_max = self.norm_max
        norm_min = self.norm_min
        cutoff_low = (0 - self.norm_max) * self.cutoff_low_scale

        if score >= norm_max:
            score = norm_max
        if score <= cutoff_low:
            score = cutoff_low

        res = (score - cutoff_low) * (norm_max - norm_min) / (
            norm_max - cutoff_low) + norm_min
        return res
