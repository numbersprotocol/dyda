import warnings
import cv2
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from dt42lab.core import tools
from dt42lab.core import plot
from dyda.core import data_plotter_base
plt.switch_backend('Agg')
warnings.filterwarnings("ignore")


class DataFrameHistPlotter(data_plotter_base.DataPlotterBase):
    """ Plot the histogram of input data
        The DataFrame in class name means that this component only
        support for DataFrame.

        @param hist_feature: the feature we want to plot histogram
        @belongs_to: if the DataFrame contains one column that records
                     the data belongs to who, like "id", feed the column
                     name in this parameter, and the title of plot will
                     contain the name.
        @bins: the bins of histogram
        @range: the lower and upper range of the bins, if not specified,
                will auto use minimum and maximum of data.
                the length of range must equal the length of hist_feature,
                if there are features you do not want to specify range, use
                None instead.
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(DataFrameHistPlotter, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        self.hist_feature = self.param['hist_feature']
        if not isinstance(self.hist_feature, list):
            self.hist_feature = [self.hist_feature]

        if 'belongs_to' in self.param.keys():
            self.belongs_to = self.param['belongs_to']
        else:
            self.belongs_to = None
        if 'bins' in self.param.keys():
            self.bins = self.param['bins']
        else:
            self.bins = 10
        if 'range' in self.param.keys():
            self.hist_range = self.param['range']
        else:
            self.hist_range = [None for i in self.hist_feature]

    def main_process(self):
        """ Main function of dyda component. """
        self.pack_input_as_list()

        # let input_data will always be list of list
        if not any(isinstance(i, list) for i in self.input_data):
            self.input_data = [self.input_data]

        for dfs in self.input_data:
            fig, axeses = plt.subplots(len(dfs), len(self.hist_feature),
                                       figsize=(6.4 * len(self.hist_feature),
                                                4.8 * len(dfs)))

            # deal the case with len(dfs) or len(self.hist_feaure) = 1
            axeses = np.array(axeses)
            axeses = axeses.reshape(len(dfs), len(self.hist_feature))

            for df, axes in zip(dfs, axeses):

                if self.belongs_to is not None:
                    uniques = pd.unique(df[self.belongs_to]).tolist()
                    uniques = [i[:5] if len(i) > 5 else i for i in uniques]
                    if len(uniques) == 1:
                        uniques = uniques[0]

                for feature, ax, hist_range in zip(self.hist_feature,
                                                   axes,
                                                   self.hist_range):
                    ax.hist(df[feature].values,
                            bins=self.bins,
                            range=hist_range)
                    if 'uniques' in locals():
                        ax.set_title(str(feature) + '_' + str(uniques))
                    else:
                        ax.set_title(str(feature))
                    ax.ticklabel_format(useOffset=False)
            plt.tight_layout()

            # convert the fig to numpy array
            fig.canvas.draw()
            output = np.array(fig.canvas.renderer._renderer)
            # numpy default RGB, cv2 default BGR
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            self.output_data.append(output)
        self.unpack_single_output()


class LocationBubblePlotter(data_plotter_base.DataPlotterBase):
    """ Plot the bubble plot of input data
        only support DataFrame.

        @start_time: if given, only plot data after start_time
        @end_time: if given, only plot data before end_time
        @belongs_to: if the DataFrame contains one column that records
                     the data belongs to who, like "id", feed the column
                     name in this parameter, and the title of plot will
                     contain the name.
        @timezone: change the timezone of datetime, default UTC
        @overlap: if True, overlap bubble plot with give image
        @overlap_path: the path of image to overlap
        @plot_arrow: if True, plot arrow on image
    """

    def __init__(self, dyda_config_path='', param=None):
        """ Initialization function of dyda component. """

        super(LocationBubblePlotter, self).__init__(
            dyda_config_path=dyda_config_path
        )
        class_name = self.__class__.__name__
        self.set_param(class_name, param=param)

        if 'start_time' in self.param.keys():
            self.start_time = self.param['start_time']
        else:
            self.start_time = None

        if 'end_time' in self.param.keys():
            self.end_time = self.param['end_time']
        else:
            self.end_time = None

        if 'belongs_to' in self.param.keys():
            self.belongs_to = self.param['belongs_to']
        else:
            self.belongs_to = None

        if 'timezone' in self.param.keys():
            self.timezone = self.param['timezone']
        else:
            self.timezone = 'UTC'

        if 'overlap' in self.param.keys():
            self.overlap = self.param['overlap']
        else:
            self.overlap = False

        if 'overlap_path' in self.param.keys():
            self.overlap_path = self.param['overlap_path']
        else:
            self.overlap_path = None

        if 'plot_arrow' in self.param.keys():
            self.plot_arrow = self.param['plot_arrow']
        else:
            self.plot_arrow = False

    def main_process(self):
        """ Main function of dyda component. """
        self.pack_input_as_list()

        # let input_data will always be list of list
        if not any(isinstance(i, list) for i in self.input_data):
            self.input_data = [self.input_data]

        for dfs in self.input_data:
            outputs = []

            for df in dfs:

                self.get_datetime(df)
                if self.belongs_to is not None:
                    uniques = pd.unique(df[self.belongs_to]).tolist()
                    uniques = [i[:5] if len(i) > 5 else i for i in uniques]
                    if len(uniques) == 1:
                        uniques = uniques[0]

                if self.start_time is not None:
                    start_time = pd.Timestamp(self.start_time)
                    start_time = start_time.tz_localize(self.timezone)
                    mask = df['datetime'] >= start_time
                    df = df[mask]

                if self.end_time is not None:
                    end_time = pd.Timestamp(args.end_time)
                    end_time = end_time.tz_localize(self.timezone)
                    mask = df['datetime'] <= end_time
                    df = df[mask]
                df.sort_values(by='datetime', inplace=True)
                df.loc[:, 'weekday'] = df.loc[:, 'datetime'].dt.weekday
                df.reset_index(drop=True, inplace=True)
                fig = self.bubble_plot(df, overlap=self.overlap,
                                       overlap_path=self.overlap_path)

                if 'uniques' in locals():

                    plt.title(str(uniques), fontsize=20)

                fig.canvas.draw()
                output = np.array(fig.canvas.renderer._renderer)
                outputs.append(output)

            output_fig = self.combine_fig(outputs)
            # numpy default RGB, cv2 default BGR
            output_fig = cv2.cvtColor(output_fig, cv2.COLOR_RGB2BGR)
            self.output_data.append(output_fig)

        self.unpack_single_output()

    def combine_fig(self, outputs):
        temp = outputs.copy()
        column_lengths = [output.shape[1] for output in temp]
        to_pad_lengths = (max(column_lengths) - np.array(column_lengths))
        for i, j in enumerate(temp):
            temp[i] = np.pad(j, ((0, 0), (0, to_pad_lengths[i]), (0, 0)),
                             'constant',
                             constant_values=0)

        return np.concatenate(temp, axis=0)

    def get_datetime(self, df):
        df.loc[:, 'datetime'] = pd.to_datetime(df['timestamps'], unit='ms')
        df.loc[:, 'datetime'] = df.loc[:, 'datetime'].dt.tz_localize('UTC')
        df.loc[:, 'datetime'] = \
            df.loc[:, 'datetime'].dt.tz_convert(self.timezone)

    def bubble_plot(self, df, overlap=False,
                    overlap_path=None):

        dfs_bp_clean, right, left, up, down = self.process_df(df)

        width_ratios = [16] * len(dfs_bp_clean) + [1] * 2
        fig, axes = plt.subplots(1, len(dfs_bp_clean) + 2,
                                 figsize=(9.6 * len(dfs_bp_clean) +
                                          2 * 0.3, 6.4),
                                 gridspec_kw={'width_ratios': width_ratios})
        anno_cm = plt.cm.get_cmap('binary')
        anno_norm = mpl.colors.Normalize(
            vmin=min([i['speed'].min() for i in dfs_bp_clean]) - 10,
            vmax=max([i['speed'].max() for i in dfs_bp_clean]) + 10
        )
        weekday_cm = plt.cm.get_cmap('Paired')
        weekday_bounds = np.arange(-0.5, 7.5, step=1)
        weekday_norm = mpl.colors.BoundaryNorm(weekday_bounds, weekday_cm.N)
        weekday_ticks = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        # cb stand for colorbar
        weekday_cb = mpl.colorbar.ColorbarBase(axes[-2],
                                               cmap=weekday_cm,
                                               norm=weekday_norm,
                                               boundaries=weekday_bounds,
                                               ticks=np.arange(0, 7),
                                               spacing='proportional',
                                               orientation='vertical')
        weekday_cb.ax.set_yticklabels(weekday_ticks)
        weekday_cb.set_label('Weekday Color')

        anno_cb = mpl.colorbar.ColorbarBase(axes[-1],
                                            cmap=anno_cm,
                                            norm=anno_norm,
                                            orientation='vertical')
        anno_cb.set_label('Speed')

        if overlap:
            img = plt.imread(overlap_path)

        for h, temp_df in enumerate(dfs_bp_clean):

            if overlap:
                axes[h].imshow(img, extent=[left, right, down, up])

            # plot every weekday separately because plot with
            # every weekday different color
            for day in temp_df['weekday'].drop_duplicates():
                df_to_plot = temp_df[temp_df['weekday'] == day]
                x = df_to_plot['longitude']
                y = df_to_plot['latitude']
                z = df_to_plot['time_passing']
                w = df_to_plot['weekday']

                axes[h].scatter(x, y,
                                c=weekday_cm(weekday_norm(day)),
                                edgecolor='face',
                                alpha=0.65,
                                s=z * 0.01)
            axes[h].set_xlim(left=left, right=right)
            axes[h].set_ylim(bottom=down, top=up)
            start_time = temp_df['datetime'].iloc[0].strftime(
                '%B %d, %a, %Y, %r')
            end_time = temp_df['datetime'].iloc[-1].strftime(
                '%B %d, %a, %Y, %r')
            axes[h].set_title(str(start_time) + '~' + str(end_time))
            axes[h].set_xlabel('longitude')
            axes[h].set_ylabel('latitude')
            axes[h].ticklabel_format(useOffset=False)
            axes[h].set_aspect('auto')

            # plot arrow
            if self.plot_arrow:

                z_now = temp_df['speed']
                # m stand for moving
                m_z_now = z_now[z_now != 0]
                m_indexs = m_z_now.index

                m_x_now = temp_df.loc[m_indexs, 'longitude'].values
                m_x_next = temp_df.loc[m_indexs + 1, 'longitude'].values
                m_y_now = temp_df.loc[m_indexs, 'latitude'].values
                m_y_next = temp_df.loc[m_indexs + 1, 'latitude'].values

                for ii, jj, kk, ll, mm in zip(m_z_now,
                                              m_x_now, m_x_next,
                                              m_y_now, m_y_next):
                    z_norm = anno_norm(ii)
                    axes[h].annotate('',
                                     xy=(kk, mm),
                                     xytext=(jj, ll),
                                     arrowprops=dict(facecolor=anno_cm(z_norm),
                                                     edgecolor=anno_cm(z_norm),
                                                     width=0.5 * 5,
                                                     headlength=1.6 * 6,
                                                     headwidth=0.8 * 10,
                                                     shrink=0.011,
                                                     alpha=0.7))
        plt.tight_layout()

        return fig

    def process_df(self, df):

        df_bp = df.copy()
        df_bp.drop(labels=['timestamps'],
                   axis=1,
                   inplace=True)
        right, left, up, down = self.get_boundary(df_bp)

        dfs_bp = self.slice_df_by_continuous_day(df_bp)
        dfs_bp_clean = self.shrink_df_by_same_location(dfs_bp)

        for i in dfs_bp_clean:

            time_passing = i['datetime'].diff(periods=-1).dt.total_seconds()
            i['time_passing'] = -1 * time_passing
            i['time_passing'].iloc[-1] = 0
            distance = self.geodetic_distance(i['longitude'].iloc[1:].values,
                                              i['longitude'].iloc[:-1].values,
                                              i['latitude'].iloc[1:].values,
                                              i['latitude'].iloc[:-1].values)
            i['distance'] = np.append(distance, 0)

            # compute speed
            time_passing_hr = i['time_passing'][:-1] / 3600
            speed = i['distance'][:-1] / (time_passing_hr)
            i['speed'] = np.append(speed, 0)

        return dfs_bp_clean, right, left, up, down

    def get_boundary(self, df):
        right = df['longitude'].max() + 0.02
        left = df['longitude'].min() - 0.02
        up = df['latitude'].max() + 0.005
        down = df['latitude'].min() - 0.005

        return right, left, up, down

    def shrink_df_by_same_location(self, dfs_bp_continuous):
        dfs_bp_clean = []

        for temp_df in dfs_bp_continuous:

            s_i = self.drop_continuous_duplicates_numeric(
                temp_df.loc[:,
                            ['longitude',
                             'latitude',
                             'weekday']]
            ).index.values
            s_i = np.append(s_i, temp_df.index.values.max())

            temp_df = temp_df.iloc[s_i, :]
            temp_df.reset_index(drop=True, inplace=True)
            dfs_bp_clean.append(temp_df)

        return dfs_bp_clean

    def slice_df_by_continuous_day(self, df_bp):
        df_bp_continuous = []
        temp_series = df_bp.loc[:, 'datetime'].dt.dayofyear
        # s_i stand for slice_index which be used to slice DataFrame
        s_i = df_bp[(temp_series.diff() != 1) &
                    (temp_series.diff() != 0) &
                    (temp_series.diff() != -364) &
                    (temp_series.diff() != -365)].index.values
        s_i = np.append(s_i, df_bp.index.values.max() + 1)
        for i in range(len(s_i) - 1):
            to_append = df_bp.iloc[s_i[i]:s_i[i + 1], :]
            to_append.reset_index(drop=True, inplace=True)

            df_bp_continuous.append(to_append)
        return df_bp_continuous

    def geodetic_distance(self, lon1, lon2, lat1, lat2):
        lon1 = np.radians(lon1)
        lon2 = np.radians(lon2)
        lat1 = np.radians(lat1)
        lat2 = np.radians(lat2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2)**2 + \
            np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        R = 6373.0
        distance = R * c

        return distance

    def drop_continuous_duplicates(self, df):
        same_as_previous = pd.DataFrame()
        for index, row in df.iterrows():
            if index == 0:
                compare_row = row
                same_as_previous.loc[index, 'same'] = 1
            else:
                if row.tolist() == compare_row.tolist():
                    same_as_previous.loc[index, 'same'] = 0

                else:
                    compare_row = row
                    same_as_previous.loc[index, 'same'] = 1
        return(df[same_as_previous['same'] == 1])

    def drop_continuous_duplicates_numeric(self, df):

        temp = ((df.diff() != 0).sum(axis=1))

        return df[temp != 0]
