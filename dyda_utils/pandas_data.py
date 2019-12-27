import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame, Series
from dyda_utils import tools
from dyda_utils import lab_tools
from dyda_utils import data


def _cond_wd(x):
    """Conditions to select entries for sign_diff"""
    return x.weekday()


def get_wd_series(date_series, fm='%Y-%m-%d'):
    """Convert Date string series to pd.DatetimeIndex
       and return a weekday series.

    @param date_series: Raw string series

    Keyword arguments:
    fm -- format of the input date (default: '%Y-%m-%d'

    """
    dates = pd.to_datetime(date_series, format=fm)
    wd = dates.apply(lambda x: _cond_wd(x))
    return dates, wd


def is_empty_df(df):
    """ Check if the DataFrame is empty """
    return df.empty


def rebin_df(df, nbins):
    """Rebin DataFrame"""

    return df.groupby(pd.qcut(df.index, nbins)).mean()


def sel_df_row(df, row_index):
    """ Select one row from DataFrame"""
    return df.iloc[row_index]


def df_to_lab_anno(df_lab):
    """ Revert DataFrame which was converted by lab_anno_to_df
        back to lab-format annotations dict """

    return df_lab.to_dict("records")


def transpose_anno(lab_result):
    """ Transpost annotations from list of dict to dict of list """

    if not lab_tools.if_result_match_lab_format(lab_result):
        print("[pandas_data] WARNING: No lab format result found")
        return None

    empty_anno = lab_tools._lab_annotation_dic()
    anno_t = {}
    for key in empty_anno:
        if key == "labinfo":
            continue
        if key not in anno_t.keys():
            anno_t[key] = []
        for anno in lab_result["annotations"]:
            found = False
            for anno_key in anno.keys():
                if anno_key == key:
                    anno_t[key].append(anno[anno_key])
                    found = True
            if not found:
                anno_t[key].append(None)
                cound = False

    return anno_t


def lab_anno_to_df(lab_result):
    """ Create Pandas.DataFrame of annotations
        0.00093 on gc to run one result
    """

    input_anno_t = transpose_anno(lab_result)
    if input_anno_t is None:
        return None

    anno_df = pd.DataFrame.from_dict(input_anno_t)
    return anno_df


def create_anno_df_and_concat(lab_results_list, debug=True):
    """ Create DataFrame from a list of lab_result
        0.00422 seconds on gc2 to concat three results
        Note: the performance can be bad if annotations are big
        > 100ms for AIKEA results
    """

    df_list = []
    df_key_list = []
    for i in range(0, len(lab_results_list)):
        result = lab_results_list[i]
        df = lab_anno_to_df(result)
        if is_empty_df(df) and debug:
            print('[pandas_data] WARNING: empty DataFrame detected')
            continue

        df_list.append(df)

        if not isinstance(result["filename"], str):
            df_key_list.append(str(i))
        elif len(result["filename"]) < 1:
            df_key_list.append(str(i))
        else:
            df_key_list.append(result["filename"])

        if len(df_list) == 0:
            print('[pandas_data] WARNING: no DataFrame concated')
            return None

    concat_df = pd.concat(df_list, keys=df_key_list)
    return concat_df


def group_df(df, groups, comp_rule='mean'):
    """ Group DataFrame based on the max value of target mean

    @param df: input DataFrame
    @groups: list of keys used for groupby (e.g. ['label', 'id'])

    Keyword arguments:
        comp_rule -- mean to compare the mean, none else to compare sum
    """

    if isinstance(groups, str) or isinstance(groups, int):
        groups = [groups]

    if not isinstance(groups, list):
        print('[pandas_data] input groups is not str, int or list.')
        return None

    if comp_rule == 'off':
        mean_df = df.groupby(groups)
    elif comp_rule == 'mean':
        mean_df = df.groupby(groups).mean()
    else:
        mean_df = df.groupby(groups).sum()

    return mean_df


def select_item_from_target_values(
        df, groups, sel_name, target, filter_rule='max', comp_rule='mean'):
    """ Select item value from DataFrame based on the max value of target mean

    @param df: input DataFrame
    @groups: list of keys used for groupby (e.g. ['label', 'id'])
    @sel_name: target value you want to select (label, track_id, etc)
    @target: the target which will be mean (e.g. confidence)

    Keyword arguments:
        filter_rule -- max to select max in mean, else to select min in mean
        comp_rule -- mean to compare the mean, else to compare sum

    0.00226 seconds on gc2 to group and get mean
    0.00014 seconds on gc2 to cal max
    0.00057 seconds on gc2 to extra target

    """

    mean_df = group_df(
        df, groups, comp_rule=comp_rule
    )

    if filter_rule == 'max':
        filter_value = mean_df[target].max()
    else:
        filter_value = mean_df[target].min()

    sel_index = groups.index(sel_name)
    selected_value = mean_df.loc[
        mean_df[target] == filter_value].index.values[0][sel_index]

    return selected_value, filter_value


def norm_df(raw_df, exclude=None):
    """Normalize pandas DataFrame

    @param raw_df: raw input dataframe

    Keyword arguments:
    exclude -- a list of columns to be excluded

    """

    if exclude is not None:
        excluded = raw_df[exclude]
        _r = raw_df.drop(exclude, axis=1)
        _r = (_r - _r.mean()) / (_r.max() - _r.min())
        return pd.merge(excluded, _r)
    else:
        return (raw_df - raw_df.mean()) / (raw_df.max() - raw_df.min())


def filter_df_col(df, key, filter_value):
    """Filter DataFrame by column values

    @param df: input DataFrame
    @param key: key to be filtered
    @param filter_value: value to be selected
                         (should match key type)
    """
    return df.loc[df[key] == filter_value]


def export_df_csv(df, csvfile="./df.csv"):
    """Export DataFrame to csv"""

    df.to_csv(csvfile)
    return True


def conv_csv_df(csvfile, target=0):
    """Convert csv file to dataframe

    @param csvfile: file name of the csv to be read

    Keyword arguments:

    target   -- target column (default: 0)

    """
    return pd.DataFrame.from_csv(csvfile)


def conv_to_df(array, ffields=None, target=None):
    """Convert array to pandas.DataFrame

    @param array: input array to be converted

    Keyword arguments:
    ffields -- json file of the fields (default: None)
    target  -- if ffields is specified, can also specified
               the target column to be used (default: None)

    """
    if ffields is not None:
        fields = data.parse_json(ffields)
        if isinstance(target, int):
            print('[pandas_data] Converting field from %s to target'
                  % fields[target])
            fields[target] = 'target'
        return pd.DataFrame(array, columns=fields)
    return pd.DataFrame(array)


def df_header(df):
    """Get the header of the DataFrame as a list"""

    header = df.columns.values.tolist()
    print('[pandas_data] DataFrame header:')
    print(header)
    return header


def read_json_to_df(fname, orient='columns', np=False):
    """Read json file as pandas DataFrame

    @param fname: input filename

    Keyword arguments:
    orient -- split/records/index/columns/values (default: 'columns')
    np     -- true to direct decoding to numpy arrays (default: False)
    @return pandas DataFranm

    """
    if tools.check_exist(fname):
        return pd.read_json(fname, orient=orient, numpy=np)


def read_jsons_to_df(flist, orient='columns', np=False):
    """Read json files as one pandas DataFrame

    @param fname: input file list

    Keyword arguments:
    orient -- split/records/index/columns/values (default: 'columns')
    np     -- true to direct decoding to numpy arrays (default: False)
    @return concated pandas DataFranm

    """
    dfs = []
    for f in flist:
        dfs.append(read_json_to_df(f, orient=orient, np=np))
    return pd.concat(dfs)


def write_df_json(self, df, fname='df.json'):
    """Wtite pandas.DataFrame to json output"""

    df.to_json(fname)
    print('[pandas_data] DataFrame is written to %s' % fname)


def conv_to_np(array):
    """Convert DataFrame or list to np.ndarray"""

    if type(array) in [DataFrame, Series]:
        return array.as_matrix()

    if isinstance(array, list):
        return np.array(array)

    if tools.is_np(array):
        return array

    print("[pandas_data] WARNING: the type of input array is not correct!")
    print(type(array))
    return array


def conv_csv_svmft(csvfile, target=0, ftype=float, classify=True):
    """Convert csv file to SVM format

    @param csvfile: file name of the csv to be read

    Keyword arguments:

    target   -- target column (default: 0)
    ftype    -- convert data to the type (default: None)
    classify -- true convert target to int type (default: True)

    """
    indata = tools.read_csv(csvfile, ftype=ftype)
    df = conv_to_df(indata)

    _data = df.drop(df.columns[[target]], axis=1)
    data = conv_to_np(_data)
    target = conv_to_np(df[target])

    write_svmft(target, data, classify=classify)


def write_svmft(target, data, classify=True,
                fname='./data.svmft'):
    """Output data with the format libsvm/wusvm accepts

    @param target: array of the target (1D)
    @param data: array of the data (multi-dimensional)

    Keyword arguments:
    classify -- true convert target to int type (default: True)
    fname    -- output file name (default: ./data.svmft)

    """

    length = data.check_len(target, data)
    if classify:
        target = conv_to_np(target)
        target = target.astype(int)

    with open(fname, 'w') as outf:
        for i in range(0, length):
            output = []
            output.append(str(target[i]))
            for j in range(0, len(data[i])):
                output.append(str(j + 1) + ':' + str(data[i][j]))
            output.append('\n')
            libsvm_format = ' '.join(output)
            outf.write(libsvm_format)


def append_data_to_df(df, row_name, column_name, to_append):
    """ let append data in DataFrame can act like a function"""
    df.loc[row_name, column_name] = to_append
    return(df)


def _integer_generator():
    """
        A generator that acts like range(n), n = infinity. This generator is
        writed for the purpose of automatically generate column name of
        DataFrame.
    """
    n = 0
    while True:
        yield n
        n += 1


def _record_time_in_df(event_name):
    """
       A generator that stores the DataFrame, time, and column name of
       record_time_in_df().
    """
    df = pd.DataFrame()
    gen = tools._record_time()
    col = _integer_generator()
    next(gen)
    column_name = yield

    while True:
        if column_name is None:
            column_name = next(col)
        column_name = (yield append_data_to_df(df, event_name,
                                               column_name,
                                               to_append=next(gen)))


def record_time_in_df(action=None,
                      gen=_record_time_in_df('event'),
                      **kwargs):
    """
       Call this function two times, and at the second call, this function
       would automatically output a DataFrame that records the time passing
       between two calling. Moreover, if call this function three times, it
       would automatically output a DataFrame that records the time passing
       of first to second call and second to third call.

       Example:
               record_time_in_df()
               code1
               record_time_in_df() ==> output                  0
                                               event  code1_time
               code2
               record_time_in_df() ==> output                  0           1
                                               event  code1_time  code2_time

       The default row name of DataFrame is event, and we can specify row name
       by specify a generator.

       Example:
               recorder1 = record_time_in_df(event_name='re1')
               code1
               record_time_in_df(gen=recorder1,  ==> output               0
                                 action='start')            re1  code1_time

       The default column name of DataFrame is from 0, and every time append
       a column plus 1. We can specify column name by set column name when
       record.

       Example:
              record_time_in_df()
              code
              record_time_in_df(column_name='run') ==> output              run
                                                              event  code_time

       Note: The first time call record_time_in_df() can not set column_name,
             or we will get error.

    """

    if action == 'start':
        event_name = kwargs.get('event_name', 'event')
        gen = _record_time_in_df(event_name)
        next(gen)
        return gen

    else:
        column_name = kwargs.get('column_name', None)

        return gen.send(column_name)


def drop_continuous_duplicates(df):
    '''
        function that remove continuous duplicates.
        If you want to remove all duplicats, you should
        use df.drop_duplicats().

        example:
        df >>    a b
              1  1 1
              2  1 1
              3  2 2
              4  2 2
              5  1 1

        drop_continuous_duplicates(df) >>    a b
                                          1  1 1
                                          3  2 2
                                          5  1 1
    '''
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


def is_pandas_df(test_object):

    return isinstance(test_object, DataFrame)
