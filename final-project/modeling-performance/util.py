"""
Utility functions
"""
from copy import copy
from cStringIO import StringIO
import os
import re
import sys

from IPython.display import Image
import numpy as np
import pandas as pd
import pydot
from sklearn.tree import export_graphviz


# common data
UCS = ['date_stamp', 'park_sv_id', 'ab_total', 'ab_count', 'pitcher_id',
       'batter_id', 'ab_id', 'des', 'type', 'sz_top', 'sz_bot',
       'mlbam_pitch_name', 'zone_location', 'stand', 'strikes', 'balls',
       'p_throws', 'pdes', 'spin', 'norm_ht', 'inning', 'pfx_x', 'pfx_z',
       'x0', 'y0', 'z0', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'start_speed',
       'px', 'pz']
DTYPES = {'ab_total': np.uint8, 'ab_count': np.uint8,
          'pitcher_id': np.int64, 'batter_id': np.int64, 'ab_id': np.uint8,
          'des': str, 'type': str, 'sz_top': np.float64,
          'sz_bot': np.float64, 'mlbam_pitch_name': str,
          'zone_location': np.float64, 'stand': str,
          'strikes': np.uint8, 'balls': np.uint8,
          'p_throws': str, 'pdes': str, 'spin': np.float64,
          'norm_ht': np.float64, 'inning': np.uint8, 'pfx_x': np.float64,
          'pfx_z': np.float64, 'x0': np.float64, 'y0': np.float64,
          'z0': np.float64, 'vx0': np.float64, 'vy0': np.float64,
          'vz0': np.float64, 'ax': np.float64, 'ay': np.float64,
          'az': np.float64, 'start_speed': np.float64, 'px': np.float64,
          'pz': np.float64}
VALID_PTS = {'CH', 'CU', 'FC', 'FF', 'FT', 'SI', 'SL'}
FACT_COLS = ('pitcher_id', 'batter_id', 'park', 'type',
             'mlbam_pitch_name', 'zone_location',
             'stand', 'p_throws', 'des', 'pdes')

def fdate(date_str):
    """
    Parse a date string into a datetime object
    :param date_str: date string to parse
    :type date_str: str
    :return: datetime.datetime object
    """
    return pd.to_datetime(date_str, errors='coerce')


def fmt_brooks(brooks_path, min_matchups=30, dest='brooks'):
    """
    Preprocess Brooks dataset.
    :param brooks_path: full path to Brooks dataset
    :type brooks_path: str
    :param min_matchups: minimum number of events per matchup
    :type min_matchups: int
    :param dest: folder to output CSVs into
    :type dest: str
    :return: None (write CSVs)
    """
    try:
        os.mkdir(dest)
    except OSError:
        pass
    convs = {'date_stamp': fdate}
    df = pd.read_csv(brooks_path, converters=convs, usecols=UCS,
                     dtype=DTYPES, engine='c')
    filtered_matchups = df[(df.zone_location.notnull()) & \
        (df.pdes.notnull()) & (df.mlbam_pitch_name.isin(VALID_PTS))].groupby(
        ['batter_id', 'pitcher_id']).filter(lambda g: len(g) >= min_matchups)
    del df
    filtered_matchups['park'] = filtered_matchups.park_sv_id.map(lambda s: s[-3:])
    filtered_matchups.drop('park_sv_id', axis=1, inplace=True)
    for year, grp in filtered_matchups.groupby(
        lambda i: filtered_matchups.date_stamp[i].year):
        fname = os.path.join(dest, 'brooks-{}.csv'.format(year))
        write_factorized(grp, fname, FACT_COLS)



def write_factorized(df, fname, to_factorize):
    """
    Factorize categorial variables and then save to CSV
    :param df: dataframe to process
    :type df: pandas.DataFrame object
    :param fname: filename to write to
    :type fname: str
    :param to_factorize: names of columns to factorize
    :type to_factorize: sequence of str
    :return: None (writes CSV)
    """
    for col in to_factorize:
        df.loc[:, col] = factorize(df, col)
    df.to_csv(fname, index=False)


def factorize(df, col):
    """
    Factorize categorical column
    :type df: dataframe to process
    :param df: pandas.DataFrame object
    :param col: name of column to factorize
    :type col: str
    :return: pandas.DataFrame object
    """
    df['factorized_{}'.format(col)] = pd.factorize(df[col])[0]
    return df


def fac_keys(df, col):
    return dict(zip(*pd.factorize(df[col])))


def read_brooks(year, data_dir='brooks', max_devs=3):
    """
    Read a preprocessed Brooks data CSV by year
    :param year: year to retrieve data for
    :type year: int
    :param data_dir: directory containing CSVs
    :type data_dir: str
    :param max_devs: maximum number of multiples of a column's standard
                     deviation to restrict outliers to
    :type max_devs: int
    :return: pandas.DataFrame
    """
    dt = copy(DTYPES)
    dt['zone_location'] = np.uint8
    df = pd.read_csv(os.path.join(data_dir, 'brooks-{}.csv'.format(year)),
                     dtype=dt, converters={'date_stamp': fdate})
    dc = discrete_cols(df)
    for col in dc:
        df = remove_outliers(df, col, max_devs)
    return df


def corr(df, col_names=None, col_pat=None, type_='X'):
    """
    Calculate correlation with a `type_` column. Columns can either be
    listed out in full or selected with a regular expression.
    :param df: pandas dataframe to use in calculation
    :type df: pandas.DataFrame object
    :param col_names: names of columns to use in calculations
    :type col_names: sequence of str
    :param col_pat: regex pattern match against column names
    :type col_pat: str
    :param type_: event type to correlate against
    :param type_: str (one of 'S', 'X', or 'B')
    :return: pandas.Series
    """
    type_col = df['type_{}'.format(type_)]
    if col_name:
        target_cols = df[col_names]
    elif col_pat:
        target_cols = df[[c for c in df.columns if re.match(col_pat, c)]]
    else:
        raise ValueError('must supply col_name or col_pat')
    return target_cols.corrwith(type_col)


def is_discrete(col):
    """
    Test whether a column represents a discrete (non-binary, non-categorical)
    value.
    :param col: column or series to test
    :type col: pandas.Series object
    :return: bool
    """
    return col.dtype == np.float64 and not col.isin({0,1}).all()


def discrete_cols(df):
    """
    Get all columns in a dataset that represent discrete values
    :param df: dataframe
    :type df: pandas.DataFrame
    :return: list of str
    """
    return [c for c in df.columns if is_discrete(df[c])]



def remove_outliers(df, col_name, max_devs=3):
    """
    Remove rows from a dataset where the absolute value of a given column is
    more than a certain number of standard deviations away from the mean
    :param df: dataframe to filter
    :type df: pandas.DataFrame object
    :param col_name: name of column to filter on
    :type col_name: str
    :param max_devs: maximum number of standard deviations from the mean to
                     permit
    :return: pandas.DataFrame object
    """
    col = df[col_name]
    std = col.std()
    mean = col.mean()
    return df[(np.abs(col - mean)) <= (std * max_devs)]


def show_tree(model):
    dot_data = StringIO()
    export_graphviz(model, 
                    out_file=dot_data, 
                    feature_names=X.columns, 
                    filled=True, rounded=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    return Image(graph[0].create_png())



if __name__ == '__main__' and len(sys.argv) > 1:
    fmt_brooks(sys.argv[1])

