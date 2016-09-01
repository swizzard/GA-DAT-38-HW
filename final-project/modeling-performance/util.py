"""
Utility functions
"""
from copy import copy
from cStringIO import StringIO
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


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


def plot_confusion_matrix(true_labels, pred_labels, title, cmap=plt.cm.Blues):
    """
    Plot a confusion matrix. Adapted from 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    :param true_labels: true labels
    :type true_labels: pandas.DataFrame
    :param pred_labels: predicted labels
    :type true_labels: numpy.ndarray
    :param title: title of the figure
    :type title: str
    :param cmap: color map to use in the plot
    :type cmap: matplotlib.colors.LinearSegmentedColormap object
    :return: None
    """
    cm = confusion_matrix(true_labels, pred_labels)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(3)
    labels = ['B', 'S', 'X']
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_weights(model, cols):
    """
    Plot the relative weights assigned by a model to the columns in the data
    :param model: the model generating the weights
    :param cols: names of the features in the data
    :return: None
    """
    pd.DataFrame({'Features': cols,
                  'Importances': model.feature_importances_}).sort_values(
        'Importances', ascending=False).plot(kind='bar', x='Features',
                                             y='Importances')


def eval_model(true_labels, pred_labels, title):
    """
    Evaluate a model by calculating and printing its F1 score and
    plotting its confusion matrix
    """
    print '{} model F1 score: {}'.format(title, 
                                         f1_score(true_labels, pred_labels,
                                                  average='micro'))
    labels = ('B', 'S', 'X')
    true_pcts = [pct_of(true_labels, label) for label in labels]
    pred_pcts = [pct_of(pred_labels, label) for label in labels]
    print 'True B: {:%}\tTrue S: {:%}\tTrue X: {:%}'.format(
        *true_pcts)
    print 'Predicted B: {:%}\tPredicted S: {:%}\tPredicted X: {:%}'.format(
        *pred_pcts)
    plot_confusion_matrix(true_labels, pred_labels, title, plt.cm.Blues)


def pct_of(arr, val):
    """
    Calculate the % of values in an array matching a target value
    :param arr: array containing values
    :type arr: sequence
    :param val: value to match
    :type val: any
    :return: float
    """
    return float(len([item for item in arr if item == val])) / len(arr)


def preds_labels(df, cols):
    """
    Retrieve the predictors and labels from a dataframe
    :param df: dataframe to extract from
    :type df: pandas.DataFrame object
    :param cols: predictor columns
    :type cols: list of str
    :return: (pd.DataFrame, pd.Series)
    """
    return df[cols], df.type


def train_and_plot(model, train, test, pred_cols, name, **model_args):
    """
    Train and test a model and display the results
    :param model: model to train
    :param train_preds: training predictors
    :param train_y: training labels
    :param test_preds: test predictors
    :param test_y: test labels
    :param cmap: color map
    :kwargs model_args: keyword arguments to use when initializing the model
    :return: (fitted) model, predicted labels
    """
    mod = model(**model_args)
    train_x, train_y = preds_labels(train, pred_cols)
    mod.fit(train_x, train_y)
    test_x, test_y = preds_labels(test, pred_cols)
    pred_labels = mod.predict(test_x)
    eval_model(test_y, pred_labels, name)
    plot_weights(mod, pred_cols)
    return mod, pred_labels


def pad_training(df):
    ss = df[df.type == 'S']
    sl = len(ss)
    xs = df[df.type == 'X']
    xl = len(xs)
    x_samp = xs.sample(n=(sl - xl), replace=True)
    bs = df[df.type == 'B']
    bl = len(bs)
    b_samp = bs.sample(n=(sl - bl), replace=True)
    return pd.concat([df, b_samp, x_samp])


if __name__ == '__main__' and len(sys.argv) > 1:
    fmt_brooks(sys.argv[1])

