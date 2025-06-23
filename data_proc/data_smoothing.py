"""
@File    : data_smoothing.py
@Time    : 2025/3/8 17:39
@Author  : mingyang.zong
"""
import copy, time
import functools
import numpy as np
import pandas as pd
from datetime import timedelta
import logging
from tsmoothie.smoother import *

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


def exponentsmooth(data, window_len=20, alpha=0.3):
    """
    This function is to do Exponential smoothing.
    :param data: array of shape. Raw data received by the smoother.
    :param window_len: int. The length of the window used to compute the Exponential smoothing.
    :param alpha: float.
        Between 0 and 1. (1-alpha) provides the importance of the past obsevations
        when computing the smoothing.
    :return: array of shape. Smoothed data derived from the smoothing operation.
    """
    smoother = ExponentialSmoother(window_len=window_len, alpha=alpha)
    smoother.smooth(data)
    return smoother.smooth_data


def convolutionsmooth(data, window_len=20, window_type='ones'):
    """
    This function is to do Convolution smoothing.
    :param data: array of shape. Raw data received by the smoother.
    :param window_len: int. The length of the window used to compute the Convolution smoothing.
    :param window_type: str.
        The type of the window used to compute the convolutions.
        Supported types are: 'ones', 'hanning', 'hamming', 'bartlett', 'blackman'.
    :return: array of shape. Smoothed data derived from the smoothing operation.
    """
    smoother = ConvolutionSmoother(window_len=window_len, window_type=window_type)
    smoother.smooth(data)
    return smoother.smooth_data


def polynomialsmooth(data, degree=6):
    """
    This function is to do Polynomial smoothing.
    :param data: array of shape. Raw data received by the smoother.
    :param window_len: int. The length of the window used to compute the Polynomial smoothing.
    :param degree: int. The polynomial order used to build the basis.
    :return: array of shape. Smoothed data derived from the smoothing operation.
    """
    smoother = PolynomialSmoother(degree=degree)
    smoother.smooth(data)
    return smoother.smooth_data


def splinesmooth(data, n_knots=6, spline_type='natural_cubic_spline'):
    """
    This function is to do Spline smoothing.
    :param data: array of shape. Raw data received by the smoother.
    :param n_knots:int
        Between 1 and timesteps for 'linear_spline' and 'natural_cubic_spline'.
        Between 3 and timesteps for 'natural_cubic_spline'.
        Number of equal intervals used to divide the input
        space and smooth the timeseries. A lower value of n_knots
        will result in a smoother curve
    :param spline_type: str.
        Type of spline smoother to operate. Supported types are 'linear_spline',
        'cubic_spline' or 'natural_cubic_spline'
    :return: array of shape. Smoothed data derived from the smoothing operation.
    """
    smoother = SplineSmoother(n_knots=n_knots, spline_type=spline_type)
    smoother.smooth(data)
    return smoother.smooth_data


def gaussiansmooth(data, n_knots=6, sigma=0.1):
    """
    This function is to do Gaussian smoothing.
    :param data: array of shape. Raw data received by the smoother.
    :param n_knots: int.
        Between 1 and timesteps. Number of equal intervals used to divide the input
        space and smooth the timeseries. A lower value of n_knots
        will result in a smoother curve.
    :param sigma: int. The sigma value for gaussian kernel regression.
    :return: array of shape. Smoothed data derived from the smoothing operation.
    """
    smoother = GaussianSmoother(n_knots=n_knots, sigma=sigma)
    smoother.smooth(data)
    return smoother.smooth_data


def binnersmooth(data, n_knots=6):
    """
    This function is to do Binner smoothing.
    :param data: array of shape. Raw data received by the smoother.
    :param n_knots: int
        Between 1 and timesteps. Number of equal intervals used to divide the input
        space and smooth the timeseries. A lower value of n_knots
        will result in a smoother curve.
    :return:array of shape. Smoothed data derived from the smoothing operation.
    """
    smoother = BinnerSmoother(n_knots=n_knots)
    smoother.smooth(data)
    return smoother.smooth_data


def lowesssmooth(data, smooth_fraction=0.1, iterations=1):
    """
    This function is to do  Lowess smoothing.
    :param data: array of shape. Raw data received by the smoother.
    :param smooth_fraction: float
        Between 0 and 1. The smoothing span. A larger value of smooth_fraction
        will result in a smoother curve.
    :param iterations: int
        Between 1 and 6. The number of residual-based reweightings to perform.
    :return: array of shape. Smoothed data derived from the smoothing operation.
    """
    smoother = LowessSmoother(smooth_fraction=smooth_fraction, iterations=iterations)
    smoother.smooth(data)
    return smoother.smooth_data


def kalmansmoother(data, component='level_trend',
                   component_noise={'level': 0.1, 'trend': 0.1}):
    """
    This function is to do  Kalman smoothing.
    :param data: array of shape. Raw data received by the smoother.
    :param component: str
        Specify the patterns and the dinamycs present in our series.
        The possibilities are: 'level', 'level_trend',
        'level_season', 'level_trend_season', 'level_longseason',
        'level_trend_longseason', 'level_season_longseason',
        'level_trend_season_longseason'. Each single component is
        delimited by the '_' notation.
    :param component_noise: dict
        Specify in a dictionary the noise (in float term) of each single
        component provided in the 'component' argument. If a noise of a
        component, not provided in the 'component' argument, is provided, it's
        automatically ignored.
    :return: array of shape. Smoothed data derived from the smoothing operation.
    """
    smoother = KalmanSmoother(component=component, component_noise=component_noise)
    smoother.smooth(data)
    return smoother.smooth_data



def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        logging.info(f'{func.__name__} took {end - start} s')
        return res

    return wrapper


@log_execution_time
def get_smoothed_data(data, pred_col="Y"):
    fixed_col = ['mode', 'States']
    all_columns = list(data.columns)
    dc = [x for x in all_columns if x not in fixed_col]
    data.index = pd.to_datetime(data.index)
    data_copy = copy.deepcopy(data)
    logging.info(f'start abnormal values detection and handle.........')
    use_month = sorted(set(list(data_copy.index.strftime('%Y-%m'))))
    for col in dc:
        if col in ['Y']:
            proc_data = data_copy[[col]]['2022':'2022']
            for month in use_month:
                temp = pd.DataFrame(data_copy[col][month])
                temp_index = temp.index
                temp = _diff_smooth(temp.reset_index(drop=True), pred_col=pred_col)
                temp.index = temp_index
                proc_data = pd.concat([proc_data, temp])
            data_copy[col] = proc_data[col].values.flatten()
            logging.info(f'handled {col}.........')
        else:
            continue
    return data_copy



def _diff_smooth(data, coef=1.5, pred_col="Y"):
    """
    :param data: pd.Series
    :param coef: const, need to >=1
    :return: smoothed data
    """
    ts = copy.deepcopy(data)

    dif = ts.diff().fillna(0)  # .dropna()
    td = dif.describe()

    high = (td.T['75%'] + coef * (td.T['75%'] - td.T['25%'])).values.flatten()[0]
    low = (td.T['25%'] - coef * (td.T['75%'] - td.T['25%'])).values.flatten()[0]

    forbid_index = list(dif[(dif[pred_col] > high) | (dif[pred_col] < low)].index)
    i = 0
    while i < len(forbid_index):
        n = 1
        if (forbid_index[i] == len(dif) - 1) or (i + n >= len(forbid_index)):  #
            break

        start = forbid_index[i]  # 
        # while (forbid_index[i+n] == start + timedelta(minutes=n*10)) & (forbid_index[i+n] != forbid_index[-1]):
        while forbid_index[i + n] == start + n:
            n += 1
            if i + n >= len(forbid_index):
                break
            # print(i+n)
        i += n - 1
        end = forbid_index[i]  # 
        if (forbid_index[i] == len(dif) - 1) or (i + n >= len(forbid_index)):  # 
            break
        # print(f"start:{start},end:{end}")
        # 
        # print((start, end))
        value = np.linspace(ts.iloc[start - 1].values.flatten()[0], ts.iloc[end + 1].values.flatten()[0], n)
        ts.iloc[start: end + 1] = value.reshape(ts.iloc[start: end + 1].shape)
        i += 1

    return ts
