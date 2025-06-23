"""
@File    : feature_engineer.py
@Time    : 2025/3/7 21:41
@Author  : mingyang.zong
"""
import time
import copy
import logging
import functools
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime
from datetime import timedelta
from workalendar.asia import China
from chinese_calendar import is_workday

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

TimeFreq = Enum('time_freq', (
    'Minute',
    'Hour',
    'Day',
    'Week',
    'Month',
    'Year'
))


def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        logging.info(f'{func.__name__} took {end - start} s')
        return res

    return wrapper


class AddFeatures:
    def __init__(self, data, add_workday=False):
        """load data with timestamp index
        :param data: pd.DataFrame, with timestamp index
        """
        self.data = data
        self.add_workday = add_workday

    def get_all_feature(self):
        pass

    @log_execution_time
    def add_basic_time_features(self, data, add_name=''):
        # 
        df = copy.deepcopy(data)
        df.index = pd.to_datetime(df.index)
        df['year' + add_name] = df.index.year
        df['month' + add_name] = df.index.month
        df['day' + add_name] = df.index.day
        df['day_of_year' + add_name] = df.index.dayofyear
        df['week_of_year' + add_name] = df.index.weekofyear
        df['quarter' + add_name] = df.index.quarter
        df['season' + add_name] = df.month % 12 // 3 + 1
        df['week_day' + add_name] = df.index.weekday
        df['hour_of_day' + add_name] = df.index.hour
        df['minute' + add_name] = df.index.minute
        df['month_sin' + add_name] = np.sin(2 * np.pi * df.month / 12)
        df['month_cos' + add_name] = np.cos(2 * np.pi * df.month / 12)
        df['week_sin' + add_name] = np.sin(2 * np.pi * df.week_of_year / 52)
        df['week_cos' + add_name] = np.cos(2 * np.pi * df.week_of_year / 52)
        df['weekday_sin' + add_name] = np.sin(2 * np.pi * df.week_day / 7)
        df['weekday_cos' + add_name] = np.cos(2 * np.pi * df.week_day / 7)
        df['hour_sin' + add_name] = np.sin(2 * np.pi * df.hour_of_day / 24)
        df['hour_cos' + add_name] = np.cos(2 * np.pi * df.hour_of_day / 24)
        df['minute_sin' + add_name] = np.sin(2 * np.pi * df.index.minute / 60)
        df['minute_cos' + add_name] = np.cos(2 * np.pi * df.index.minute / 60)
        df['if_weekend' + add_name] = [1 if x > 4 else 0 for x in list(df.index.weekday)]
        df['if_workday' + add_name] = [self.judge_workday(x) for x in df.index]
        df.drop(['year' + add_name], axis=1, inplace=True)
        if self.add_workday:
            df.fillna(0, inplace=True)
            return df
        else:
            df.drop(['if_workday' + add_name], axis=1, inplace=True)
            df.fillna(0, inplace=True)
            return df

    @log_execution_time
    def add_next_day_basic_time_features(self, ori_data, next_date, freq='10min'):
        time_to_add = pd.date_range(pd.to_datetime(next_date), pd.to_datetime(next_date) + timedelta(days=1),
                                    freq=freq, closed='left')  # .strftime('%Y-%m-%d %H:%M:%S').to_list()
        ori_index = ori_data.index
        ori_data.index = ori_data.index[len(time_to_add):].to_list() + time_to_add.to_list()
        temp = self.add_basic_time_features(ori_data, add_name='_next_day')
        temp.index = ori_index
        return temp

    def add_diff_feature(self, data, col_list):
        df = copy.deepcopy(data)
        # 
        for col in col_list:
            df[f'{col}_diff_1'] = df[col].diff()
        # df[f'{col}Y_diff_1'] = df['Y'].diff()
        # df['F1_diff_1'] = df['F1'].diff()
        # # States
        # df['States_sin'] = np.sin(2 * np.pi * df.States / 24)
        # df['States_cos'] = np.cos(2 * np.pi * df.States / 24)
        return df

    # 
    def judge_workday(self, date):
        """
        Judge one day if is a workday.
        :param date: datetime, eg.datetime(2021, 2, 7)
        :return: 0 or 1
        """
        if is_workday(date):
            return 1
        else:
            return 0

    @log_execution_time
    def add_sta_features(self, ts_df, pred_col="y_h", k=2, m=1):
        temp = ts_df[pred_col]
        sta_dict = {f"y_mean_k{k}_m{m}": [], f"y_median_k{k}_m{m}": [],
                    f"y_max_k{k}_m{m}": [], f"y_min_k{k}_m{m}": [], f"y_mean-std_k{k}_m{m}": [],
                    f"y_mean+std_k{k}_m{m}": []}
        for i in range(len(temp)):
            if i < (k * 144 + m):
                sta_dict[f"y_mean_k{k}_m{m}"].append(np.nan)
                sta_dict[f"y_median_k{k}_m{m}"].append(np.nan)
                sta_dict[f"y_max_k{k}_m{m}"].append(np.nan)
                sta_dict[f"y_min_k{k}_m{m}"].append(np.nan)
                sta_dict[f"y_mean-std_k{k}_m{m}"].append(np.nan)
                sta_dict[f"y_mean+std_k{k}_m{m}"].append(np.nan)
            else:
                values = []
                for j in range(1, k + 1):
                    temp_values = temp.iloc[i - j * 144 - m:i - j * 144 + m + 1].values.flatten()
                    values.append(temp_values)
                values = np.array(values).flatten()
                sta_dict[f"y_mean_k{k}_m{m}"].append(np.mean(values))
                sta_dict[f"y_median_k{k}_m{m}"].append(np.median(values))
                sta_dict[f"y_max_k{k}_m{m}"].append(np.max(values))
                sta_dict[f"y_min_k{k}_m{m}"].append(np.min(values))
                sta_dict[f"y_mean-std_k{k}_m{m}"].append(np.mean(values) - np.std(values))
                sta_dict[f"y_mean+std_k{k}_m{m}"].append(np.mean(values) + np.std(values))
        sta_pd = pd.DataFrame(sta_dict)
        sta_pd.index = temp.index
        ts_df = pd.concat([ts_df, sta_pd], axis=1)
        return ts_df

    # 
    @log_execution_time
    def add_front_sta_features(self, col, k, m):
        df = copy.deepcopy(self.data)
        temp = df[[col]]
        sta_dict = {f"y_front_mean_k{k}_m{m}": [], f"y_front_median_k{k}_m{m}": [],
                    f"y_front_max_k{k}_m{m}": [], f"y_front_min_k{k}_m{m}": [],
                    f"y_front_mean-std_k{k}_m{m}": [], f"y_front_mean+std_k{k}_m{m}": []}
        for i in range(len(temp)):
            if i < (k * 144 + m):
                sta_dict[f"y_front_mean_k{k}_m{m}"].append(np.nan)
                sta_dict[f"y_front_median_k{k}_m{m}"].append(np.nan)
                sta_dict[f"y_front_max_k{k}_m{m}"].append(np.nan)
                sta_dict[f"y_front_min_k{k}_m{m}"].append(np.nan)
                sta_dict[f"y_front_mean-std_k{k}_m{m}"].append(np.nan)
                sta_dict[f"y_front_mean+std_k{k}_m{m}"].append(np.nan)
            else:
                values = []
                for j in range(1, k + 1):
                    temp_values = temp.iloc[i - j * 144 - m:i - j * 144 + 1].values.flatten()
                    values.append(temp_values)
                values = np.array(values).flatten()
                sta_dict[f"y_front_mean_k{k}_m{m}"].append(np.mean(values))
                sta_dict[f"y_front_median_k{k}_m{m}"].append(np.median(values))
                sta_dict[f"y_front_max_k{k}_m{m}"].append(np.max(values))
                sta_dict[f"y_front_min_k{k}_m{m}"].append(np.min(values))
                sta_dict[f"y_front_mean-std_k{k}_m{m}"].append(np.mean(values) - np.std(values))
                sta_dict[f"y_front_mean+std_k{k}_m{m}"].append(np.mean(values) + np.std(values))
        sta_pd = pd.DataFrame(sta_dict)
        sta_pd.index = temp.index
        final_data = pd.concat([df, sta_pd], axis=1)
        return final_data

    # 
    @log_execution_time
    def add_back_sta_features(self, col, k, m):
        df = copy.deepcopy(self.data)
        temp = df[[col]]
        sta_dict = {f"y_back_mean_k{k}_m{m}": [], f"y_back_median_k{k}_m{m}": [],
                    f"y_back_max_k{k}_m{m}": [], f"y_back_min_k{k}_m{m}": [],
                    f"y_back_mean-std_k{k}_m{m}": [], f"y_back_mean+std_k{k}_m{m}": []}
        for i in range(len(temp)):
            if i < (k * 144):
                sta_dict[f"y_back_mean_k{k}_m{m}"].append(np.nan)
                sta_dict[f"y_back_median_k{k}_m{m}"].append(np.nan)
                sta_dict[f"y_back_max_k{k}_m{m}"].append(np.nan)
                sta_dict[f"y_back_min_k{k}_m{m}"].append(np.nan)
                sta_dict[f"y_back_mean-std_k{k}_m{m}"].append(np.nan)
                sta_dict[f"y_back_mean+std_k{k}_m{m}"].append(np.nan)
            else:
                values = []
                for j in range(1, k + 1):
                    temp_values = temp.iloc[i - j * 144:i - j * 144 + m + 1].values.flatten()
                    values.append(temp_values)
                values = np.array(values).flatten()
                sta_dict[f"y_back_mean_k{k}_m{m}"].append(np.mean(values))
                sta_dict[f"y_back_median_k{k}_m{m}"].append(np.median(values))
                sta_dict[f"y_back_max_k{k}_m{m}"].append(np.max(values))
                sta_dict[f"y_back_min_k{k}_m{m}"].append(np.min(values))
                sta_dict[f"y_back_mean-std_k{k}_m{m}"].append(np.mean(values) - np.std(values))
                sta_dict[f"y_back_mean+std_k{k}_m{m}"].append(np.mean(values) + np.std(values))
        sta_pd = pd.DataFrame(sta_dict)
        sta_pd.index = temp.index
        final_data = pd.concat([df, sta_pd], axis=1)
        return final_data


def calc_time_freq(ts_df, time_col):
    '''
    calculate time frequency from timestamp column values.

    Parameters
    ----------
    ts_df : DataFrame
        time series dataframe
    time_col : str
        timestamp column

    Returns
    -------
    tuple
        time frequency defined in core.const.TimeFreq & time delta.
    '''

    timedelta = int(ts_df[time_col].diff(
    ).value_counts().index[0].total_seconds())
    if timedelta < 60 * 60:
        freq, delta = TimeFreq.Minute, timedelta // 60
    elif timedelta < 60 * 60 * 24:
        freq, delta = TimeFreq.Hour, timedelta // 60 // 60
    elif timedelta < 60 * 60 * 24 * 7:
        freq, delta = TimeFreq.Day, timedelta // 60 // 60 // 24
    elif timedelta < 60 * 60 * 24 * 7 * 4:
        freq, delta = TimeFreq.Week, timedelta // 60 // 60 // 24 // 7
    elif timedelta < 60 * 60 * 24 * 364:
        freq, delta = TimeFreq.Month, timedelta // 60 // 60 // 24 // 28
    else:
        freq, delta = TimeFreq.Year, timedelta // 60 // 60 // 24 // 364

    return freq, delta


def fix_negative_values(values):
    """
    set negative values to positive.

    Parameters
    ----------
    values : ndarray
        one-dimensional numpy array

    Returns
    -------
        no negative values

    """
    a = values.copy()
    if a.min() < 0:
        a[a < 0] += -a.min()
    return a


def get_holidays(df, time_col, holiday):
    '''get holiday feature of dataframe'''
    time_feature = []
    calc = China()
    holiday_list = []
    df_time = df[time_col].tolist()
    for i in range(len(df_time)):
        if calc.is_working_day(df_time[i]):
            holiday_list.append(0)
        else:
            holiday_list.append(1)
    df['is_weekday'] = holiday_list
    time_feature.append('is_weekday')
    return df, time_feature


def create_time_feature(df, time_col):
    df['month'] = df[time_col].dt.month
    df['dayofweek'] = df[time_col].dt.dayofweek
    df['hour'] = df[time_col].dt.hour
    df['minute'] = df[time_col].dt.minute
    df['day'] = df[time_col].dt.day
    df['yday'] = df[time_col].dt.dayofyear
    time_feature = ['month', 'dayofweek', 'hour', 'day', 'minute']
    return df, time_feature


def create_history_feature(df, moment_name, pred_col, holiday, col_add_str=''):
    df_copy = df.copy()
    if holiday:
        days = 7
    else:
        days = 7
    df_res, feature_names = create_statistical_feature(df_copy, moment_name, pred_col, history_days=days,
                                                       col_add_str=col_add_str)
    df_res.sort_index(inplace=True)
    return df_res, feature_names


def create_hour_statistical_feature(df, pred_col, hours=7):
    df_temp = df.copy()
    groups = []
    feature_name = []
    for name, group in df_temp.groupby('hour'):
        for s in ['mean', 'max', 'min', 'median', 'std']:
            group['hour_day_' + s] = group[pred_col].rolling(hours, min_periods=1).agg(s)
            feature_name.append('hour_day_' + s)
        groups.append(group)
    feature_names = list(set(feature_name))
    group_df = pd.concat([*groups], axis=0)
    group_df.sort_index(inplace=True)
    return group_df, feature_names


def create_statistical_feature(df, moment, pred_col, history_days, col_add_str=''):
    df_temp = df.copy()
    groups = []
    feature_name = []
    for name, group in df_temp.groupby(moment):
        for k in range(2, history_days):
            for s in ['mean', 'max', 'min', 'median', 'std']:
                group[f'{col_add_str}moment_day_{k}_{s}'] = group[pred_col].rolling(history_days, min_periods=1).agg(s)
                feature_name.append(f'{col_add_str}moment_day_{k}_{s}')
        groups.append(group)
    feature_names = list(set(feature_name))
    group_df = pd.concat([*groups], axis=0)
    group_df.sort_index(inplace=True)
    return group_df, feature_names


def create_moment_feature(df, time_col):
    df['moment'] = df[time_col].dt.time
    return df, ['moment']


def create_power_price_feature(df):
    power_price = []
    for i in range(len(df)):
        if df.loc[i, 'hour'] >= 8 and df.loc[i, 'hour'] < 11 or df.loc[i, 'hour'] >= 18 and df.loc[i, 'hour'] < 21:
            power_price.append(0)
        elif df.loc[i, 'hour'] >= 22 and df.loc[i, 'hour'] < 24 or df.loc[i, 'hour'] >= 0 and df.loc[i, 'hour'] < 6:
            power_price.append(2)
        elif df.loc[i, 'hour'] >= 13 and df.loc[i, 'hour'] < 15 and df.loc[i, 'month'] >= 7 and df.loc[i, 'month'] <= 9:
            power_price.append(0)
        else:
            power_price.append(1)
    df['power_price'] = power_price
    return df, ['power_price']


def split_day_period(x):
    if x >= 6 and x < 8:
        return 1
    elif x >= 8 and x < 17:
        return 2
    elif x >= 17 and x < 18:
        return 3
    else:
        return 0


def create_day_period(df):
    df['day_period'] = df['hour'].apply(lambda x: split_day_period(x))
    return df, ['day_period']


def create_day_period_rank_reindex(df):
    df_copy = df.copy()
    groups = []
    for name, group in df_copy.groupby(['month', 'day', 'day_period']):
        group_copy = group.copy()
        group_copy.reset_index(inplace=True)
        if group_copy['day_period'][0] == 0:
            group['rank_reindex'] = 0
            groups.append(group)
            continue
        group_copy.sort_values(by=['value'], inplace=True)
        group['rank_reindex'] = group_copy.index.tolist()
        groups.append(group)
    feature_names = ['rank_reindex']
    group_df = pd.concat([*groups], axis=0)
    group_df.sort_index(inplace=True)
    return group_df, feature_names


def get_raio_features(ts_df, pred_col, col_add_str=''):
    groups = []
    for name, group in ts_df.groupby('dayofweek'):
        group_copy = group.copy()
        group_copy[f'{col_add_str}lastWeekVal'] = group_copy[pred_col].shift(144)
        group_copy[f'{col_add_str}PreWeekVal'] = group_copy[pred_col].shift(144 * 2)
        group_copy[f'{col_add_str}LastWeekByPreWeek'] = group_copy[f'{col_add_str}lastWeekVal'].fillna(0) / \
                                                        (group_copy[f'{col_add_str}PreWeekVal'].fillna(0) + 1e-5)
        group_copy[f'{col_add_str}Last2WeeksAverage'] = ((group_copy[f'{col_add_str}lastWeekVal'] +
                                                          group_copy[f'{col_add_str}PreWeekVal'])
                                                         / 2.0).fillna(group_copy[f'{col_add_str}lastWeekVal'])
        group_copy[f'{col_add_str}LastWeekByMean'] = group_copy[f'{col_add_str}lastWeekVal'] / ts_df[pred_col].mean()
        group_copy[f'{col_add_str}LastWeekByMax'] = group_copy[f'{col_add_str}lastWeekVal'] / ts_df[pred_col].max()
        group_copy[f'{col_add_str}lastWeekByMin'] = group_copy[f'{col_add_str}lastWeekVal'] / (
                    ts_df[pred_col].min() + 1e-5)
        groups.append(group_copy)
    df_group = pd.concat([*groups], axis=0)
    df_group.sort_index(inplace=True)
    return df_group


def get_percentage_feature(ts_df, pred_col, col_add_str=''):
    groups = []
    for name, group in ts_df.groupby(['month', 'day']):
        group_temp = group.copy()
        group_temp[f'{col_add_str}moment_ratio'] = group_temp[pred_col].apply(lambda x: x / group_temp[pred_col].sum())
        groups.append(group_temp)
    group_df = pd.concat([*groups], axis=0)
    group_df.sort_index(inplace=True)

    group2 = []
    '''ratio of 1/2/3 days before now'''
    for name, group in group_df.groupby('moment'):
        group[f'{col_add_str}moment_ratio_1'] = group[f'{col_add_str}moment_ratio'].shift(1)
        group[f'{col_add_str}moment_ratio_2'] = group[f'{col_add_str}moment_ratio'].shift(2)
        group[f'{col_add_str}moment_ratio_3'] = group[f'{col_add_str}moment_ratio'].shift(3)
        group2.append(group)
    group_res = pd.concat([*group2], axis=0)
    group_res.sort_index(inplace=True)
    return group_res


def get_yesterday_type(ts_df):
    df_copy = ts_df.copy()
    df_copy.reset_index(inplace=True)
    calc = China()
    day_type = []
    for i in range(len(ts_df)):
        if calc.is_working_day(
                datetime(2020, int(df_copy.loc[i, 'month']), int(df_copy.loc[i, 'day'])) - timedelta(days=1)):
            day_type.append(0)
        else:
            day_type.append(1)
    ts_df['is_yesterday_weekday'] = day_type
    return ts_df


def judge_holiday(x):
    calc = China()
    if calc.is_working_day(x):
        return 0
    else:
        return 1


def create_weekday_feature(ts_df, time_col, new_col):
    '''create is_weekday feature, cut down time consuming'''
    df_copy = ts_df.copy()
    df_copy['date'] = df_copy[time_col].dt.date
    df_list = []
    for name, group in df_copy.groupby('date'):
        if judge_holiday(name) == 0:
            group[new_col] = [0 for _ in range(len(group))]
        else:
            group[new_col] = [1 for _ in range(len(group))]
        df_list.append(group)
    df_res = pd.concat([*df_list], axis=0)
    df_res.sort_index(inplace=True)
    ts_df[new_col] = df_res[new_col].tolist()
    return ts_df


def feature_engineer(ts_df, time_col, pred_col, holiday):
    ts_df, time_features = create_time_feature(ts_df, time_col)
    ts_df, moment_name = create_moment_feature(ts_df, time_col)
    ts_df, statistical_features = create_history_feature(ts_df, moment_name, pred_col, holiday)
    ts_df, day_period = create_day_period(ts_df)
    ts_df = get_percentage_feature(ts_df, pred_col)
    ts_df = get_raio_features(ts_df, pred_col)
    ts_df = create_hour_rolling_feature(ts_df, pred_col, 3)
    return ts_df


def intra_feature_engineer(ts_df, time_col, pred_col, holiday, col_add_str='intra_'):
    ts_df, time_features = create_time_feature(ts_df, time_col)
    ts_df, moment_name = create_moment_feature(ts_df, time_col)
    ts_df, statistical_features = create_history_feature(ts_df, moment_name, pred_col, holiday, col_add_str=col_add_str)
    ts_df, day_period = create_day_period(ts_df)
    ts_df = get_percentage_feature(ts_df, pred_col, col_add_str=col_add_str)
    ts_df = get_raio_features(ts_df, pred_col, col_add_str=col_add_str)
    ts_df = create_hour_rolling_feature(ts_df, pred_col, 3, col_add_str=col_add_str)
    return ts_df


def day_ahead_feature_engineer(ts_df, time_col, pred_col, holiday, col_add_str='day_ahead'):
    ts_df, time_features = create_time_feature(ts_df, time_col)
    ts_df, moment_name = create_moment_feature(ts_df, time_col)
    ts_df, statistical_features = create_history_feature(ts_df, moment_name, pred_col, holiday, col_add_str=col_add_str)
    ts_df = get_percentage_feature(ts_df, pred_col, col_add_str=col_add_str)
    ts_df = get_raio_features(ts_df, pred_col, col_add_str=col_add_str)
    ts_df = create_hour_rolling_feature(ts_df, pred_col, 3, col_add_str=col_add_str)
    return ts_df


def add_hour_sin_feature(df):
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['wday_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 6)
    df['wday_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 6)
    df['yday_sin'] = np.sin(2 * np.pi * df['yday'] / 365)
    df['yday_cos'] = np.cos(2 * np.pi * df['yday'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.sin(2 * np.pi * df['month'] / 12)
    return df


def create_hour_rolling_feature(df, pred_col, window=3, delay=0, col_add_str=''):
    df_copy = df.copy()
    if delay > 0:
        df_copy[f'{col_add_str}point_delay'] = df_copy.shift(delay)
    else:
        df_copy[f'{col_add_str}point_delay'] = df_copy[pred_col]
    for i in range(1, window + 1):
        df_copy[f'{col_add_str}{i}_hour_shift'] = df_copy[f'{col_add_str}point_delay'].shift(i * 4)
        for s in ['max', 'min', 'mean', 'median', 'std']:
            df[f'{col_add_str}{i}_hour_{s}'] = df_copy[f'{col_add_str}point_delay'].rolling(i * 4 + 1,
                                                                                            min_periods=1).agg(s)
    df[f'{col_add_str}lastMinuteRatio'] = df_copy[f'{col_add_str}point_delay'] / (
                df_copy[f'{col_add_str}point_delay'].shift(1).fillna(0) + 1e-5)
    df[f'{col_add_str}lastMinuteDiff'] = df_copy[f'{col_add_str}point_delay'].diff(1)
    return df


def create_index(feature_df, cv=5):
    import random
    random.seed(123)
    index_random = []
    feature_df['date'] = feature_df['time'].dt.date
    if 'time' in feature_df.columns.tolist():
        time_col_index = 'time'
    else:
        time_col_index = 'index'
    for name, group in feature_df.groupby('date'):
        list_temp = list(range(len(group)))
        random.shuffle(list_temp)
        list_temp = [i % cv for i in list_temp]
        index_random.extend((list_temp))
    feature_df['i_set'] = index_random
    feature_df.drop(['date'], axis=1, inplace=True)
    feature_df.set_index(time_col_index, drop=True, inplace=True)
    return feature_df


@log_execution_time
def get_intraday_fe(ori_intraday_x, time_col='time', pred_col='Y', if_workday=1):
    holiday = False if if_workday else True
    df = feature_engineer(ori_intraday_x, time_col, pred_col, holiday).drop('moment', axis=1)
    df.set_index(time_col, inplace=True)
    df.interpolate(inplace=True)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df


# @log_execution_time
# def get_intraday_fe(ori_intraday_x, time_col='time', pred_col='Y', if_workday=1):
#     # add features
#     holiday = False if if_workday else True
#     df = intra_feature_engineer(ori_intraday_x, time_col, pred_col, holiday).drop('moment', axis=1)
#     df.set_index(time_col, inplace=True)
#     df.interpolate(inplace=True)
#     df = df.fillna(method='ffill').fillna(method='bfill')
#     amb_col = df.columns[:7]
#     df.columns = ['intra_' + n if n in amb_col else n for n in df.columns]
#     return df


# @log_execution_time
# def get_day_ahead_fe(ori_intraday_x, time_col='time', pred_col='Y', if_workday=1):
#     # add features
#     holiday = False if if_workday else True
#     df = day_ahead_feature_engineer(ori_intraday_x, time_col, pred_col, holiday).drop(['moment', 'month',
#                                                                                        'dayofweek', 'yday',
#                                                                                        'hour', 'day', 'minute'], axis=1)
#     df.set_index(time_col, inplace=True)
#     df.interpolate(inplace=True)
#     df = df.fillna(method='ffill').fillna(method='bfill')
#     amb_col = df.columns[:7]
#     df.columns = ['day_ahead_' + n if n in amb_col else n for n in df.columns]
#     return df

@log_execution_time
def get_day_ahead_fe(day_ahead_x, k=(1, 2, 3), m=(1, 1, 1), time_col='time', pred_col='Y'):
    # add features
    day_ahead_x.set_index(time_col, inplace=True)
    day_ahead_x.index = pd.to_datetime(day_ahead_x.index)
    df = copy.deepcopy(day_ahead_x)
    # df = AddFeatures(df).add_basic_time_features(df)
    for l in range(len(k)):
        df = AddFeatures(df).add_sta_features(df, pred_col=pred_col, k=k[l], m=m[l])
        df = AddFeatures(df).add_front_sta_features(pred_col, k=k[l], m=m[l])
        df = AddFeatures(df).add_back_sta_features(pred_col, k=k[l], m=m[l])
    df.interpolate(inplace=True)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df
