import os, copy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from chinese_calendar import is_workday
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data_proc.abnormal_process import get_chebyshev_index, process_abnornal_value

parent_dir = os.path.dirname(os.path.abspath(__file__))
from workalendar.asia import China
from data_proc.data_filling import get_all_filled_data
from data_proc.data_smoothing import get_smoothed_data
from feature_engineer.fe import get_intraday_fe, get_day_ahead_fe


def judge_workday(date):
    """
    Judge one day if is a workday.
    :param date: datetime, eg.datetime(2021, 2, 7)
    :return: 0 or 1
    """
    if is_workday(date):
        return 1
    else:
        return 0


def get_mode_status(x):
    if x.month in [12, 1, 2]:
        return 1
    else:
        return 0


def get_processed_df(forecast_time: datetime, original_df: pd.DataFrame, target_column="Y", add_mode=False):
    # judge the mode of load
    mode = get_mode_status(forecast_time)

    original_df["mode"] = original_df["time"].apply(lambda x: get_mode_status(x))

    load_df = copy.deepcopy(original_df)
    load_df["is_working"] = load_df["time"].apply(lambda x: judge_workday(x))

    forecast_type = judge_workday(forecast_time)

    load_df["month"] = load_df["time"].apply(lambda x: x.month)
    load_df = load_df[(load_df["month"] == 1) | (load_df["month"] == 2) | (load_df["month"] == 12)]

    load_df = load_df[(load_df["time"] < datetime(2018, 2, 15)) | (load_df["time"] >= datetime(2018, 2, 22))]

    load_df = load_df[(load_df["time"] < datetime(2019, 2, 4)) | (load_df["time"] >= datetime(2019, 2, 11))]

    load_df = load_df[(load_df["time"] < datetime(2020, 1, 23)) | (load_df["time"] >= datetime(2020, 3, 1))]

    load_df = load_df[(load_df["mode"] == mode) & (load_df["is_working"] == forecast_type)]
    load_df.index = [_ for _ in range(len(load_df))]

    # get the abnormal index
    abnormal_index = get_chebyshev_index(load_df)

    # abnormal process
    processed_df = process_abnornal_value(load_df, abnormal_index)
    if add_mode:
        ts_df = processed_df[["time", "mode", target_column]]
    else:
        ts_df = processed_df[["time", target_column]]

    return ts_df


def feature_engineer(ts_df, time_col, pred_col):
    ts_df, time_features = create_time_feature(ts_df, time_col)

    ts_df, moment_name = create_moment_feature(ts_df, time_col)

    ts_df, statistical_features = create_history_feature(ts_df, moment_name, pred_col)

    ts_df, day_period = create_day_period(ts_df)

    ts_df = get_percentage_feature(ts_df, pred_col)

    ts_df = get_lastday_load(ts_df, pred_col)

    ts_df = get_hist_static_features(ts_df, pred_col, k=2, m=2)

    ts_df = get_hist_static_features(ts_df, pred_col, k=3, m=1)

    ts_df = add_hour_sin_feature(ts_df)

    ts_df = create_minute_rolling_feature(ts_df, pred_col, window=6, delay=0)

    ts_df["moment"] = LabelEncoder().fit_transform(ts_df["moment"].values)

    return ts_df


def get_cnn_lstm_input(forecast_time: datetime, original_df: pd.DataFrame, hyp,
                       pre_col="Y", forecast_horizon=144):
    processed_df = get_processed_df(forecast_time, original_df, pre_col)

    # shift to get the history load
    processed_df["load"] = processed_df[pre_col].shift(forecast_horizon).values
    all_load_value = processed_df[pre_col].values
    processed_df.drop(pre_col, axis=1, inplace=True)

    ts_df = feature_engineer(processed_df, time_col="time", pred_col="load")

    ts_df.insert(1, pre_col, all_load_value)
    ts_df = ts_df.iloc[forecast_horizon * 4:]
    ts_df = ts_df.interpolate()

    # get the train and validation dataset
    feat_cols = [col for col in ts_df.columns if col not in ["time", pre_col]]
    x_df = ts_df[feat_cols]
    x_data = x_df.values
    x_scaler = MinMaxScaler()
    x_scaled_data = x_scaler.fit_transform(x_data)

    # select the y data and standard
    max_load = np.max(ts_df[pre_col].values[:-forecast_horizon])
    original_load = ts_df[pre_col].values
    y_scaled_data = original_load / max_load
    assert len(x_scaled_data) == len(y_scaled_data)
    assert len(y_scaled_data) % forecast_horizon == 0

    seq_len = hyp["seq_len"]
    pre_len = copy.deepcopy(forecast_horizon)

    test_x = x_scaled_data[-seq_len:]
    test_y = y_scaled_data[-pre_len:]

    test_x = np.array(test_x, dtype=np.float32)
    test_y = np.array(test_y, dtype=np.float32)
    # reshape
    test_x = np.reshape(test_x, (1, seq_len, -1))
    test_y = np.reshape(test_y, (1, pre_len))

    seq_x = []
    seq_y = []
    index = copy.deepcopy(seq_len)
    while index <= len(x_scaled_data) - pre_len:
        sample_x = x_scaled_data[(index - seq_len): index]
        sample_y = y_scaled_data[(index - pre_len): index]
        # update the input data
        seq_x.append(sample_x.tolist())
        seq_y.append(sample_y.tolist())
        # update the sample index
        index += pre_len
    # transform to array
    seq_x = np.array(seq_x, dtype=np.float32)
    seq_y = np.array(seq_y, dtype=np.float32)
    seq_y = np.reshape(seq_y, (-1, pre_len))

    train_ratio, val_ratio = 0.9, 0.1
    train_index = int(len(seq_x) * train_ratio)

    # get the train, validate, test set
    train_x, train_y = seq_x[:train_index], seq_y[:train_index]
    val_x, val_y = seq_x[train_index:], seq_y[train_index:]
    print("Model inputs is got. ")
    return train_x, train_y, val_x, val_y, test_x, test_y, max_load


def get_enhance_input(forecast_time: datetime, original_df: pd.DataFrame, look_back=144 * 3,
                      pre_col="Y", forecast_horizon=144):
    ts_df = get_processed_df(forecast_time, original_df, pre_col)

    ts_df = ts_df.interpolate()

    # get the train and validation dataset
    y_scaler = StandardScaler()
    data = ts_df[pre_col].values
    data = np.reshape(data, (-1, 1))
    scaled_data = y_scaler.fit_transform(data)
    assert len(scaled_data) % forecast_horizon == 0

    test_x = scaled_data[-(look_back + forecast_horizon):-forecast_horizon]
    test_y = scaled_data[-forecast_horizon:]

    test_x = np.array(test_x, dtype=np.float32)
    test_y = np.array(test_y, dtype=np.float32)
    # reshape
    test_x = np.reshape(test_x, (1, look_back, 1))
    test_y = np.reshape(test_y, (1, forecast_horizon, 1))

    seq_x = []
    seq_y = []
    index = copy.deepcopy(look_back + forecast_horizon)
    while index <= len(scaled_data) - forecast_horizon:
        sample_x = scaled_data[(index - look_back - forecast_horizon): (index - forecast_horizon)]
        sample_y = scaled_data[(index - forecast_horizon): index]
        # update the input data
        seq_x.append(sample_x.tolist())
        seq_y.append(sample_y.tolist())
        # update the sample index
        index += forecast_horizon
    # transform to array
    seq_x = np.array(seq_x, dtype=np.float32)
    seq_y = np.array(seq_y, dtype=np.float32)
    seq_x = np.reshape(seq_x, (-1, look_back, 1))
    seq_y = np.reshape(seq_y, (-1, forecast_horizon, 1))

    train_ratio, val_ratio = 0.9, 0.1
    train_index = int(len(seq_x) * train_ratio)

    # get the train, validate, test set
    train_x, train_y = seq_x[:train_index], seq_y[:train_index]
    val_x, val_y = seq_x[train_index:], seq_y[train_index:]
    print("Model inputs is got. ")
    return train_x, train_y, val_x, val_y, test_x, test_y, y_scaler


############################## generate the features ##############################3
def create_time_feature(df, time_col):
    df['month'] = df[time_col].dt.month
    df['dayofweek'] = df[time_col].dt.dayofweek
    df['hour'] = df[time_col].dt.hour
    df['minute'] = df[time_col].dt.minute
    df['day'] = df[time_col].dt.day
    df['yday'] = df[time_col].dt.dayofyear
    time_feature = ['month', 'dayofweek', 'hour', 'day', 'minute']
    return df, time_feature


def create_history_feature(df, moment_name, pred_col):
    df_copy = df.copy()
    days = 6
    df_res, feature_names = create_statistical_feature(df_copy, moment_name, pred_col, days)
    df_res.sort_index(inplace=True)
    return df_res, feature_names


def create_statistical_feature(df, moment, pred_col, history_days):
    df_temp = df.copy()
    groups = []
    feature_name = []
    for name, group in df_temp.groupby(moment):
        for k in range(2, history_days):
            for s in ['mean', 'max', 'min', 'median', 'std']:
                if s == "std":
                    mean = group[pred_col].rolling(history_days, min_periods=1).agg("mean")
                    std = group[pred_col].rolling(history_days, min_periods=1).agg("std")
                    group[f"y_moment_day_{k}-{s}"] = mean - std
                    group[f"y_moment_day_{k}_{s}"] = mean + std
                    feature_name.append(f'y_moment_day_{k}_{s}')
                    feature_name.append(f'y_moment_day_{k}-{s}')
                else:
                    group[f'y_moment_day_{k}_{s}'] = group[pred_col].rolling(history_days, min_periods=1).agg(s)
                    feature_name.append(f'y_moment_day_{k}_{s}')
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
    if x >= 6 and x < 12:
        return 1
    elif x >= 12 and x < 14:
        return 2
    elif x >= 14 and x < 18:
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


def get_lastday_load(ts_df, pre_col):
    ts_df["lastday_" + pre_col] = ts_df[pre_col].shift(144)
    ts_df["last2day_" + pre_col] = ts_df[pre_col].shift(144 * 2)
    ts_df["lastday_delta"] = ts_df[pre_col] - ts_df["lastday_" + pre_col]
    return ts_df


def get_hist_static_features(ts_df, pred_col="Y", k=2, m=1):
    temp = ts_df[pred_col]
    sta_dict = {f"y_mean_k{k}_m{m}": [], f"y_median_k{k}_m{m}": [],
                f"y_max_k{k}_m{m}": [], f"y_min_k{k}_m{m}": [], f"y_std_k{k}_m{m}": [],
                f"y_std-k{k}_m{m}": []}
    for i in range(len(temp)):
        if i < (k * 144 + m):
            sta_dict[f"y_mean_k{k}_m{m}"].append(np.nan)
            sta_dict[f"y_median_k{k}_m{m}"].append(np.nan)
            sta_dict[f"y_max_k{k}_m{m}"].append(np.nan)
            sta_dict[f"y_min_k{k}_m{m}"].append(np.nan)
            sta_dict[f"y_std-k{k}_m{m}"].append(np.nan)
            sta_dict[f"y_std_k{k}_m{m}"].append(np.nan)
        else:
            values = []
            for j in range(k):
                values.extend(temp.iloc[i - j * 144 - m: i - j * 144 + m + 1].values.tolist())
            sta_dict[f"y_mean_k{k}_m{m}"].append(np.mean(values))
            sta_dict[f"y_median_k{k}_m{m}"].append(np.median(values))
            sta_dict[f"y_max_k{k}_m{m}"].append(np.max(values))
            sta_dict[f"y_min_k{k}_m{m}"].append(np.min(values))
            sta_dict[f"y_std-k{k}_m{m}"].append(np.mean(values) - np.std(values))
            sta_dict[f"y_std_k{k}_m{m}"].append(np.mean(values) + np.std(values))
    sta_pd = pd.DataFrame(sta_dict)
    sta_pd.index = temp.index
    ts_df = pd.concat([ts_df, sta_pd], axis=1)
    return ts_df


def get_percentage_feature(ts_df, pred_col):
    groups = []
    for name, group in ts_df.groupby(['month', 'day']):
        group_temp = group.copy()
        group_temp['moment_ratio'] = group_temp[pred_col].apply(lambda x: x / group_temp[pred_col].sum())
        groups.append(group_temp)
    group_df = pd.concat([*groups], axis=0)
    group_df.sort_index(inplace=True)

    group2 = []
    '''ratio of 1/2 days before now'''
    for name, group in group_df.groupby('moment'):
        group['moment_ratio_1'] = group['moment_ratio'].shift(1)
        group['moment_ratio_2'] = group['moment_ratio'].shift(2)
        group2.append(group)
    group_res = pd.concat([*group2], axis=0)
    group_res.sort_index(inplace=True)
    return group_res


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


def add_hour_sin_feature(df):
    df["minute_sin"] = np.sin(2 * np.pi * df['hour'] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df['hour'] / 60)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['wday_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 6)
    df['wday_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 6)
    df['yday_sin'] = np.sin(2 * np.pi * df['yday'] / 365)
    df['yday_cos'] = np.cos(2 * np.pi * df['yday'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.sin(2 * np.pi * df['month'] / 12)
    return df


def create_minute_rolling_feature(df, pred_col, window=6, delay=0):
    df_copy = df.copy()
    if delay > 0:
        df_copy['point_delay'] = df_copy[pred_col].shift(delay)
    else:
        df_copy['point_delay'] = df_copy[pred_col]
    for i in range(2, window + 1):
        df_copy[f'{i}_minute_shift'] = df_copy['point_delay'].shift(-i).fillna(method="bfill").fillna(method="ffill")
        for s in ['max', 'min', 'mean', 'median']:
            # 向前取统计值
            df[f'{i}_minute_{s}'] = df_copy['point_delay'].rolling(i + 1, min_periods=1).agg(s)
    df['lastMinuteRatio'] = df_copy['point_delay'] / (df_copy['point_delay'].shift(1).fillna(0) + 1e-5)
    df['lastMinuteDiff'] = df_copy['point_delay'].diff(1)
    return df

def get_use_data(ori_data, pred_col='Y'):
    temp_data = copy.deepcopy(ori_data)
    temp_data['time'] = pd.to_datetime(temp_data['time'])
    # add work label
    temp_data['if_workday'] = temp_data['time'].apply(lambda x: judge_workday(x))
    temp_data.set_index('time', inplace=True)

    # use_month = ["2017-08", "2017-09", "2018-06", "2018-07", "2018-08", "2018-09", "2019-06", "2019-07", "2019-08",
    #              "2019-09", "2020-06", "2020-07", "2020-08", "2021-06", "2021-07", "2021-08", "2021-09"]
    use_month = ["2017-12", "2018-01", "2018-02", "2018-12", "2019-01", "2019-02", "2019-12", 
                 "2020-01", "2020-02", "2020-12", "2021-01", "2021-02"]

    tmp_df = pd.DataFrame()
    
    temp_data_copy = copy.deepcopy(temp_data)
    temp_data_copy['ym'] = [tt.strftime("%Y-%m") for tt in temp_data.index]
    
    for m in use_month:
        tmp_df = pd.concat([tmp_df, temp_data_copy[temp_data_copy['ym'] == m]])
    
    tmp_df = tmp_df.drop('ym', axis=1)
    tmp_df[pred_col] = np.where((tmp_df[pred_col] == 0), np.nan, tmp_df[pred_col])
    tmp_df.reset_index(inplace=True)
    return tmp_df


def get_intra_xgb_input(target_time, data, if_workday=1, mode_type=1, pre_len=36, if_add_day_ahead_fe=True,
                        flag="train", time_field="time", pred_col='Y'):
    day_type = 1 if judge_workday(target_time) else 0
    # get all useful data
    add_data = data[(data["if_workday"] == day_type) & (data['time'] >= target_time)].iloc[:pre_len]
    end_time = add_data['time'].values.flatten()[-1]
    all_data = data[data['time'] <= end_time]
    # all_data = copy.deepcopy(data)
    # get all data before predict ending time
    if flag == "train":
        all_data = all_data
    else:
        all_data = all_data.iloc[-30 * 144:]
    all_data.set_index('time', inplace=True)
    # get filled data
    data_filled = get_all_filled_data(all_data)
    # abnormal values detection and handle
    data_abn = get_smoothed_data(data_filled)
    # load mode_type data
    type_data = data_abn[data_abn['mode'] == mode_type].drop(['mode'], axis=1)
    # get if_workday data
    if_workday_data = type_data[type_data['if_workday'] == if_workday].drop('if_workday', axis=1)
    # get ori_x ori_y
    if_workday_data.reset_index(inplace=True)
    ori_y = if_workday_data[['time', 'Y']].set_index('time')
    ori_y = pd.DataFrame(all_data.loc[ori_y.index, pred_col], columns=[pred_col]).interpolate()
    ori_x = copy.deepcopy(if_workday_data)
    # add features
    if if_add_day_ahead_fe:
        # get intraday feature
        ori_intraday_x = ori_x.shift(pre_len)
        ori_intraday_x['time'] = ori_x['time']
        add_fe_intraday_x = get_intraday_fe(ori_intraday_x, if_workday=if_workday)
        # get day-ahead feature
        ori_day_ahead_x = ori_x.shift(144)
        ori_day_ahead_x['time'] = ori_x['time']
        add_fe_day_ahead_x = get_day_ahead_fe(ori_day_ahead_x)
        add_fe_day_ahead_x.columns = ["day_ahead_" + str(n) for n in add_fe_day_ahead_x.columns]
        # get concat data without day-ahead real data **
        temp_data_x = pd.concat([add_fe_intraday_x, add_fe_day_ahead_x.iloc[:, 7:]], axis=1)
    else:
        # get intraday feature
        ori_intraday_x = ori_x.shift(pre_len)
        ori_intraday_x['time'] = ori_x['time']
        add_fe_intraday_x = get_intraday_fe(ori_intraday_x, if_workday=if_workday)
        temp_data_x = add_fe_intraday_x
    # get data_x data_y
    data_x = copy.deepcopy(temp_data_x.iloc[144 * 4:])
    data_y = ori_y.loc[data_x.index, pred_col]
    x_train, y_train = data_x.iloc[:-pre_len], data_y.iloc[:-pre_len]
    x_test, y_test = data_x.iloc[-pre_len:], data_y.iloc[-pre_len:]
    x_train.drop([pred_col], axis=1, inplace=True); x_test.drop([pred_col], axis=1, inplace=True)
    # return x_train, y_train, x_test, y_test
    if flag == "train":
        return x_train, y_train, x_test, y_test
    else:
        return x_test, y_test
