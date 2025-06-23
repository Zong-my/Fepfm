"""
@File    : data_filling.py
@Time    : 2025/3/8 15:39
@Author  : mingyang.zong
"""
import pandas as pd
import functools
import datetime
import logging
import copy
import time

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)



def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        logging.info(f'{func.__name__} took {end - start} s')
        return res

    return wrapper


# get all filled columns
@log_execution_time
def get_all_filled_data(use_data):
    """This function is fixed to this load_prediction project
    :param use_data: pd.DataFrame, with time index
    :return: pd.DataFrame, filled data
    """
    ud_copy = copy.deepcopy(use_data)
    columns = ud_copy.columns
    logging.info(f'********** Start filling in data **********')
    endings = ['mode', 'States', 'precipitation','rain','snowfall','snow_depth','weather_code',
               'is_day','_duration','_radiation','_irradiance','_instant','if_workday']
    for col in columns:
        if any(col.endswith(end) for end in endings):
            ud_copy[col].fillna(method='ffill', inplace=True)
            ud_copy[col].fillna(method='bfill', inplace=True)
            logging.info(f'******** complete filling {col} ********')
        elif col in ['Y']:
            ud_copy[col] = DataFiller().get_filled_data(ud_copy[[col]]).values.flatten()
            logging.info(f'******** complete filling {col} ********')
        else:
            continue
    ud_copy.interpolate(inplace=True)
    logging.info(f'******** complete filling all columns! ********')
    return ud_copy


class DataFiller:
    def __init__(self, threshold=0.5):
        self.thre = threshold

    def get_filled_data(self, useful_data):

        useful_data.index = pd.to_datetime(useful_data.index)
        ud_copy = copy.deepcopy(useful_data)
        use_month = sorted(set(list(useful_data.index.strftime('%Y-%m'))))
        for month in use_month:

            temp_ud_copy = copy.deepcopy(ud_copy)
            temp_ud_copy['ym'] = [tt.strftime("%Y-%m") for tt in temp_ud_copy.index]
            tmp_ud_copy = temp_ud_copy[temp_ud_copy['ym'] == month]
            tmp_ud_copy = tmp_ud_copy.drop('ym', axis=1)

            self.fill_by_week(ud_copy, tmp_ud_copy)
        ud_copy.interpolate(inplace=True)
        return ud_copy


    def find_big_miss_day(self, useful_data):
        threshold = self.thre
        useful_data.index = pd.to_datetime(useful_data.index)
        temp_data = copy.deepcopy(useful_data)
        date_by_day = list(sorted(set([x.date().strftime('%Y-%m-%d') for x in temp_data.index])))
        miss_day = []
        for day in date_by_day:
            temp = temp_data[day:day]
            try:
                value = temp.isna().sum().values[-1] / len(temp)
            except:
                value = temp.isna().sum()[-1] / len(temp)
            if value > threshold:
                miss_day.append(day)
        return miss_day


    def judge_full(self, day_data):
        value = day_data.isna().sum().values[0] / len(day_data)
        return (True if value <= self.thre else False)


    def find_date(self, data, kind='day'):
        k = kind
        data.index = pd.to_datetime(data.index)
        if k == 'day':
            return (list(sorted(set([x.date().strftime('%Y-%m-%d') for x in data.index]))))
        if k == 'month':
            return (list(sorted(set([x.date().strftime('%Y-%m') for x in data.index]))))


    def fill_by_week(self, temp_data, month_data):
        temp_data.index = pd.to_datetime(temp_data.index)
        month_data.index = pd.to_datetime(month_data.index)
        big_miss_day = self.find_big_miss_day(month_data)
        all_day = self.find_date(temp_data)

        for bmd in big_miss_day:
            bmd_in_week = pd.to_datetime(bmd).weekday()
            left_day = all_day[:all_day.index(bmd)][::-1]
            for temp_day in left_day:
                
                temp_data_copy = copy.deepcopy(temp_data)
                temp_data_copy['ym'] = [tt.strftime("%Y-%m-%d") for tt in temp_data.index]
                temp_data_copy = temp_data_copy[temp_data_copy['ym']==temp_day]
                
                if (pd.to_datetime(temp_day).weekday() == bmd_in_week) & (self.judge_full(temp_data_copy)):
     
                    tmp_month_data = month_data[(month_data.index >= bmd)&(month_data.index < pd.to_datetime(bmd) + datetime.timedelta(1))]
                    tmp_data = temp_data[(temp_data.index >= temp_day)&(temp_data.index < pd.to_datetime(temp_day) + datetime.timedelta(1))]
                    if len(tmp_month_data) == len(tmp_data):
                        month_data[(month_data.index >= bmd)&(month_data.index < pd.to_datetime(bmd) + datetime.timedelta(1))] = tmp_data.values                        
                        # print(f"{bmd} filled by {temp_day}")
                        break
                    else:
                        continue
                else:
                    continue
