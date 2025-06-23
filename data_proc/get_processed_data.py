"""
@File    : get_processed_data.py
@Time    : 2025/3/8 11:02
@Author  : mingyang.zong
"""
from datetime import datetime
import pandas as pd
import numpy as np
import functools
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


class GetUsefulData:
    def __init__(self, data_path):
        self.data_path = data_path

    @log_execution_time
    def get_useful_data(self):
        """make June to September and December to February as useful month.
        data need with time index
        """
        data = self.get_sys_data().set_index('time')
        data.index = pd.to_datetime(data.index)
        all_month = sorted(set(list(data.index.strftime('%Y-%m'))))
        valid_month = all_month[0:2] + all_month[4:7] + all_month[9:14] + all_month[16:19] + \
                      all_month[21:26] + all_month[28:30] + all_month[33:38] + all_month[40:]

        logging.info(f'get useful month data..........')
        use_month = [x for x in valid_month if x not in ['2018-05', '2019-05', '2020-05']]
        useful_data = data['2022':'2022']
        for m in use_month:
            useful_data = pd.concat([useful_data, data[m:m]])
        return useful_data

    @log_execution_time
    def get_sys_data(self):
        # read original data
        logging.info(f'read original data..........')
        system_df = pd.read_excel(self.data_path, sheet_name=0)
        logging.info(f'get original data..........')
        temp_system = copy.deepcopy(system_df)
        # rename the columns
        logging.info(f'rename the columns of original data..........')
        temp_system.columns = ['time', 'mode', 'dry_tmp', 'wet_tmp', 'r_hum', 'T1', 'T2', 'F1', 'States', 'Y']
        # normalized time column
        logging.info(f'normalized time column..........')
        temp_system['time'] = temp_system['time'].apply(
            lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:8] + ' ' + str(x)[8:10] + ':' + str(x)[10:])
        # convert all values ​​in the'time' column to integer multiples of 10min
        logging.info(f"convert all values ​​in the'time' column to integer multiples of 10min..........")
        processed_time = []
        for i in range(len(temp_system['time'])):
            year = pd.to_datetime(temp_system['time'][i]).year
            month = pd.to_datetime(temp_system['time'][i]).month
            day = pd.to_datetime(temp_system['time'][i]).day
            hour = pd.to_datetime(temp_system['time'][i]).hour
            minute = pd.to_datetime(temp_system['time'][i]).minute
            if minute == 59:
                hour = hour + 1
                if hour == 24:
                    day = day + 1
                    hour = 0
                minute = 0
                processed_time.append(datetime(year, month, day, hour, minute).strftime("%Y-%m-%d %H:%M"))
            else:
                minute = round(minute / 10) * 10
                processed_time.append(datetime(year, month, day, hour, minute).strftime("%Y-%m-%d %H:%M"))
        temp_system['time'] = processed_time
        # 
        logging.info(f"Data deduplication..........")
        temp_system.drop_duplicates(subset=['time'], keep='last', inplace=True)
        logging.info(f"Missing timestamp filling..........")
        temp_system = self.complete_timestamp(temp_system)
        # 
        logging.info(f"Preliminary treatment of abnormal values..........")
        temp_system = self.adp_1(temp_system)
        return temp_system

    @log_execution_time
    def complete_timestamp(self, temp_system):
        """data need with time index"""
        temp_system_value = temp_system.values.tolist()
        standard_time = pd.date_range('2017-08-04 11:00', '2021-02-08 09:50', freq='10min').strftime(
            "%Y-%m-%d %H:%M").tolist()
        total_df = []
        system_time = temp_system['time'].values.tolist()
        while standard_time:
            if system_time[0] == standard_time[0]:
                total_df.append(temp_system_value.pop(0))
                system_time.pop(0)
                standard_time.pop(0)
            else:
                total_df.append([standard_time.pop(0)] + [np.nan] * 9)
        df = pd.DataFrame(total_df, columns=temp_system.columns)
        return df

    @log_execution_time
    def adp_1(self, data):
        """data need with time columns"""
        df = data.set_index('time')
        columns = df.columns
        for i in range(len(columns)):
            if columns[i] in ['dry_tmp', 'wet_tmp', 'T1', 'T2', 'Y']:
                df[columns[i]] = np.where((df[columns[i]] == 0), np.nan, df[columns[i]])
            elif columns[i] in ['r_hum']:
                df[columns[i]] = np.where((df[columns[i]] < 10), np.nan, df[columns[i]])
            elif columns[i] in ['Y']:
                df[columns[i]] = np.where((df[columns[i]] > 50000), np.nan, df[columns[i]])
            else:
                pass
        df = df.reset_index()
        return df
