"""
@File    : data_nor.py
@Time    : 2025/3/5 15:23
@Author  : mingyang.zong
"""
import copy



class DataNormalization:
    """This class mainly integrates various data standardization methods.
    """

    def __int__(self):
        pass

    def min_max_nor_1d(self, data):
        min_ = min(data)
        max_ = max(data)
        data_nor = [(x - min_) / (max_ - min_) for x in data]
        return min_, max_, data_nor

    def inverse_min_max_nor(self, ori_min, ori_max, data_nor):
        data = [(x * (ori_max - ori_min) + ori_min) for x in data_nor]
        return data

    def min_max_nor_multi(self, col_list, data_pd):
        max_list, min_list = [], []
        temp = copy.deepcopy(data_pd)
        temp_columns = col_list
        for i in range(len(temp_columns)):
            temp_array = temp[temp_columns[i]].values
            new_one = self.min_max_nor_1d(temp_array)
            min_list.append(new_one[0])
            max_list.append(new_one[1])
            temp[temp_columns[i]] = new_one[2]
        return min_list, max_list, temp
