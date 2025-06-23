"""
@File    : data_split.py
@Time    : 2025/3/5 15:22
@Author  : mingyang.zong
"""
from sklearn.model_selection import train_test_split
import numpy as np


def CreateSample(data, look_back, look_after, feature_dim=None, label_dim=None,
                 label=None, time_skip=1, feature=None, decoder_dim=False, MinMax=True):
    dataX = list()
    dataY = list()
    labelY = list()
    featureX = list()
    decoder_info = list()
    if MinMax:
        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        print(data)
    count = 0
    if not (label_dim or label):
        print('dataset need label')
        return False
    for i in range(data.shape[0] - look_back - look_after + 1):
        if count % time_skip == 0:
            if feature_dim != None:
                dataX.append(data[:, feature_dim][i:i + look_back:])
            else:
                dataX.append(data[i:i + look_back:])
            if label_dim:
                dataY.append(data[:, label_dim][i + look_back:i + look_back + look_after:])
            if decoder_dim:
                decoder_info.append(data[:, decoder_dim][i + look_back:i + look_back + look_after:])
            if label:
                labelY.append(label[i + look_back:i + look_back + look_after:])
            if feature:
                featureX.append(feature[i:i + look_back:])
        count += 1

    if feature:
        dataX = np.stack((dataX, featureX), 2)
    if label_dim:
        if label:
            dataY = np.stack((labelY, dataY), 2)
    else:
        dataY = labelY
    if decoder_dim:
        return np.array(dataX), np.array(decoder_info), np.array(dataY)
    else:

        return np.array(dataX), np.array(dataY)


class DataSplit:
    def __init__(self, data_x, data_y, kind=2, random=True, size=[0, 0.2]):
        """
        """
        self.data_x = data_x
        self.data_y = data_y
        self.kind = kind
        self.random = random
        self.size = size

    def get_split_data(self):
        """Get split data by users' rules (kind and random).
        """
        if self.kind == 2 and self.random:
            return self.random_split_2()
        if self.kind == 2 and not self.random:
            return self.order_split2()
        if self.kind == 3 and self.random:
            return self.random_split_3()
        if self.kind == 3 and not self.random:
            return self.order_split3()

    def random_split_2(self):
        """Generate x_train, y_train, x_test, y_test,
        in proportion and random order.
        """
        x_train, x_test, y_train, y_test = train_test_split(self.data_x, self.data_y, test_size=self.size[1])
        return x_train, y_train, x_test, y_test

    def random_split_3(self):
        """Generate x_train, y_train, x_val, y_val, x_test, y_test,
        in proportion and random order.
        """
        X_train, x_test, Y_train, y_test = train_test_split(self.data_x, self.data_y, test_size=self.size[1])
        x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train,
                                                          test_size=(self.size[0] / (1 - self.size[1])))
        return x_train, y_train, x_val, y_val, x_test, y_test

    def order_split2(self):
        """Generate x_train, y_train, x_test, y_test,
        in proportion and order.
        """
        x_train, x_test, y_train, y_test = train_test_split(self.data_x, self.data_y, test_size=self.size[1],
                                                            shuffle=self.random)
        return x_train, y_train, x_test, y_test

    def order_split3(self):
        """Generate x_train, y_train, x_val, y_val, x_test, y_test,
        in proportion and order.
        """
        X_train, x_test, Y_train, y_test = train_test_split(self.data_x, self.data_y, test_size=self.size[1],
                                                            shuffle=self.random)
        x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train,
                                                          test_size=(self.size[0] / (1 - self.size[1])),
                                                          shuffle=self.random)
        return x_train, y_train, x_val, y_val, x_test, y_test
