# -*- encoding: utf-8 -*-
'''
@File    :   gbdt_based_models.py
@Time    :   2025/04/18 12:17:18
@Author  :   myz 
'''
import os
import time
import shap
import pickle
import logging
import functools
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from loguru import logger
from sklearn.svm import SVR
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        logging.info(f'{func.__name__} took {end - start} s')
        return res
    return wrapper
# saved scaled data and function
def save_pickle(model, path_name):
    s = pickle.dumps(model)
    with open(path_name, 'wb+') as f:
        f.write(s)

# pickle trained model
def load_pickle(path_name):
    f = open(path_name, 'rb')  # 
    s = f.read()
    Model = pickle.loads(s)
    return Model


class XgbModel(object):
    def __init__(self, train_day=None, nths=-1, save_path=None):
        """
        init from params dict
        """
        self.cv = 2
        self.model = None
        self.features = []
        self.nan_count = 0
        self.train_day = train_day
        self.nthreads = nths
        self.save_path = save_path

    def create_cv_index(self, i_set, shuffle=False, seed=123):
        """
        Return the labels for k-fold cross-validation sets, ensuring that data published at the same time 
        are either all in the training set or the validation set during the split."""
        i_set_unique = np.unique(i_set)
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(i_set_unique)
        cv_index = np.ones_like(i_set, dtype=int) * -1
        for k in range(self.cv):
            for m in i_set_unique[k::self.cv]:
                cv_index[i_set == m] = k
        return cv_index

    def fit(self, x_train, y_train, x_val, y_val):
        x_train, y_train = pd.DataFrame(x_train), pd.DataFrame(y_train)
        x_val, y_val = pd.DataFrame(x_val), pd.DataFrame(y_val)
        x_train['i_set'] = self.create_cv_index(range(len(x_train)))
        i_fold = x_train[['i_set']]
        x_train.drop(['i_set'], axis=1, inplace=True)
        logger.info("features of train {}".format(x_train.columns.tolist()))
        grid_params = {
            'eta': (0.01, 0.1), 'max_depth': (2, 7),
            'min_child_weight': (0.3, 8), 'subsample': (0.3, 1),
            'lambdas': (0.1, 100)
        }

        def xgb_evaluate(eta, max_depth, min_child_weight, subsample, lambdas):
            params = {
                # 'booster': 'gblinear',
                # 'eval_metric': 'rmse',
                'max_depth': int(max_depth),
                'min_child_weight': float(min_child_weight),
                'subsample': float(subsample),
                'eta': float(eta),
                'lambda': float(lambdas),
                'seed': 1000,
                'nthread': self.nthreads
                # 'n_estimators': int(n_estimators),
            }
            scores = 0
            for i in range(self.cv):
                logger.info(f"training fold {i + 1}")
                dtrain = xgb.DMatrix(x_train.loc[i_fold[i_fold['i_set'] != i].index, :],
                                    y_train.loc[i_fold[i_fold['i_set'] != i].index, :], missing=0)
                dtest = xgb.DMatrix(x_train.loc[i_fold[i_fold['i_set'] == i].index, :],
                                    y_train.loc[i_fold[i_fold['i_set'] == i].index, :], missing=0)
                bst = xgb.train(params, dtrain, 4000,  # feval=xgb_mape,
                                evals=[(dtrain, 'train'), (dtest, 'eval')], early_stopping_rounds=5, verbose_eval=100)
                scores += bst.best_score
            # lntree = bst.best_ntree_limit
            # logger.info("lntree is :{}".format(lntree))
            return -1.0 * scores/self.cv

        xgb_bo = BayesianOptimization(xgb_evaluate, grid_params)
        # xgb_bo.maximize(init_points=20, n_iter=50)
        xgb_bo.maximize()

        bst_params = xgb_bo.max['params']
        logger.info("best param: {}".format(bst_params))
        bst_params['max_depth'] = int(bst_params['max_depth'])
        
        # train all data with best params
        dtrain = xgb.DMatrix(x_train, y_train, missing=0)
        dtest = xgb.DMatrix(x_val, y_val, missing=0)
        bst = xgb.train(bst_params, dtrain, 4000,
                        evals=[(dtrain, 'train'), (dtest, 'eval')], early_stopping_rounds=10, verbose_eval=100)
        # self.model[str(i + 1)] = bst
        self.model = bst
        # lntree = bst.best_ntree_limit
        importance_by_gain = sorted(bst.get_score(importance_type='gain').items(), key=lambda k: k[1], reverse=True)
        
        self.features = list(x_train.columns)
        self.importance = importance_by_gain
        self.bst_params = bst_params

        # logger.info("lntree number is :{}".format(lntree))
        logger.info("feature importance is :{}".format(importance_by_gain))
        
        # save features' importance 
        fi_save_path = os.path.join(self.save_path, 'xgb')
        os.makedirs(fi_save_path, exist_ok=True)
        f = open(os.path.join(fi_save_path, 'xgb_fi.txt'), "a")
        
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{formatted_datetime} \n {self.train_day}bst_params{bst_params}\n importance_by_gain{importance_by_gain}" + '\n')
        f.close()

    def predict(self, x_test):
        return self.model.predict(xgb.DMatrix(x_test[self.features]))

    # save trained model
    def save_model(self, path_name):
        s = pickle.dumps({'model':self.model, 
                          'features':self.features,
                          'importance':self.importance,
                          'bst_params':self.bst_params})
        with open(path_name, 'wb+') as f:
            f.write(s)

    # load trained model
    def load_model(self, path_name):
        f = open(path_name, 'rb')  #  
        s = f.read()
        Model = pickle.loads(s)
        return Model


class XgbModelGrid(object):
    def __init__(self, train_day=None, nths=-1, save_path=None):
        """
        init from params dict
        """
        self.cv = 2
        self.model = None
        self.features = []
        self.nan_count = 0
        self.train_day = train_day
        self.nthreads = nths
        self.save_path = save_path

    def create_cv_index(self, i_set, shuffle=False, seed=123):
        """
        Return the labels for k-fold cross-validation sets, ensuring that data published at the same time 
        are either all in the training set or the validation set during the split.
        """
        i_set_unique = np.unique(i_set)
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(i_set_unique)
        cv_index = np.ones_like(i_set, dtype=int) * -1
        for k in range(self.cv):
            for m in i_set_unique[k::self.cv]:
                cv_index[i_set == m] = k
        return cv_index

    def fit(self, x_train, y_train, x_val, y_val):
        x_train, y_train = pd.DataFrame(x_train), pd.DataFrame(y_train)
        x_val, y_val = pd.DataFrame(x_val), pd.DataFrame(y_val)
        x_train['i_set'] = self.create_cv_index(range(len(x_train)))
        i_fold = x_train[['i_set']]
        x_train.drop(['i_set'], axis=1, inplace=True)
        logger.info("features of train {}".format(x_train.columns.tolist()))
        
        # Define parameter grid for GridSearchCV
        param_grid = {
            'eta': [0.01, 0.05, 0.1],
            'max_depth': [2, 5, 7],
            'min_child_weight': [0.3, 5, 8],
            'subsample': [0.3, 0.7, 1.0],
            'lambda': [0.1, 10, 100]
        }

        best_score = float('inf')
        best_params = None

        # Perform grid search
        for params in itertools.product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            scores = 0
            for i in range(self.cv):
                logger.info(f"training fold {i + 1}")
                dtrain = xgb.DMatrix(x_train.loc[i_fold[i_fold['i_set'] != i].index, :],
                                    y_train.loc[i_fold[i_fold['i_set'] != i].index, :], missing=0)
                dtest = xgb.DMatrix(x_train.loc[i_fold[i_fold['i_set'] == i].index, :],
                                    y_train.loc[i_fold[i_fold['i_set'] == i].index, :], missing=0)
                bst = xgb.train(param_dict, dtrain, 4000,
                                evals=[(dtrain, 'train'), (dtest, 'eval')], early_stopping_rounds=5, verbose_eval=100)
                scores += bst.best_score
            avg_score = scores / self.cv
            if avg_score < best_score:
                best_score = avg_score
                best_params = param_dict

        logger.info("best param: {}".format(best_params))

        # Train all data with best params
        dtrain = xgb.DMatrix(x_train, y_train, missing=0)
        dtest = xgb.DMatrix(x_val, y_val, missing=0)
        bst = xgb.train(best_params, dtrain, 4000,
                        evals=[(dtrain, 'train'), (dtest, 'eval')], early_stopping_rounds=10, verbose_eval=100)
        self.model = bst
        importance_by_gain = sorted(bst.get_score(importance_type='gain').items(), key=lambda k: k[1], reverse=True)

        self.features = list(x_train.columns)
        self.importance = importance_by_gain
        self.bst_params = best_params

        logger.info("feature importance is :{}".format(importance_by_gain))

        # Save features' importance
        fi_save_path = os.path.join(self.save_path, 'xgb')
        os.makedirs(fi_save_path, exist_ok=True)
        f = open(os.path.join(fi_save_path, 'xgb_fi.txt'), "a")
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{formatted_datetime} \n {self.train_day}bst_params{best_params}\n importance_by_gain{importance_by_gain}" + '\n')
        f.close()

    def predict(self, x_test):
        return self.model.predict(xgb.DMatrix(x_test[self.features]))

    # save trained model
    def save_model(self, path_name):
        s = pickle.dumps({'model': self.model, 
                          'features': self.features,
                          'importance': self.importance,
                          'bst_params': self.bst_params})
        with open(path_name, 'wb+') as f:
            f.write(s)

    # load trained model
    def load_model(self, path_name):
        f = open(path_name, 'rb')  #  
        s = f.read()
        Model = pickle.loads(s)
        return Model
    

class LgbModel(object):
    def __init__(self, train_day=None, nths=-1, save_path=None):
        """
        init from params dict
        """
        self.cv = 2
        self.model = None
        self.features = []
        self.nan_count = 0
        self.train_day = train_day
        self.nthreads = nths
        self.save_path = save_path

    def create_cv_index(self, i_set, shuffle=False, seed=123):
        """
        Return the labels for k-fold cross-validation sets, ensuring that data published at the same time 
        are either all in the training set or the validation set during the split."""
        i_set_unique = np.unique(i_set)
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(i_set_unique)
        cv_index = np.ones_like(i_set, dtype=int) * -1
        for k in range(self.cv):
            for m in i_set_unique[k::self.cv]:
                cv_index[i_set == m] = k
        return cv_index

    def fit(self, x_train, y_train, x_val, y_val):
        x_train, y_train = pd.DataFrame(x_train), pd.DataFrame(y_train)
        x_val, y_val = pd.DataFrame(x_val), pd.DataFrame(y_val)
        x_train['i_set'] = self.create_cv_index(range(len(x_train)), shuffle=True)
        i_fold = x_train[['i_set']]
        x_train.drop(['i_set'], axis=1, inplace=True)
        logger.info("features of train {}".format(x_train.columns.tolist()))
        grid_params = {
            'learning_rate': (0.01, 0.3), 'max_depth': (2, 20),
            'min_child_samples': (1, 80), 'subsample': (0.01, 1.0),
            'lambda_l2': (0.1, 100)
        }

        def lgb_evaluate(learning_rate, max_depth, min_child_samples, subsample, lambda_l2):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'max_depth': int(max_depth),
                'min_child_samples': int(min_child_samples),
                'subsample': float(subsample),
                'learning_rate': float(learning_rate),
                'lambda_l2': float(lambda_l2),
                'early_stopping_round': 10,
                'saved_feature_importance_type':1, 
                'verbosity': -1,
                'seed': 1000,
                'nthread': self.nthreads
            }
            scores = 0
            for i in range(self.cv):
                logger.info(f"training fold {i + 1}")
                train_data = lgb.Dataset(x_train.loc[i_fold[i_fold['i_set'] != i].index, :],
                                        y_train.loc[i_fold[i_fold['i_set'] != i].index, :])
                val_data = lgb.Dataset(x_train.loc[i_fold[i_fold['i_set'] == i].index, :],
                                      y_train.loc[i_fold[i_fold['i_set'] == i].index, :])
                bst = lgb.train(params, train_data, num_boost_round=4000,
                                valid_sets=[train_data, val_data], valid_names=['train', 'eval'])
                scores += bst.best_score['eval']['rmse']
            return -1.0 * scores/self.cv

        lgb_bo = BayesianOptimization(lgb_evaluate, grid_params)
        lgb_bo.maximize()

        bst_params = lgb_bo.max['params']
        logger.info("best param: {}".format(bst_params))
        bst_params['max_depth'] = int(bst_params['max_depth'])
        bst_params['min_child_samples'] = int(bst_params['min_child_samples'])
        bst_params['early_stopping_rounds'] = 50
        bst_params['verbosity'] = -1
        
        # train all data with best params
        train_data = lgb.Dataset(x_train, y_train)
        val_data = lgb.Dataset(x_val, y_val)
        bst = lgb.train(bst_params, train_data, num_boost_round=4000,
                        valid_sets=[train_data, val_data], valid_names=['train', 'eval'])
        self.model = bst
        ori_gain_dict = dict(zip(list(x_train.columns), list(bst.feature_importance(importance_type='gain')))) 
        importance_by_gain = sorted(ori_gain_dict.items(), key=lambda item: item[1], reverse=True)

        self.features = list(x_train.columns)
        self.importance = importance_by_gain
        self.bst_params = bst_params

        logger.info("feature importance is :{}".format(importance_by_gain))
        
        # save features' importance 
        fi_save_path = os.path.join(self.save_path, 'lgb')
        os.makedirs(fi_save_path, exist_ok=True)
        f = open(os.path.join(fi_save_path, 'lgb_fi.txt'), "a")
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{formatted_datetime} \n {self.train_day}bst_params{bst_params}\n importance_by_gain{importance_by_gain}" + '\n')
        f.close()

    def predict(self, x_test):
        return self.model.predict(x_test)

    # save trained model
    def save_model(self, path_name):
        s = pickle.dumps({'model':self.model, 
                          'features':self.features,
                          'importance':self.importance,
                          'bst_params':self.bst_params})
        with open(path_name, 'wb+') as f:
            f.write(s)

    # load trained model
    def load_model(self, path_name):
        f = open(path_name, 'rb')  #  
        s = f.read()
        Model = pickle.loads(s)
        return Model


class LgbModelGrid(object):
    def __init__(self, train_day=None, nths=-1, save_path=None):
        """
        init from params dict
        """
        self.cv = 2
        self.model = None
        self.features = []
        self.nan_count = 0
        self.train_day = train_day
        self.nthreads = nths
        self.save_path = save_path

    def create_cv_index(self, i_set, shuffle=False, seed=123):
        """
        Return the labels for k-fold cross-validation sets, ensuring that data published at the same time 
        are either all in the training set or the validation set during the split."""
        i_set_unique = np.unique(i_set)
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(i_set_unique)
        cv_index = np.ones_like(i_set, dtype=int) * -1
        for k in range(self.cv):
            for m in i_set_unique[k::self.cv]:
                cv_index[i_set == m] = k
        return cv_index

    def fit(self, x_train, y_train, x_val, y_val):
        x_train, y_train = pd.DataFrame(x_train), pd.DataFrame(y_train)
        x_val, y_val = pd.DataFrame(x_val), pd.DataFrame(y_val)
        x_train['i_set'] = self.create_cv_index(range(len(x_train)), shuffle=True)
        i_fold = x_train[['i_set']]
        x_train.drop(['i_set'], axis=1, inplace=True)
        logger.info("features of train {}".format(x_train.columns.tolist()))
        
        # Define parameter grid for GridSearchCV
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [2, 10, 20],
            'min_child_samples': [1, 50, 80],
            'subsample': [0.01, 0.5, 1.0],
            'lambda_l2': [0.1, 50, 100]
        }

        best_score = float('inf')
        best_params = None

        # Perform grid search
        for params in itertools.product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            param_dict['objective'] = 'regression'
            param_dict['metric'] = 'rmse'
            param_dict['early_stopping_round'] = 10
            param_dict['saved_feature_importance_type'] = 1,
            param_dict['verbosity'] = -1
            param_dict['seed'] = 1000
            param_dict['nthread'] = self.nthreads
            scores = 0
            for i in range(self.cv):
                logger.info(f"training fold {i + 1}")
                train_data = lgb.Dataset(x_train.loc[i_fold[i_fold['i_set'] != i].index, :],
                                        y_train.loc[i_fold[i_fold['i_set'] != i].index, :])
                val_data = lgb.Dataset(x_train.loc[i_fold[i_fold['i_set'] == i].index, :],
                                      y_train.loc[i_fold[i_fold['i_set'] == i].index, :])
                bst = lgb.train(param_dict, train_data, num_boost_round=4000,
                                valid_sets=[train_data, val_data], valid_names=['train', 'eval'])
                scores += bst.best_score['eval']['rmse']
            avg_score = scores / self.cv
            if avg_score < best_score:
                best_score = avg_score
                best_params = param_dict

        logger.info("best param: {}".format(best_params))
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_child_samples'] = int(best_params['min_child_samples'])
        best_params['early_stopping_rounds'] = 50
        best_params['verbosity'] = -1
        
        # Train all data with best params
        train_data = lgb.Dataset(x_train, y_train)
        val_data = lgb.Dataset(x_val, y_val)
        bst = lgb.train(best_params, train_data, num_boost_round=4000,
                        valid_sets=[train_data, val_data], valid_names=['train', 'eval'])
        self.model = bst
        ori_gain_dict = dict(zip(list(x_train.columns), list(bst.feature_importance(importance_type='gain')))) 
        importance_by_gain = sorted(ori_gain_dict.items(), key=lambda item: item[1], reverse=True)

        self.features = list(x_train.columns)
        self.importance = importance_by_gain
        self.bst_params = best_params

        logger.info("feature importance is :{}".format(importance_by_gain))
        
        # Save features' importance 
        fi_save_path = os.path.join(self.save_path, 'lgb')
        os.makedirs(fi_save_path, exist_ok=True)
        f = open(os.path.join(fi_save_path, 'lgb_fi.txt'), "a")
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{formatted_datetime} \n {self.train_day}bst_params{best_params}\n importance_by_gain{importance_by_gain}" + '\n')
        f.close()

    def predict(self, x_test):
        return self.model.predict(x_test)

    # save trained model
    def save_model(self, path_name):
        s = pickle.dumps({'model':self.model, 
                          'features':self.features,
                          'importance':self.importance,
                          'bst_params':self.bst_params})
        with open(path_name, 'wb+') as f:
            f.write(s)

    # load trained model
    def load_model(self, path_name):
        f = open(path_name, 'rb')  #  
        s = f.read()
        Model = pickle.loads(s)
        return Model
    

class CatBoostModel(object):
    def __init__(self, train_day=None, nths=-1, save_path=None):
        """
        parameters init
        """
        self.cv = 2
        self.model = None
        self.features = []
        self.nan_count = 0
        self.train_day = train_day
        self.nthreads = nths
        self.save_path = save_path

    def create_cv_index(self, i_set, shuffle=False, seed=123):
        """
        Return the labels for k-fold cross-validation sets, ensuring that data published at the same time 
        are either all in the training set or the validation set during the split."""
        i_set_unique = np.unique(i_set)
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(i_set_unique)
        cv_index = np.ones_like(i_set, dtype=int) * -1
        for k in range(self.cv):
            for m in i_set_unique[k::self.cv]:
                cv_index[i_set == m] = k
        return cv_index

    def fit(self, x_train, y_train, x_val, y_val):
        x_train, y_train = pd.DataFrame(x_train), pd.DataFrame(y_train)
        x_val, y_val = pd.DataFrame(x_val), pd.DataFrame(y_val)
        x_train['i_set'] = self.create_cv_index(range(len(x_train)), shuffle=True)
        i_fold = x_train[['i_set']]
        x_train.drop(['i_set'], axis=1, inplace=True)
        logger.info("features of train {}".format(x_train.columns.tolist()))

        grid_params = {
            'learning_rate': (0.01, 0.3),
            'depth': (2, 10),
            'l2_leaf_reg': (0.1, 10),
            'subsample': (0.5, 1.0),
            'colsample_bylevel': (0.5, 1.0)
        }

        def catboost_evaluate(learning_rate, depth, l2_leaf_reg, subsample, colsample_bylevel):
            params = {
                'loss_function': 'RMSE',
                'learning_rate': float(learning_rate),
                'depth': int(depth),
                'l2_leaf_reg': float(l2_leaf_reg),
                'subsample': float(subsample),
                'colsample_bylevel': float(colsample_bylevel),
                'early_stopping_rounds': 10,
                'verbose': False,
                'random_seed': 1000,
                'thread_count': self.nthreads
            }
            scores = 0
            for i in range(self.cv):
                logger.info(f"training fold {i + 1}")
                train_data = Pool(x_train.loc[i_fold[i_fold['i_set'] != i].index, :],
                                  y_train.loc[i_fold[i_fold['i_set'] != i].index, :])
                val_data = Pool(x_train.loc[i_fold[i_fold['i_set'] == i].index, :],
                                y_train.loc[i_fold[i_fold['i_set'] == i].index, :])
                bst = CatBoostRegressor(**params)
                bst.fit(train_data, eval_set=val_data, use_best_model=True)
                scores += bst.get_best_score()['validation']['RMSE']
            return -1.0 * scores / self.cv

        catboost_bo = BayesianOptimization(catboost_evaluate, grid_params)
        catboost_bo.maximize()

        bst_params = catboost_bo.max['params']
        logger.info("best param: {}".format(bst_params))
        bst_params['depth'] = int(bst_params['depth'])
        bst_params['early_stopping_rounds'] = 50
        bst_params['verbose'] = False

        train_data = Pool(x_train, y_train)
        val_data = Pool(x_val, y_val)
        bst = CatBoostRegressor(**bst_params)
        bst.fit(train_data, eval_set=val_data, use_best_model=True)
        self.model = bst

        ori_gain_dict = dict(zip(list(x_train.columns), list(bst.get_feature_importance())))
        importance_by_gain = sorted(ori_gain_dict.items(), key=lambda item: item[1], reverse=True)

        self.features = list(x_train.columns)
        self.importance = importance_by_gain
        self.bst_params = bst_params

        logger.info("feature importance is :{}".format(importance_by_gain))

        fi_save_path = fi_save_path = os.path.join(self.save_path, 'catboost')
        os.makedirs(fi_save_path, exist_ok=True)
        
        f = open(os.path.join(fi_save_path, 'catboost_fi.txt'), "a")
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{formatted_datetime} \n {self.train_day}bst_params{bst_params}\n importance_by_gain{importance_by_gain}" + '\n')
        f.close()

    def predict(self, x_test):
        return self.model.predict(x_test)

    def save_model(self, path_name):
        self.model.save_model(path_name)

    def load_model(self, path_name):
        model = CatBoostRegressor()
        model.load_model(path_name)
        return model


class CatBoostModelGrid(object):
    def __init__(self, train_day=None, nths=-1, save_path=None):
        """
        初始化参数
        """
        self.cv = 2
        self.model = None
        self.features = []
        self.nan_count = 0
        self.train_day = train_day
        self.nthreads = nths
        self.save_path = save_path

    def create_cv_index(self, i_set, shuffle=False, seed=123):
        """
        Return the labels for k-fold cross-validation sets, ensuring that data published at the same time 
        are either all in the training set or the validation set during the split."""
        i_set_unique = np.unique(i_set)
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(i_set_unique)
        cv_index = np.ones_like(i_set, dtype=int) * -1
        for k in range(self.cv):
            for m in i_set_unique[k::self.cv]:
                cv_index[i_set == m] = k
        return cv_index

    def fit(self, x_train, y_train, x_val, y_val):
        x_train, y_train = pd.DataFrame(x_train), pd.DataFrame(y_train)
        x_val, y_val = pd.DataFrame(x_val), pd.DataFrame(y_val)
        x_train['i_set'] = self.create_cv_index(range(len(x_train)), shuffle=True)
        i_fold = x_train[['i_set']]
        x_train.drop(['i_set'], axis=1, inplace=True)
        logger.info("features of train {}".format(x_train.columns.tolist()))

        param_grid = {
            'learning_rate': [0.01, 0.1, 0.3],
            'depth': [2, 6, 10],
            'l2_leaf_reg': [0.1, 5, 10],
            'subsample': [0.5, 0.8, 1.0],
            'colsample_bylevel': [0.5, 0.8, 1.0]
        }

        best_score = float('inf')
        best_params = None

        for params in itertools.product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            param_dict['loss_function'] = 'RMSE'
            param_dict['early_stopping_rounds'] = 10
            param_dict['verbose'] = False
            param_dict['random_seed'] = 1000
            param_dict['thread_count'] = self.nthreads
            scores = 0
            for i in range(self.cv):
                logger.info(f"training fold {i + 1}")
                train_data = Pool(x_train.loc[i_fold[i_fold['i_set'] != i].index, :],
                                  y_train.loc[i_fold[i_fold['i_set'] != i].index, :])
                val_data = Pool(x_train.loc[i_fold[i_fold['i_set'] == i].index, :],
                                y_train.loc[i_fold[i_fold['i_set'] == i].index, :])
                bst = CatBoostRegressor(**param_dict)
                bst.fit(train_data, eval_set=val_data, use_best_model=True)
                scores += bst.get_best_score()['validation']['RMSE']
            avg_score = scores / self.cv
            if avg_score < best_score:
                best_score = avg_score
                best_params = param_dict

        logger.info("best param: {}".format(best_params))
        best_params['depth'] = int(best_params['depth'])
        best_params['early_stopping_rounds'] = 50
        best_params['verbose'] = False

        train_data = Pool(x_train, y_train)
        val_data = Pool(x_val, y_val)
        bst = CatBoostRegressor(**best_params)
        bst.fit(train_data, eval_set=val_data, use_best_model=True)
        self.model = bst

        ori_gain_dict = dict(zip(list(x_train.columns), list(bst.get_feature_importance())))
        importance_by_gain = sorted(ori_gain_dict.items(), key=lambda item: item[1], reverse=True)

        self.features = list(x_train.columns)
        self.importance = importance_by_gain
        self.bst_params = best_params

        logger.info("feature importance is :{}".format(importance_by_gain))

        fi_save_path = os.path.join(self.save_path, 'catboost')
        os.makedirs(fi_save_path, exist_ok=True)
        
        f = open(os.path.join(fi_save_path, 'catboost_fi.txt'), "a")
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"{formatted_datetime} \n {self.train_day}bst_params{best_params}\n importance_by_gain{importance_by_gain}" + '\n')
        f.close()

    def predict(self, x_test):
        return self.model.predict(x_test)

    def save_model(self, path_name):
        self.model.save_model(path_name)

    def load_model(self, path_name):
        model = CatBoostRegressor()
        model.load_model(path_name)
        return model