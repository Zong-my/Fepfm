# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2025/04/19 20:04:01
@Author  :   myz 
'''
import os
import pandas as pd
from loguru import logger
from sklearn.svm import SVR
import statsmodels.api as sm
from visualization import *
from features_select import *
from gbdt_based_models import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from evaluation import rmse_loss, mae_loss, mape_loss, smape_loss

import warnings
warnings.filterwarnings("ignore")

def data_split(X, y1, y2):
    X3_train, X3_temp, y3_1_train, y3_1_temp, y3_2_train, y3_2_temp = train_test_split(
        X, y1, y2, test_size=0.2, random_state=42)
    X3_val, X3_test, y3_1_val, y3_1_test, y3_2_val, y3_2_test = train_test_split(
        X3_temp, y3_1_temp, y3_2_temp, test_size=0.5, random_state=42)
    return (X3_train, y3_1_train, y3_2_train), (X3_val, y3_1_val, y3_2_val), (X3_test, y3_1_test, y3_2_test)

@log_execution_time
def data_proc(v, data_path, ms=None, features=None, train_ratio=None):
    # Data Standardization
    scaler_X = StandardScaler()
    scaler_y1 = StandardScaler()
    scaler_y2 = StandardScaler()

    y1_train = pd.read_csv(f'{data_path}y_1_train.csv', index_col=0)
    y1_test = pd.read_csv(f'{data_path}y_1_test.csv', index_col=0)
    y1_val = pd.read_csv(f'{data_path}y_1_val.csv', index_col=0)
    y2_train = pd.read_csv(f'{data_path}y_2_train.csv', index_col=0)
    y2_test = pd.read_csv(f'{data_path}y_2_test.csv', index_col=0)
    y2_val = pd.read_csv(f'{data_path}y_2_val.csv', index_col=0)
    
    X_train = pd.read_csv(f'{data_path}X_train.csv', index_col=0)
    X_test = pd.read_csv(f'{data_path}X_test.csv', index_col=0)
    X_val = pd.read_csv(f'{data_path}X_val.csv', index_col=0)

    if ms:
        # Generate Feature Sequences at Specified Time Scales
        X_train_solid, X_test_solid, X_val_solid = X_train.iloc[:, :7], X_test.iloc[:, :7], X_val.iloc[:, :7]
        cols = X_train.columns[7:]
        cols_use = []
        for cl in cols:
            if int(cl.split('_')[-1]) <= ms: # 
                cols_use.append(cl)
        X_train_tmp, X_test_tmp, X_val_tmp = X_train[cols_use], X_test[cols_use], X_val[cols_use]
        X_train = pd.concat([X_train_solid, X_train_tmp], axis=1)
        X_test = pd.concat([X_test_solid, X_test_tmp], axis=1)
        X_val = pd.concat([X_val_solid, X_val_tmp], axis=1)  
    
    if features:
        X_train = X_train[features]
        X_test = X_test[features]
        X_val = X_val[features]
    
    if train_ratio:
        X_train = X_train.sample(frac=train_ratio, random_state=42)
        y1_train, y2_train = y1_train.loc[X_train.index], y2_train.loc[X_train.index]
    
    scaler_X.fit(X_train)
    scaler_y1.fit(y1_train)
    scaler_y2.fit(y2_train)

    X_train_scaled = scaler_X.transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    y1_train_scaled = scaler_y1.transform(y1_train)
    y2_train_scaled = scaler_y2.transform(y2_train)
    
    return (X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test), \
            (X_train_scaled, X_val_scaled, X_test_scaled, y1_train_scaled, y2_train_scaled), \
            (scaler_y1, scaler_y2, scaler_X)

@log_execution_time
def arimax_train_test(v, X1_train, y1_train, X2_train, y2_train,
                      X1_test, y1_test, X2_test, y2_test,
                      ms=None, result_path=None):
    print(""" arimax """)
    csv_path = os.path.join(result_path, 'result.csv')
    if os.path.exists(csv_path): 
        result = pd.read_csv(csv_path, index_col=0)
    else:
        result = pd.DataFrame()  
    
    result_path = os.path.join(result_path, 'arimax')
    os.makedirs(result_path, exist_ok=True)

    arimax1 = sm.tsa.statespace.SARIMAX(y1_train, exog=X1_train).fit(disp=-1)
    arimax2 = sm.tsa.statespace.SARIMAX(y2_train, exog=X2_train).fit(disp=-1)

    # predict
    y1_pred = arimax1.forecast(steps=len(X1_test), exog=X1_test).values.flatten()
    y2_pred = arimax2.forecast(steps=len(X2_test), exog=X2_test).values.flatten()
    tmp_pred_y1['arimax'] = y1_pred
    tmp_pred_y2['arimax'] = y2_pred

    # model evaluate
    y1_test = pd.DataFrame(y1_test).values.flatten()
    y2_test = pd.DataFrame(y2_test).values.flatten()
    mae_y1, mae_y2 = mae_loss(y1_test, y1_pred), mae_loss(y2_test, y2_pred)
    rmse_y1, rmse_y2 = rmse_loss(y1_test, y1_pred), rmse_loss(y2_test, y2_pred)
    mape_y1, mape_y2 = mape_loss(y1_test, y1_pred), mape_loss(y2_test, y2_pred)
    smape_y1, smape_y2 = smape_loss(y1_test, y1_pred), smape_loss(y2_test, y2_pred)

    print('fpu_deltamax:')
    print(f"y1 mae: {mae_y1}")
    print(f"y1 rmse: {rmse_y1}")
    print(f"y1 mape: {mape_y1}")
    print(f"y1 smape: {smape_y1}\n")

    print('t_delta:')
    print(f"y2 mae: {mae_y2}")
    print(f"y2 rmse: {rmse_y2}")
    print(f"y2 mape: {mape_y2}")
    print(f"y2 smape: {smape_y2}")

    # Save Prediction Result Images
    if ms:
        plot_prediction(y1_test, y1_pred, y2_test, y2_pred, 
                        save_path=os.path.join(result_path, f'arimax_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}_{timestamp_str}.svg'), dpi=600)
    else:
        plot_prediction(y1_test, y1_pred, y2_test, y2_pred, 
                        save_path=os.path.join(result_path, f'arimax_v{v}_rt_{rt}_{pop}_{timestamp_str}.svg'), dpi=600) 

    # Save Results to CSV File
    cols = ['y1(fd) mae', 'y1(fd) rmse', 'y1(fd) mape', 'y1(fd) smape', 'y2(td) mae', 'y2(td) rmse', 'y2(td) mape', 'y2(td) smape']
    arimax_values = [mae_y1, rmse_y1, mape_y1, smape_y1, mae_y2, rmse_y2, mape_y2, smape_y2]
    arimax_index = [f'arimax_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}'] if ms else [f'arimax_v{v}_rt_{rt}_{pop}']
    df_tmp = pd.DataFrame([arimax_values], columns=cols, index=[arimax_index])
    result = pd.concat([result, df_tmp], axis=0)
    result.to_csv(csv_path)

    # best model saved
    if ms:
        save_pickle(arimax1, os.path.join(result_path, f'y1_arimax_model_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}_{timestamp_str}.pickle'))
        save_pickle(arimax2, os.path.join(result_path, f'y2_arimax_model_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}_{timestamp_str}.pickle'))
    else:
        save_pickle(arimax1, os.path.join(result_path, f'y1_arimax_model_v{v}_rt_{rt}_{pop}_{timestamp_str}.pickle'))
        save_pickle(arimax2, os.path.join(result_path, f'y2_arimax_model_v{v}_rt_{rt}_{pop}_{timestamp_str}.pickle'))

    print('arimax complete!')

@log_execution_time
def svr_train_test(v, X1_train_scaled, y1_train_scaled, X2_train_scaled, y2_train_scaled,
                   X1_test_scaled, y1_test, X2_test_scaled, y2_test, scaler_y1, scaler_y2, 
                   ms=None, result_path=None):
    print(""" svr """)
    csv_path = os.path.join(result_path, 'result.csv')
    if os.path.exists(csv_path): 
        result = pd.read_csv(csv_path, index_col=0)
    else:
        result = pd.DataFrame()  

    result_path = os.path.join(result_path, 'svr')
    os.makedirs(result_path, exist_ok=True)

    # Define Hyperparameter Grid Search Space
    y1_param_grid = {
        'C': [1, 10, 100],
        'gamma': [0.01, 0.05, 0.1],
        'epsilon': [0.001, 0.005, 0.01]
        }
    
    y2_param_grid = {
        'C': [10, 100, 1000],
        'gamma': [0.01, 0.1, 0.5],
        'epsilon': [0.01, 0.1, 0.5]
        }
    
    # GridSearchCV
    svr1 = SVR(kernel='rbf')
    grid_search1 = GridSearchCV(svr1, y1_param_grid, cv=2, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search1.fit(X1_train_scaled, y1_train_scaled.ravel())
    best_params1 = grid_search1.best_params_
    logger.info(f"Best SVR1 parameters: {best_params1}")

    svr2 = SVR(kernel='rbf')
    grid_search2 = GridSearchCV(svr2, y2_param_grid, cv=2, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search2.fit(X2_train_scaled, y2_train_scaled.ravel())
    best_params2 = grid_search2.best_params_
    logger.info(f"Best SVR2 parameters: {best_params2}")

    # save best params 
    fi_save_path = os.path.join(result_path, 'svr')
    os.makedirs(fi_save_path, exist_ok=True)
    f = open(os.path.join(fi_save_path, 'svr_best_params.txt'), "a")
    f.write(f"{timestamp_str} \n m_s:{ms}_{pop} \n best_params1:{best_params1}\n best_params2:{best_params2}" + '\n')
    f.close()

    svr_y1 = SVR(kernel='rbf', **best_params1)
    svr_y1.fit(X1_train_scaled, y1_train_scaled.ravel())

    svr_y2 = SVR(kernel='rbf', **best_params2)
    svr_y2.fit(X2_train_scaled, y2_train_scaled.ravel())

    # predict
    y1_pred_scaled = svr_y1.predict(X1_test_scaled)
    y1_pred = scaler_y1.inverse_transform(y1_pred_scaled.reshape(-1, 1)).ravel()

    y2_pred_scaled = svr_y2.predict(X2_test_scaled)
    y2_pred = scaler_y2.inverse_transform(y2_pred_scaled.reshape(-1, 1)).ravel()

    tmp_pred_y1['svr'] = y1_pred
    tmp_pred_y2['svr'] = y2_pred

    # model evaluate
    y1_test = pd.DataFrame(y1_test).values.flatten()
    y2_test = pd.DataFrame(y2_test).values.flatten()
    mae_y1, mae_y2 = mae_loss(y1_test, y1_pred), mae_loss(y2_test, y2_pred)
    rmse_y1, rmse_y2 = rmse_loss(y1_test, y1_pred), rmse_loss(y2_test, y2_pred)
    mape_y1, mape_y2 = mape_loss(y1_test, y1_pred), mape_loss(y2_test, y2_pred)
    smape_y1, smape_y2 = smape_loss(y1_test, y1_pred), smape_loss(y2_test, y2_pred)

    print('fpu_deltamax:')
    print(f"y1 mae: {mae_y1}")
    print(f"y1 rmse: {rmse_y1}")
    print(f"y1 mape: {mape_y1}")
    print(f"y1 smape: {smape_y1}\n")

    print('t_delta:')
    print(f"y2 mae: {mae_y2}")
    print(f"y2 rmse: {rmse_y2}")
    print(f"y2 mape: {mape_y2}")
    print(f"y2 smape: {smape_y2}")

    # Save Prediction Result Images
    if ms:
        plot_prediction(y1_test, y1_pred, y2_test, y2_pred, 
                        save_path=os.path.join(result_path, f'svr_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}_{timestamp_str}.svg'), dpi=600)
    else:
        plot_prediction(y1_test, y1_pred, y2_test, y2_pred, 
                        save_path=os.path.join(result_path, f'svr_v{v}_rt_{rt}_{pop}_{timestamp_str}.svg'), dpi=600) 

    # Save Results to CSV File
    cols = ['y1(fd) mae', 'y1(fd) rmse', 'y1(fd) mape', 'y1(fd) smape', 'y2(td) mae', 'y2(td) rmse', 'y2(td) mape', 'y2(td) smape']
    svr_values = [mae_y1, rmse_y1, mape_y1, smape_y1, mae_y2, rmse_y2, mape_y2, smape_y2]
    svr_index = [f'svr_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}'] if ms else [f'svr_v{v}_rt_{rt}_{pop}']
    df_tmp = pd.DataFrame([svr_values], columns=cols, index=[svr_index])
    result = pd.concat([result, df_tmp], axis=0)
    result.to_csv(csv_path)

    # best model saved
    if ms:
        save_pickle(svr_y1, os.path.join(result_path, f'y1_svr_model_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}_{timestamp_str}.pickle'))
        save_pickle(svr_y2, os.path.join(result_path, f'y2_svr_model_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}_{timestamp_str}.pickle'))
    else:
        save_pickle(svr_y1, os.path.join(result_path, f'y1_svr_model_v{v}_rt_{rt}_{pop}_{timestamp_str}.pickle'))
        save_pickle(svr_y2, os.path.join(result_path, f'y2_svr_model_v{v}_rt_{rt}_{pop}_{timestamp_str}.pickle'))

    print('svr complete!')

@log_execution_time
def xgb_train_test(v, X1_train, y1_train, X2_train, y2_train, X1_test, y1_test, X2_test, y2_test, ms=None, result_path=None):
    print(""" xgb """)
    csv_path = os.path.join(result_path, 'result.csv')
    if os.path.exists(csv_path): 
        result = pd.read_csv(csv_path, index_col=0)
    else:
        result = pd.DataFrame()  

    result_path = os.path.join(result_path, 'xgb')
    os.makedirs(result_path, exist_ok=True)

    """ y1: fpu_deltamax"""
    # build model
    y1_xgb = XgbModel(train_day=f"y1-fpu_deltamax-v{v}-ms{ms}_{pop}: ", nths=nthreads, save_path=result_path)
    y1_xgb.fit(X1_train, y1_train, X1_val, y1_val)
    if ms:
        save_pickle(y1_xgb, os.path.join(result_path, f'y1_xgb_model_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}_{timestamp_str}.pickle'))
    else:
        save_pickle(y1_xgb, os.path.join(result_path, f'y1_xgb_model_v{v}_rt_{rt}_{pop}_{timestamp_str}.pickle'))
    y1_pred = y1_xgb.predict(X1_test)

    """ y2: t_delta"""
    y2_xgb = XgbModel(train_day=f"y2-t_delta-v{v}-ms{ms}_{pop}: ", nths=nthreads, save_path=result_path)
    y2_xgb.fit(X2_train, y2_train, X2_val, y2_val)
    if ms:
        save_pickle(y2_xgb, os.path.join(result_path, f'y2_xgb_model_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}_{timestamp_str}.pickle'))
    else:
        save_pickle(y2_xgb, os.path.join(result_path, f'y2_xgb_model_v{v}_rt_{rt}_{pop}_{timestamp_str}.pickle'))
    y2_pred = y2_xgb.predict(X2_test)
    
    tmp_pred_y1['xgb'] = y1_pred
    tmp_pred_y2['xgb'] = y2_pred

    # model evaluate
    y1_test = pd.DataFrame(y1_test).values.flatten()
    y2_test = pd.DataFrame(y2_test).values.flatten()
    mae_y1, mae_y2 = mae_loss(y1_test, y1_pred), mae_loss(y2_test, y2_pred)
    rmse_y1, rmse_y2 = rmse_loss(y1_test, y1_pred), rmse_loss(y2_test, y2_pred)
    mape_y1, mape_y2 = mape_loss(y1_test, y1_pred), mape_loss(y2_test, y2_pred)
    smape_y1, smape_y2 = smape_loss(y1_test, y1_pred), smape_loss(y2_test, y2_pred)
    
    print('fpu_deltamax:')
    print(f"y1 mae: {mae_y1}")
    print(f"y1 rmse: {rmse_y1}")
    print(f"y1 mape: {mape_y1}")
    print(f"y1 smape: {smape_y1}\n")

    print('t_delta:')
    print(f"y2 mae: {mae_y2}")
    print(f"y2 rmse: {rmse_y2}")
    print(f"y2 mape: {mape_y2}")
    print(f"y2 smape: {smape_y2}")
    if ms:
        plot_prediction(y1_test, y1_pred, y2_test, y2_pred, 
                        save_path=os.path.join(result_path, f'xgb_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}_{timestamp_str}.svg'), dpi=600)
    else:
        plot_prediction(y1_test, y1_pred, y2_test, y2_pred, 
                        save_path=os.path.join(result_path, f'xgb_v{v}_rt_{rt}_{pop}_{timestamp_str}.svg'), dpi=600) 
    cols = ['y1(fd) mae', 'y1(fd) rmse', 'y1(fd) mape', 'y1(fd) smape', 'y2(td) mae', 'y2(td) rmse', 'y2(td) mape', 'y2(td) smape']
    xgb_values = [mae_y1, rmse_y1, mape_y1, smape_y1, mae_y2, rmse_y2, mape_y2, smape_y2]
    xgb_index = [f'xgb_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}'] if ms else [f'xgb_v{v}_rt_{rt}_{pop}']
    df_tmp = pd.DataFrame([xgb_values], columns=cols, index=[xgb_index])
    result = pd.concat([result, df_tmp], axis=0)
    result.to_csv(csv_path)  
    print('xgb complete!')

@log_execution_time
def lgb_train_test(v, X1_train, y1_train, X2_train, y2_train, X1_test, y1_test, X2_test, y2_test, ms=None, result_path=None, pops='bayes'):
    print(""" lgb """)
    csv_path = os.path.join(result_path, 'result.csv')
    if os.path.exists(csv_path): 
        result = pd.read_csv(csv_path, index_col=0)
    else:
        result = pd.DataFrame()   

    result_path = os.path.join(result_path, 'lgb')
    os.makedirs(result_path, exist_ok=True)

    """ y1: fpu_deltamax"""
    # build model
    if pops == 'bayes':
        y1_lgb = LgbModel(train_day=f"y1-fpu_deltamax-v{v}-ms{ms}_{pop}: ", nths=nthreads, save_path=result_path)
    else:
        y1_lgb = LgbModelGrid(train_day=f"y1-fpu_deltamax-v{v}-ms{ms}_{pop}: ", nths=nthreads, save_path=result_path)

    y1_lgb.fit(X1_train, y1_train, X1_val, y1_val)
    if ms:
        save_pickle(y1_lgb, os.path.join(result_path, f'y1_lgb_model_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}_{timestamp_str}.pickle'))
    else:
        save_pickle(y1_lgb, os.path.join(result_path, f'y1_lgb_model_v{v}_rt_{rt}_{pop}_{timestamp_str}.pickle'))
    y1_pred = y1_lgb.predict(X1_test)

    """ y2: t_delta"""
    if pops == 'bayes':
        y2_lgb = LgbModel(train_day=f"y2-fpu_deltamax-v{v}-ms{ms}_{pop}: ", nths=nthreads, save_path=result_path)
    else:
        y2_lgb = LgbModelGrid(train_day=f"y2-fpu_deltamax-v{v}-ms{ms}_{pop}: ", nths=nthreads, save_path=result_path)
    y2_lgb.fit(X2_train, y2_train, X2_val, y2_val)
    if ms:
        save_pickle(y2_lgb, os.path.join(result_path, f'y2_lgb_model_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}_{timestamp_str}.pickle'))
    else:
        save_pickle(y2_lgb, os.path.join(result_path, f'y2_lgb_model_v{v}_rt_{rt}_{pop}_{timestamp_str}.pickle'))
    y2_pred = y2_lgb.predict(X2_test)
    
    tmp_pred_y1['lgb'] = y1_pred
    tmp_pred_y2['lgb'] = y2_pred
    
    # model evaluate
    y1_test = pd.DataFrame(y1_test).values.flatten()
    y2_test = pd.DataFrame(y2_test).values.flatten()
    mae_y1, mae_y2 = mae_loss(y1_test, y1_pred), mae_loss(y2_test, y2_pred)
    rmse_y1, rmse_y2 = rmse_loss(y1_test, y1_pred), rmse_loss(y2_test, y2_pred)
    mape_y1, mape_y2 = mape_loss(y1_test, y1_pred), mape_loss(y2_test, y2_pred)
    smape_y1, smape_y2 = smape_loss(y1_test, y1_pred), smape_loss(y2_test, y2_pred)
    
    print('fpu_deltamax:')
    print(f"y1 mae: {mae_y1}")
    print(f"y1 rmse: {rmse_y1}")
    print(f"y1 mape: {mape_y1}")
    print(f"y1 smape: {smape_y1}\n")

    print('t_delta:')
    print(f"y2 mae: {mae_y2}")
    print(f"y2 rmse: {rmse_y2}")
    print(f"y2 mape: {mape_y2}")
    print(f"y2 smape: {smape_y2}")
    if ms:
        plot_prediction(y1_test, y1_pred, y2_test, y2_pred, 
                        save_path=os.path.join(result_path, f'lgb_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}_{timestamp_str}.svg'), dpi=600)
    else:
        plot_prediction(y1_test, y1_pred, y2_test, y2_pred, 
                        save_path=os.path.join(result_path, f'lgb_v{v}_rt_{rt}_{pop}_{timestamp_str}.svg'), dpi=600) 
    cols = ['y1(fd) mae', 'y1(fd) rmse', 'y1(fd) mape', 'y1(fd) smape', 'y2(td) mae', 'y2(td) rmse', 'y2(td) mape', 'y2(td) smape']
    lgb_values = [mae_y1, rmse_y1, mape_y1, smape_y1, mae_y2, rmse_y2, mape_y2, smape_y2]
    lgb_index = [f'lgb_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}'] if ms else [f'lgb_v{v}_rt_{rt}_{pop}']
    df_tmp = pd.DataFrame([lgb_values], columns=cols, index=[lgb_index])
    result = pd.concat([result, df_tmp], axis=0)
    result.to_csv(csv_path)  
    print('lgb complete!')

@log_execution_time
def catboost_train_test(v, X1_train, y1_train, X2_train, y2_train, X1_test, y1_test, X2_test, y2_test, ms=None, result_path=None, pops='bayes'):
    print(""" catboost """)
    csv_path = os.path.join(result_path, 'result.csv')
    if os.path.exists(csv_path): 
        result = pd.read_csv(csv_path, index_col=0)
    else:
        result = pd.DataFrame()    

    result_path = os.path.join(result_path, 'catboost')
    os.makedirs(result_path, exist_ok=True)

    """ y1: fpu_deltamax"""
    # model build
    if pops=='bayes':
        y1_catboost = CatBoostModel(train_day=f"y1-fpu_deltamax-v{v}-ms{ms}_{pop}: ", nths=nthreads, save_path=result_path)
    else:
        y1_catboost = CatBoostModelGrid(train_day=f"y1-fpu_deltamax-v{v}-ms{ms}_{pop}: ", nths=nthreads, save_path=result_path)

    y1_catboost.fit(X1_train, y1_train, X1_val, y1_val)
    
    if ms:
        save_pickle(y1_catboost, os.path.join(result_path, f'y1_catboost_model_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}_{timestamp_str}.pickle'))
    else:
        save_pickle(y1_catboost, os.path.join(result_path, f'y1_catboost_model_v{v}_rt_{rt}_{pop}_{timestamp_str}.pickle'))

    y1_pred = y1_catboost.predict(X1_test)

    """ y2: t_delta"""
    if pops=='bayes':
        y2_catboost = CatBoostModel(train_day=f"y2-fpu_deltamax-v{v}-ms{ms}_{pop}: ", nths=nthreads, save_path=result_path)
    else:
        y2_catboost = CatBoostModelGrid(train_day=f"y2-fpu_deltamax-v{v}-ms{ms}_{pop}: ", nths=nthreads, save_path=result_path)

    y2_catboost.fit(X2_train, y2_train, X2_val, y2_val)
    
    if ms:
        save_pickle(y2_catboost, os.path.join(result_path, f'y2_catboost_model_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}_{timestamp_str}.pickle'))
    else:
        save_pickle(y2_catboost, os.path.join(result_path, f'y2_catboost_model_v{v}_rt_{rt}_{pop}_{timestamp_str}.pickle'))
    
    y2_pred = y2_catboost.predict(X2_test)

    tmp_pred_y1['catboost'] = y1_pred
    tmp_pred_y2['catboost'] = y2_pred

    # model evaluate
    y1_test = pd.DataFrame(y1_test).values.flatten()
    y2_test = pd.DataFrame(y2_test).values.flatten()
    mae_y1, mae_y2 = mae_loss(y1_test, y1_pred), mae_loss(y2_test, y2_pred)
    rmse_y1, rmse_y2 = rmse_loss(y1_test, y1_pred), rmse_loss(y2_test, y2_pred)
    mape_y1, mape_y2 = mape_loss(y1_test, y1_pred), mape_loss(y2_test, y2_pred)
    smape_y1, smape_y2 = smape_loss(y1_test, y1_pred), smape_loss(y2_test, y2_pred)

    print('fpu_deltamax:')
    print(f"y1 mae: {mae_y1}")
    print(f"y1 rmse: {rmse_y1}")
    print(f"y1 mape: {mape_y1}")
    print(f"y1 smape: {smape_y1}\n")

    print('t_delta:')
    print(f"y2 mae: {mae_y2}")
    print(f"y2 rmse: {rmse_y2}")
    print(f"y2 mape: {mape_y2}")
    print(f"y2 smape: {smape_y2}")

    if ms:
        plot_prediction(y1_test, y1_pred, y2_test, y2_pred, 
                        save_path=os.path.join(result_path, f'catboost_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}_{timestamp_str}.svg'), dpi=600)
    else:
        plot_prediction(y1_test, y1_pred, y2_test, y2_pred, 
                        save_path=os.path.join(result_path, f'catboost_v{v}_rt_{rt}_{pop}_{timestamp_str}.svg'), dpi=600) 

    cols = ['y1(fd) mae', 'y1(fd) rmse', 'y1(fd) mape', 'y1(fd) smape', 'y2(td) mae', 'y2(td) rmse', 'y2(td) mape', 'y2(td) smape']
    catboost_values = [mae_y1, rmse_y1, mape_y1, smape_y1, mae_y2, rmse_y2, mape_y2, smape_y2]
    catboost_index = [f'catboost_v{v}_rt_{rt}_{pop}_ms{ms}_{pop}'] if ms else [f'catboost_v{v}_rt_{rt}_{pop}']
    df_tmp = pd.DataFrame([catboost_values], columns=cols, index=[catboost_index])
    result = pd.concat([result, df_tmp], axis=0)
    result.to_csv(csv_path)  
    print('catboost complete!')


if __name__ == "__main__":
    ratio = [1.0]
    nthreads = -1
    version_ = '5'
    data_pt = './data/ieee39/v5/'
    save_path_ = f"./runs/freq_pred/v{version_}/"
    os.makedirs(save_path_, exist_ok=True)
    from datetime import datetime
    current_time = datetime.now()
    timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    use_models = ["lgb", "catboost"]
    param_opt_mts = ['grid_search']   # 'bayes', 'grid_search'
    
    for pop in param_opt_mts:

        feature_select_methods = {'xg_lgb_cat_shap': features_select,
                                  # 'rf': feature_select_rf, 
                                  # 'mutual_info': feature_select_mutual_info, 
                                  # 'rfe': feature_select_rfe, 
                                  # 'genetic': feature_select_genetic,
                                  # 'lasso': feature_select_lasso, 
                                  # 'lgb_shap': feature_select_lgb_shap
                                }
        
        for ss, fs in feature_select_methods.items():
            save_path = os.path.join(save_path_, ss)
            os.makedirs(save_path, exist_ok=True)
            for rt in ratio:
                # Multi-time Scale Information test
                for m_s in [1, 2, 5, 10, 15, 20, 25]:
                    path_y1 = os.path.join(save_path, f'y1_pred_rt_{rt}_{pop}_ms{m_s}_{timestamp_str}.csv')
                    path_y2 = os.path.join(save_path, f'y2_pred_rt_{rt}_{pop}_ms{m_s}_{timestamp_str}.csv')
                    if os.path.exists(path_y1):
                        tmp_pred_y1 = pd.read_csv(path_y1, index_col=0)
                    else:
                        tmp_pred_y1 = pd.DataFrame()
                    if os.path.exists(path_y2):
                        tmp_pred_y2 = pd.read_csv(path_y2, index_col=0)
                    else:
                        tmp_pred_y2 = pd.DataFrame()
            
                    oris, scaleds, scalers = data_proc(version_, data_path=data_pt, ms=m_s, train_ratio=rt)  # ms
                    
                    # Feature Selection
                    y1_features = fs(oris[0], oris[3], oris[1], oris[4])
                    y2_features = fs(oris[0], oris[6], oris[1], oris[7])

                    save_pickle(y1_features, os.path.join(save_path, f'y1_features_rt{rt}_{pop}_ms{m_s}_{timestamp_str}.pickle'))
                    save_pickle(y2_features, os.path.join(save_path, f'y2_features_rt{rt}_{pop}_ms{m_s}_{timestamp_str}.pickle'))
                    oris1, scaleds1, scalers1 = data_proc(version_, data_path=data_pt, features=y1_features, train_ratio=rt) # feature select
                    oris2, scaleds2, scalers2 = data_proc(version_, data_path=data_pt, features=y2_features, train_ratio=rt)

                    save_pickle(scalers1[0], os.path.join(save_path, f'scalers_y1_rt_{rt}_{pop}_{timestamp_str}.pickle'))
                    save_pickle(scalers1[1], os.path.join(save_path, f'scalers_y2_rt_{rt}_{pop}_{timestamp_str}.pickle'))
                    save_pickle(scalers1[2], os.path.join(save_path, f'scalers_X1_rt_{rt}_{pop}_ms{m_s}_{timestamp_str}.pickle'))
                    save_pickle(scalers2[2], os.path.join(save_path, f'scalers_X2_rt_{rt}_{pop}_ms{m_s}_{timestamp_str}.pickle'))
                    
                    X1_train, y1_train, X1_val, y1_val, X1_test, y1_test = oris1[0], oris1[3], oris1[1], oris1[4], oris1[2], oris1[5]
                    X1_train_scaled, y1_train_scaled, X1_test_scaled = scaleds1[0], scaleds1[3], scaleds1[2]
                    
                    X2_train, y2_train, X2_val, y2_val, X2_test, y2_test = oris2[0], oris2[6], oris2[1], oris2[7], oris2[2], oris2[8]
                    X2_train_scaled, y2_train_scaled, X2_test_scaled = scaleds2[0], scaleds2[4], scaleds2[2]

                    scaler_y1, scaler_y2 = scalers1[0], scalers1[1]
                    tmp_pred_y1['y1_true'], tmp_pred_y2['y2_true'] = y1_test.values.flatten(), y2_test.values.flatten()

                    arimax_train_test(version_, X1_train, y1_train, X2_train, y2_train, X1_test, y1_test, X2_test, y2_test,
                                        ms=m_s, result_path=save_path)

                    svr_train_test(version_, X1_train_scaled, y1_train_scaled, X2_train_scaled, y2_train_scaled,
                                    X1_test_scaled, y1_test, X2_test_scaled, y2_test, scaler_y1, scaler_y2, 
                                    ms=m_s, result_path=save_path)
            
                    lgb_train_test(version_, X1_train, y1_train, X2_train, y2_train, X1_test, y1_test, X2_test, y2_test, 
                                    ms=m_s, result_path=save_path, pops=pop)
                    
                    catboost_train_test(version_, X1_train, y1_train, X2_train, y2_train, X1_test, y1_test, X2_test, y2_test, 
                                        ms=m_s, result_path=save_path, pops=pop)
                    
                tmp_pred_y1.to_csv(path_y1)
                tmp_pred_y2.to_csv(path_y2)

    print('totally completed!!')
