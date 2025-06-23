"""
@File    : hyp_set.py
@Time    : 2025/4/6 18:34
@Author  : mingyang.zong
"""
# deep_lstm hyper param
weekday_hyp = {"seq_len": 144 * 4,
               "pre_len": 144,
               "sample_step": 144,
               "target": "Y",
               "train_ratio": 0.9,
               "val_ratio": 0.1,
               "train_batch": 4,
               "printer_iter": 20,
               "inverse_value": None}

holiday_hyp = {"seq_len": 144 * 2,
               "pre_len": 144,
               "sample_step": 144,
               "target": "Y",
               "train_ratio": 0.9,
               "val_ratio": 0.1,
               "train_batch": 2,
               "printer_iter": 20,
               "inverse_value": None}

params = {
    'cv': 5,
    'pred_col': 'Y'
}
