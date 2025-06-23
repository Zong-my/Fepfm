import copy
import numpy as np
import pandas as pd


def get_chebyshev_index(df, target_column="Y"):
    df = df.interpolate()
    load_df = copy.deepcopy(df)
    shift_column = target_column + "_shift"
    load_df[shift_column] = load_df[target_column].shift(1).fillna(method="bfill").fillna(method="ffill")
    load_df["load_delta"] = load_df[target_column] - load_df[shift_column]
    # get the expect of load delta
    load_delta_mean = np.mean(load_df["load_delta"].values)
    load_delta_sigma = np.std(load_df["load_delta"].values)
    # get the up and down delta
    up_delta = load_delta_mean + 5 * load_delta_sigma
    down_delta = load_delta_mean - 5 * load_delta_sigma
    # select the Not satisfied with Chebyshev inequality
    chebyshev_df = load_df[(load_df["load_delta"] > up_delta) | (load_df["load_delta"] < down_delta)]
    chebyshev_index = list(chebyshev_df.index)
    return chebyshev_index


def process_abnornal_value(df, abnormal_index, target_column="Y", time_step=144):
    data_length = len(df)
    for index in abnormal_index:
        front_pos = index - time_step
        while front_pos >= 0:
            if front_pos not in abnormal_index:
                break
            else:
                front_pos -= time_step
        back_pos = index + time_step
        while back_pos < data_length:
            if back_pos not in abnormal_index:
                break
            else:
                back_pos += time_step
        if front_pos >= 0 and back_pos < data_length:
            front_value = df.loc[front_pos, target_column]
            back_value = df.loc[back_pos, target_column]
            df.loc[index, target_column] = (front_value + back_value) / 2
        else:
            continue
    df = df.interpolate()
    df = df.fillna(method="bfill").fillna(method="ffill")
    return df

# if __name__ == '__main__':
#     load_df = pd.read_csv("../data/data.csv")
#     load_df = load_df[load_df["mode"]==1]
#     load_df.index = [_ for _ in range(len(load_df))]
#     abnormal_index = get_chebyshev_index(load_df)
#     processed_load_df = process_abnornal_value(load_df, abnormal_index)
#     processed_load_df.to_csv("mode=1.csv")
#     pass
