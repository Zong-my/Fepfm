# -*- encoding: utf-8 -*-
'''
@File    :   ieee39_freq_simu_data_v5.py
@Time    :   2025/03/15 15:24:48
@Author  :   myz 
'''
import re
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from fastexcel import read_excel
from concurrent.futures import ThreadPoolExecutor  # Import thread pool module

def extract_digi(catt, ff):
    # 
    steady_fe = {'load_level':0, 'load_zip_z':0, 'load_zip_i':0, 'load_zip_p':0, 
                 'reserve_ratio':0, 'h_inertia':0, 'load_delta':0.0}
    
    if catt == 'cut_machine':
        f1 = ff.split('_')
        for f in f1:
            if f.startswith('le'):
                steady_fe['load_level'] = float(f[2:])
            if f.startswith('zip'):
                tsf = [float(z) for z in f[3:].split('-')]
                steady_fe['load_zip_z'] = float(tsf[0])
                steady_fe['load_zip_i'] = float(tsf[1])
                steady_fe['load_zip_p'] = float(tsf[2])
            if f.startswith('rr'):
                steady_fe['reserve_ratio'] = float(f[2:])
            if f.startswith('hi'):
                steady_fe['h_inertia'] = float(f1[5].split('-')[0][2:])

    if catt == 'load_change':
        tsf = re.findall(r'\d+\.\d+', ff)
        steady_fe['load_level'] = float(tsf[0])
        steady_fe['load_zip_z'] = float(tsf[1])
        steady_fe['load_zip_i'] = float(tsf[2])
        steady_fe['load_zip_p'] = float(tsf[3])
        steady_fe['reserve_ratio'] = float(tsf[4])
        steady_fe['h_inertia'] = float(tsf[5])
        steady_fe['load_delta'] = float(tsf[6])

    if catt == 'circuit_short':
        tsf = re.findall(r'\d+\.\d+', ff)
        steady_fe['load_level'] = float(tsf[0])
        steady_fe['load_zip_z'] = float(tsf[1])
        steady_fe['load_zip_i'] = float(tsf[2])
        steady_fe['load_zip_p'] = float(tsf[3])
        steady_fe['reserve_ratio'] = float(tsf[4])
        steady_fe['h_inertia'] = float(tsf[5])

    return steady_fe

def process_files(cat, files, base_path, new_base_path, hn, ws, base_freq, trigger_time, thread_id):
    """
    处理单个线程的任务，生成独立的CSV文件。
    """
    output_path = os.path.join(new_base_path, f'total_samples_{cat}_thread{thread_id}.csv')
    total_samples = pd.DataFrame()  # 
    for file in tqdm(files, desc=f"Thread {thread_id}"):
        if file.endswith('.xlsx'):
            cat_tmp_df = pd.DataFrame()
            time_index = []  # time index
            try:
                # df_all_sheets = pd.read_excel(os.path.join(base_path, cat, file), sheet_name=None)
                
                # Accelerate Excel file reading using fastexcel
                parser = read_excel(os.path.join(base_path, cat, file))
                sheet_names = parser.sheet_names
                df_all_sheets = {}
                for sheet_name in sheet_names:
                    worksheet = parser.load_sheet_by_name(sheet_name)
                    df = worksheet.to_polars()  # 
                    df_all_sheets[sheet_name] = df

                for sheet_name, df in df_all_sheets.items():
                    tmp_cs = list(df.columns)
                    df = pd.DataFrame(df, columns=tmp_cs)
                    if len(time_index) == 0 and str(df.columns[0]).startswith('__UNNAMED__0'):
                        time_index = df.iloc[:, 0].values.tolist()
                    df = df.loc[:, ~df.columns.str.contains('__UNNAMED__0')]
                    
                    # rename
                    new_columns = []
                    for col in df.columns:
                        new_col_name = f"{sheet_name}_{col}"
                        new_columns.append(new_col_name)
                    
                    df.columns = new_columns
                    cat_tmp_df = pd.concat([cat_tmp_df, df], axis=1)

                # 
                weighted_average_freq = cat_tmp_df[hn].dot(ws)
                cat_tmp_df["FREQ_system"] = weighted_average_freq
                cat_tmp_df.index = time_index
                
                # Extract the extreme values of system frequency and their corresponding timestamps.
                max_abs_value = cat_tmp_df["FREQ_system"].abs().max()
                max_abs_index = cat_tmp_df["FREQ_system"].abs().idxmax()
                t_f_deltamax_row = list(cat_tmp_df.index).index(max_abs_index)
                
                # Locate the index closest to `trigger_time` (the time when the fault is triggered), 
                # and determine `t0` and `tf` in its vicinity (to enable multi-scale feature extraction).
                abs_diff = np.abs(cat_tmp_df.index - trigger_time)
                min_diff = abs_diff.min()
                closest_indices = cat_tmp_df.index[abs_diff == min_diff]
                closest_indices_list = closest_indices.tolist()
                t0, to_row = closest_indices_list[0], list(cat_tmp_df.index).index(closest_indices_list[0])
                
                if t0 >= trigger_time: # The steady-state electrical quantities must be sampled before the trigger time.
                    to_row = to_row - 1
                    t0 = cat_tmp_df.index[to_row]
                
                trans_t0_row = to_row + 1
                while cat_tmp_df.index[trans_t0_row] <= trigger_time:  # The initial transient electrical quantities must be sampled after the trigger time.
                    trans_t0_row = trans_t0_row + 1

                tf0_row = trans_t0_row - 1 # Define the position of the disturbance occurrence.
                
                # Multi-scale Feature Selection Range (T0+1, T0+2, ..., T0+25), within 250 ms after Disturbance
                tf_rows = [int(tf0_row + r) for r in np.arange(1, 26, 1)]

                t_delta = cat_tmp_df.index[t_f_deltamax_row] - trigger_time  # y1
                fpu_deltamax = cat_tmp_df["FREQ_system"].values[t_f_deltamax_row] * base_freq  # y2, Per-unit value restoration

                # Electrical Quantity Features of Generators before Disturbance
                gen_buses = [str(n) for n in range(30, 40, 1)]
                all_columns = cat_tmp_df.columns
                filtered_columns = [col for col in all_columns if col.split('_')[-1] in gen_buses]  # Select only the electrical quantities related to generators.
                stea_fe2 = cat_tmp_df.iloc[[to_row], :][filtered_columns]
                # Multi-scale Electrical Quantity Features of Generators after Disturbance
                dyna_fe = cat_tmp_df.iloc[tf_rows, :][filtered_columns]
                stea_fe2.index = [0]
                dyna_fe.index = range(1, len(dyna_fe) + 1)
                multiscale_fe = pd.concat([stea_fe2, dyna_fe], axis=0)
                # 
                melted_df = pd.melt(multiscale_fe.reset_index(), id_vars='index', var_name='original_column', value_name='value')
                melted_df['new_column'] = melted_df['original_column'] + '_' + melted_df['index'].astype(str)
                final_df = melted_df[['new_column', 'value']].set_index('new_column').T

                # 
                stea_fe1 = extract_digi(cat, file)
                
                # 
                fetures = list(stea_fe1.keys()) + list(final_df.columns)
                feture_values = list(stea_fe1.values()) + list(final_df.values[0])
                tmp_sample = pd.DataFrame([feture_values], columns=fetures)
                
                # tag
                tmp_sample['fpu_deltamax'] = [fpu_deltamax]
                tmp_sample['t_delta'] = [t_delta]
                tmp_sample.insert(0, 'distu_kind', cat)
                tmp_sample.insert(1, 'file_name', file)

                total_samples = pd.concat([total_samples, tmp_sample], axis=0)
                # total_samples.to_csv(output_path, index=False)
            except Exception as e:
                print(e)
    
    # Save the results of the current thread.
    total_samples.to_csv(output_path, index=False)
    return output_path

if __name__ == '__main__':
    base_freq = 60.0
    trigger_time = 1.0
    version_= 'v6_不同工况_未切母线'
    num_threads = 23  # Adjustable parameter: number of threads
    
    # save samples
    total_samples = pd.DataFrame()

    # Inertia coefficients of each generator
    weights = [6.05, 3.41, 6.05, 3.41, 5.016, 3.141, 3.141, 5.32, 500]
    hn = ["FREQ_30", "FREQ_31", "FREQ_32", "FREQ_34", "FREQ_35", "FREQ_36", "FREQ_37", "FREQ_38", "FREQ_39"]
    ws =  [w / sum(weights) for w in weights]

    base_path = f"/mnt/dw_2t/ieee39/{version_}"
    new_base_path = f"/mnt/dw_2t/ieee39/{version_}"
    if not os.path.exists(new_base_path): os.makedirs(new_base_path)
    cats = ['circuit_short', 'cut_machine', 'load_change']
   
    for cat in cats:
        tmp_path = os.path.join(base_path, cat)
        xlsx_files = [f for f in os.listdir(tmp_path) if f.endswith('.xlsx')]
        
        # Split File List into Multiple Threads
        chunk_size = len(xlsx_files) // num_threads
        file_chunks = [xlsx_files[i:i + chunk_size] for i in range(0, len(xlsx_files), chunk_size)]
        if len(file_chunks) > num_threads:  #If there are remaining files, assign them to the last thread.
            file_chunks[-2].extend(file_chunks.pop(-1))
        
        # Parallel Processing Using Thread Pool
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for idx, chunk in enumerate(file_chunks):
                futures.append(executor.submit(process_files, 
                                               cat, 
                                               chunk, 
                                               base_path, 
                                               new_base_path, 
                                               hn, 
                                               ws, 
                                               base_freq, 
                                               trigger_time, 
                                               idx))
            
            # Merge CSV Files Generated by All Threads
            final_df = pd.DataFrame()
            for future in futures:
                thread_output_path = future.result()
                df = pd.read_csv(thread_output_path)
                final_df = pd.concat([final_df, df], axis=0)
            
        # Save the Complete cat CSV File
        final_df.to_csv(os.path.join(new_base_path, f'total_samples_{cat}.csv'), index=False)
    
    # Merge all CSV files across categories (cats) into a single dataset.
    total_df = pd.DataFrame()
    for cat in cats:
        df = pd.read_csv(os.path.join(new_base_path, f"total_samples_{cat}.csv"))
        total_df = pd.concat([total_df, df], axis=0)
    # Save the Final CSV File
    total_df.to_csv(os.path.join(new_base_path, 'total_samples.csv'), index=False)