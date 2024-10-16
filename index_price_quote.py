import os
import re
import pandas as pd
import numpy as np
from datetime import datetime


def process_futures_data(symbol):
    base_dir = "/nas92/data/future/quote"
    output_dir = f"/nas92/data/future/{symbol}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    date_folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])
    symbol_pattern = re.compile(f"^{re.escape(symbol)}\\d{{3,4}}\\..*$")

    for date_folder in date_folders:
        output_filename = f"{symbol}_{date_folder}_weightedaverage.parquet"
        output_filepath = os.path.join(output_dir, output_filename)

        if os.path.exists(output_filepath):
            print(f'Skip alreading processed file: {output_filepath}')
            continue

        print(date_folder)
        date_path = os.path.join(base_dir, date_folder)
        main_path = os.path.join(date_path, "main_contract")
        second_path = os.path.join(date_path, "second")
        third_path = os.path.join(date_path, "third")

        # Collect matching files from each folder
        matching_files = []
        for folder in [main_path, second_path, third_path]:
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if symbol_pattern.match(f):
                        file_path = os.path.join(folder, f)
                        df = pd.read_parquet(file_path)
                        if not df.empty:
                            matching_files.append(df)

        if not matching_files:
            print(f"No matching files for {symbol} on {date_folder}")
            continue

        combined_df = pd.concat(matching_files, ignore_index=True)

        if 'resample_time' not in combined_df.columns:
            print(f"'resample_time' column not found in combined DataFrame for {symbol} on {date_folder}")
            continue

        # 计算加权平均
        grouped = combined_df.groupby('resample_time').agg(
            last_prc=('last_prc', lambda x: np.average(x, weights=combined_df.loc[x.index, 'open_interest'])),
            ask_prc1=('ask_prc1', lambda x: np.average(x, weights=combined_df.loc[x.index, 'open_interest'])),
            bid_prc1=('bid_prc1', lambda x: np.average(x, weights=combined_df.loc[x.index, 'open_interest'])),
            middle_price=('middle_price', lambda x: np.average(x, weights=combined_df.loc[x.index, 'open_interest'])),
            volume=('volume', 'sum'),
            turnover=('turnover', 'sum'),
            open_interest=('open_interest', 'sum'),
            trading_date=('trading_date', 'first')
        ).reset_index()

        grouped['future'] = symbol

        grouped.to_parquet(output_filepath, index=False)

if __name__ == '__main__':
    future_list=['AU','ZC','RB','PP','PX']
# 使用方法
    for future in future_list:
        print(future)
        process_futures_data(future)

