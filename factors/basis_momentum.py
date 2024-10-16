# 基差动量
# 主力和近月合约，主力和远月合约，主力和次主力合约
# 过去R个交易日累计收益率之差
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
import math
def correct_czc_code(contract, query_date):
    if contract.endswith('.CZC'):
        # 提取期货代码中的字母和数字部分
        match = re.match(r'^([A-Z]+)(\d+)\.CZC$', contract)
        if match:
            letters, numbers = match.groups()
            # 提取查询时间的年份
            string_date = query_date.strftime('%Y')
            year = int(string_date)
            # year = int(query_date[:4])
            # 修正数字部分，在第一位加上年份
            if len(numbers) == 3 and numbers[0] != '9' and year >= 2019:
                corrected_numbers = '2' + numbers
            elif len(numbers) == 3:
                corrected_numbers = '1' + numbers
            else:
                corrected_numbers = numbers
            return f"{letters}{corrected_numbers}.CZC"
    return contract

# daily
def basis_momentum_daily(symbol: str, source_folder1: str,  output_folder: str,N:int):
    file_found = False
    for filename in os.listdir(source_folder1):
        if filename.startswith(f"{symbol}_") and filename.endswith('.csv'):
            nearby_file = os.path.join(source_folder1, filename)
            file_found = True
            break
    if not file_found:
        return f"No CSV file found for symbol: {symbol}"
    df = pd.read_csv(nearby_file)
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
    df = df[(df['S_DQ_CLOSE'] != 0) & (df['S_DQ_CLOSE'].notna())]

    df['S_INFO_WINDCODE'] = df.apply(lambda row: correct_czc_code(row['S_INFO_WINDCODE'], row['TRADE_DT']), axis=1)
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])
    df['prev_close'] = df.groupby('S_INFO_WINDCODE')['S_DQ_CLOSE'].shift(1)
    df = df[(df['prev_close'] != 0) & (df['prev_close'].notna())]
    df['contract_num'] = df['S_INFO_WINDCODE'].str.extract(r'(\d+)').astype(int)
    nearby = df.loc[df.groupby('TRADE_DT')['contract_num'].idxmin()][
        ['TRADE_DT', 'S_INFO_WINDCODE', 'S_DQ_CLOSE', 'prev_close']]
    nearby.rename(columns={'prev_close': 'nearby_prev_close'},inplace=True)


    far = df.loc[df.groupby('TRADE_DT')['contract_num'].idxmax()][
        ['TRADE_DT', 'S_INFO_WINDCODE', 'S_DQ_CLOSE', 'prev_close']]
    far.rename(columns={'prev_close': 'far_prev_close'},inplace=True)

    # 合并 nearby 和 main 数据，按照 trading_date 和 daynight 列
    merged = pd.merge(nearby[['TRADE_DT', 'nearby_prev_close']],far[['TRADE_DT', 'far_prev_close']],
                      on=['TRADE_DT'], how='left')

    merged[f'nearby_{N}days_prev_close'] = merged['nearby_prev_close'].shift(N)
    merged[f'far_{N}days_prev_close'] = merged['far_prev_close'].shift(N)

    # 计算基差动量
    merged[f'{symbol}_{N}days_basis_momentum'] = (
            (merged['nearby_prev_close'] / merged[f'nearby_{N}days_prev_close']) -
            (merged['far_prev_close'] / merged[f'far_{N}days_prev_close'])
    )
    merged = merged[['TRADE_DT',f'{symbol}_{N}days_basis_momentum']]
    merged.rename(columns={'TRADE_DT':'trading_date'})
    # 输出文件名
    output_file_name = f"{symbol}_{N}days_basis_daily_momentum.csv"
    output_file_path = os.path.join(output_folder, output_file_name)

    # 保存为新的CSV文件
    merged.to_csv(output_file_path, index=False)
    print(f"Output saved to {output_file_path}")

def basis_momentum_daynight(symbol: str, source_folder1: str, source_folder2: str, output_folder: str, N:int):
    # 第一步：读取 nearby 文件
    file_found = False
    for filename in os.listdir(source_folder1):
        if filename.startswith(f"{symbol}_") and filename.endswith('.csv'):
            nearby_file = os.path.join(source_folder1, filename)
            file_found = True
            break
    if not file_found:
        return f"No CSV file found for symbol: {symbol}"
    nearby = pd.read_csv(nearby_file)

    # 创建 nearby_prev_close 列
    nearby['nearby_prev_close'] = nearby['close'].shift(1)
    # 提取 nearby 中 contract 列的年和月，并生成 nearby_end_date
    file_found = False
    for filename in os.listdir(source_folder2):
        if filename.startswith(f"{symbol}_") and filename.endswith('.csv'):
            main_file = os.path.join(source_folder2, filename)
            file_found = True
            break
    if not file_found:
        return f"No CSV file found for symbol: {symbol}"
    main = pd.read_csv(main_file)
    # 过滤 tradable 不等于 4 的行
    main = main[main['tradable'] != 4]

    # 重命名 contract 列为 main_contract，避免列名冲突
    main.rename(columns={'contract': 'main_contract'}, inplace=True)

    # 合并 nearby 和 main 数据，按照 trading_date 和 daynight 列
    merged = pd.merge(nearby, main[['trading_date', 'daynight', 'main_contract', 'prev_close']],
                      on=['trading_date', 'daynight'], how='left')

    # merged.rename(columns={'prev_close': 'far_prev_close'})

    merged[f'nearby_{N}days_prev_close'] = merged['nearby_prev_close'].shift(N)
    merged[f'far_{N}days_prev_close'] = merged['prev_close'].shift(N)

    # 计算基差动量
    merged[f'{symbol}_{N}days_basis_momentum'] = (
            (merged['nearby_prev_close'] / merged[f'nearby_{N}days_prev_close']) -
            (merged['prev_close'] / merged[f'far_{N}days_prev_close'])
    )
    merged = merged[['trading_date','daynight',f'{symbol}_{N}days_basis_momentum']]

    # 输出文件名
    output_file_name = f"{symbol}_{N}days_basis_daynight_momentum.csv"
    output_file_path = os.path.join(output_folder, output_file_name)

    # 保存为新的CSV文件
    merged.to_csv(output_file_path, index=False)
    print(f"Output saved to {output_file_path}")


if __name__ == '__main__':
    symbol_list = ["A", "AG", "AL", "AO", "AP", "AU", "B", "BB", "BC", "BR", "BU", "C", "CF", 'CJ', 'CS', 'CU', 'CY', 'EB',
                   'EC', 'EG', 'FB', 'FG', 'FU', 'HC', 'I', 'IC',
                   'IF', 'IH', 'IM', 'J', 'JD', 'JM', 'JR', 'L', 'LH', 'LU', 'M', 'MA', 'ME', 'NI', 'NR', 'OI', 'P', 'PB',
                   'PF', 'PG', 'PK', 'PM', 'PP', 'PX', 'RB', 'RI', 'RM', 'RO',
                   'RR', 'RS', 'RU', 'SA', 'SC', 'SF', 'SH', 'SM', 'SN', 'SP', 'SR', 'SS', 'T', 'TA', 'TC', 'TF', 'TL',
                   'TS', 'UR', 'V', 'WH', 'WR', 'WS', 'WT', 'Y', 'ZC', 'ZN']
    source_folder1 = r'\\samba-1.quantchina.pro\quanyi4g\data\future\daybar'
    output_folder = r'\\samba-1.quantchina.pro\quanyi4g\data\future\factor\term_structure\basis_momentum_daily'

    daynight_source1 = r'\\samba-1.quantchina.pro\quanyi4g\data\future\trade_buffer\buffer_day_nearby'
    daynight_souurce2 = r'\\samba-1.quantchina.pro\quanyi4g\temporary\Steiner\data_wash\linux_so\py311\snapshot_results_oi'
    daynight_output = r'\\samba-1.quantchina.pro\quanyi4g\data\future\factor\term_structure\basis_momentum_daynight'
    N_list = [10,22,66,252]
    for symbol in symbol_list:
        for N in N_list:
            basis_momentum_daynight(symbol, daynight_source1, daynight_souurce2, daynight_output, N)