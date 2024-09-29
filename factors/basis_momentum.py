# 基差动量
# 主力和近月合约，主力和远月合约，主力和次主力合约
# 月频，wind重采样。选择合约——计算月末的ln收盘价——计算月价差
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
import math
def correct_czc_code(contract, query_date):
    '''
    处理wind中郑商所的合约代码，将wind中的合约代码转换成标准的四位数字形式
    '''
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
    '''
    计算每日基差动量（basis momentum），根据最近和远期合约的前N天的收盘价变化，计算基差动量并输出结果到CSV文件。

    参数：
    - symbol: 商品代码
    - source_folder1: 存放合约数据的文件夹路径
    - output_folder: 输出结果的文件夹路径
    - N: 计算动量时的时间窗口（N天）
    '''
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
    '''
    计算日夜盘的基差动量。基于不同时间段内的最近期和最远期合约的前N天的收盘价变化，计算基差动量并输出结果到CSV文件。

    参数：
    - symbol: 商品代码
    - source_folder1: 存放合约数据的文件夹路径（包含最近期合约）
    - source_folder2: 存放合约数据的文件夹路径（包含主力合约）
    - output_folder: 输出结果的文件夹路径
    - N: 计算动量时的时间窗口（N天）
    '''
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


# 重采样，对于每个S_INFO_WINDCODE选择每个月末的最后一个日期。如果日期的日数在20号以前就删掉。计算ln收盘价。新增一列标记标记月份。对于每一个月份选择最大的合约标记为1，第二大的标记为2，第三大的标记为3，直到全部标记。
# 斜率：选择上个月标记为1和标记为4的合约ln价格相减，减去13个月前标记为1和4的合约ln价格相减。

def slope_factor_daily(symbol: str, source_folder1: str,  output_folder: str):
    '''
    计算每日的斜率因子（slope factor）。通过每个月的合约标记，计算上个月和13个月前的标记1和4的合约的价格差，得出斜率因子，并保存到CSV文件。

    参数：
    - symbol: 商品代码
    - source_folder1: 存放合约数据的文件夹路径
    - output_folder: 输出结果的文件夹路径
    '''
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

    df['month'] = df['TRADE_DT'].dt.to_period('M')
    last_day_df = df[df['TRADE_DT'].dt.day > 20].groupby(['S_INFO_WINDCODE', 'month']).apply(
        lambda x: x.loc[x['TRADE_DT'].idxmax()]).reset_index(drop=True)
    last_day_df['ln_price'] = np.log(last_day_df['S_DQ_CLOSE'])

    last_day_df['tag'] = last_day_df.groupby('month')['contract_num'].rank(method='first', ascending=False).astype(int)

    factor_values = []
    for current_month in last_day_df['month'].unique():
        # 当前月的数据
        current_month_data = last_day_df[last_day_df['month'] == current_month]

        # 上个月的数据
        prev_month = (current_month - 1).strftime('%Y-%m')
        prev_month_data = last_day_df[last_day_df['month'] == prev_month]

        # 13个月前的数据
        prev_13_month = (current_month - 13).strftime('%Y-%m')
        prev_13_month_data = last_day_df[last_day_df['month'] == prev_13_month]

        # 获取上个月和13个月前tag为1和4的合约
        prev_1 = prev_month_data[prev_month_data['tag'] == 1]
        prev_4 = prev_month_data[prev_month_data['tag'] == 4]

        prev_13_1 = prev_13_month_data[prev_13_month_data['tag'] == 1]
        prev_13_4 = prev_13_month_data[prev_13_month_data['tag'] == 4]

        # 确保有足够数据进行计算
        if not prev_1.empty and not prev_4.empty and not prev_13_1.empty and not prev_13_4.empty:
            slope_value = (prev_1['ln_price'].values[0] - prev_4['ln_price'].values[0]) - \
                          (prev_13_1['ln_price'].values[0] - prev_13_4['ln_price'].values[0])
            factor_values.append((current_month.end_time, slope_value,current_month))
    slope_factor_df = pd.DataFrame(factor_values, columns=['TRADE_DT', f'{symbol}_slope_factor','month'])

    filtered_df = df.drop_duplicates(subset=['TRADE_DT'], keep='first')
    # filtered_df['month'] = filtered_df['TRADE_DT'].dt.to_period('M')
    # slope_factor_df['month'] = slope_factor_df['TRADE_DT'].dt.to_period('M')

    output_df = pd.merge(filtered_df, slope_factor_df, on='month', how='left', suffixes=('', '_slope'))
    print(output_df.columns)
    output_df = output_df[['TRADE_DT',f'{symbol}_slope_factor']]

    output_df.rename(columns={'TRADE_DT': 'trading_date'}, inplace=True)
    # print(output_df['trading_date'])
    output_df = output_df[output_df['trading_date']>=pd.to_datetime('2014-01-01')]
    # 输出文件名
    output_file_name = f"{symbol}_slope_factor.csv"
    output_file_path = os.path.join(output_folder, output_file_name)

    # 保存为新的CSV文件
    output_df.to_csv(output_file_path, index=False)
    print(f"Output saved to {output_file_path}")


# 曲率：对于13个月前到1个月前的每个月，计算（标记1和4，标记2和5的合约之间的价格差值）然后求和
def curvature_daily(symbol: str, source_folder1: str,  output_folder: str):
    '''
    计算每日的曲率因子（curvature factor）。通过计算不同标记的合约之间的价格差值，求和得到曲率因子，并输出结果到CSV文件。

    参数：
    - symbol: 商品代码
    - source_folder1: 存放合约数据的文件夹路径
    - output_folder: 输出结果的文件夹路径
    '''
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

    df['month'] = df['TRADE_DT'].dt.to_period('M')
    last_day_df = df[df['TRADE_DT'].dt.day > 20].groupby(['S_INFO_WINDCODE', 'month']).apply(
        lambda x: x.loc[x['TRADE_DT'].idxmax()]).reset_index(drop=True)
    last_day_df['ln_price'] = np.log(last_day_df['S_DQ_CLOSE'])

    last_day_df['tag'] = last_day_df.groupby('month')['contract_num'].rank(method='first', ascending=False).astype(int)

    def calculate_curvature_factor(resample_df):
        # 创建时间窗口列，以便方便后面筛选
        resample_df['month_offset'] = resample_df['month'].apply(lambda x: x - 1)  # 当前月为 month - 1
        resample_df['month'] = resample_df['month'].dt.to_timestamp()

        # 提取标记 1 和 4 的差值以及标记 2 和 5 的差值
        tag_diff = resample_df.pivot_table(index='month', columns='tag', values='ln_price')
        if 5 not in tag_diff.columns or 4 not in tag_diff.columns:
            print("Warning: Tag 5 not found in the dataset.")
            return pd.DataFrame(columns=['TRADE_DT', f'{symbol}_curvature_factor', 'month'])
        # 计算 tag 1 和 4，tag 2 和 5 的差值
        tag_diff['diff_1_4'] = tag_diff[1] - tag_diff[4]
        tag_diff['diff_2_5'] = tag_diff[2] - tag_diff[5]

        # 计算最终的差值：diff_1_4 - diff_2_5
        tag_diff['curvature_diff'] = tag_diff['diff_1_4'] - tag_diff['diff_2_5']
        tag_diff = tag_diff.dropna(subset=['diff_1_4', 'diff_2_5'])
        # 对于每个当前月，计算时间窗口的累计值
        curvature_values = []
        for current_month in tag_diff.index:
            start_month = current_month - pd.DateOffset(months=13)
            end_month = current_month - pd.DateOffset(months=2)

            # 筛选13个月前到2个月前的数据并进行累加
            window_data = tag_diff.loc[start_month:end_month, 'curvature_diff'].sum()

            # 将计算结果保存
            curvature_values.append((current_month, window_data))

        # 转换为 DataFrame
        curvature_factor_df = pd.DataFrame(curvature_values, columns=['TRADE_DT', f'{symbol}_curvature_factor'])
        curvature_factor_df['month'] = curvature_factor_df['TRADE_DT'].dt.to_period('M')
        return curvature_factor_df

        # 确保有足够数据进行计算
    curvature_factor_df = calculate_curvature_factor(last_day_df)

    filtered_df = df.drop_duplicates(subset=['TRADE_DT'], keep='first')


    output_df = pd.merge(filtered_df, curvature_factor_df, on='month', how='left', suffixes=('', '_slope'))
    print(output_df.columns)
    output_df = output_df[['TRADE_DT',f'{symbol}_curvature_factor']]

    output_df.rename(columns={'TRADE_DT': 'trading_date'}, inplace=True)
    # print(output_df['trading_date'])
    output_df = output_df[output_df['trading_date']>=pd.to_datetime('2014-01-01')]
    # 输出文件名
    output_file_name = f"{symbol}_curvature_factor.csv"
    output_file_path = os.path.join(output_folder, output_file_name)

    # 保存为新的CSV文件
    output_df.to_csv(output_file_path, index=False)
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
    daynight_souurce2 = r'\\samba-1.quantchina.pro\quanyi4g\temporary\Steiner\data_wash\linux_so\py311\snapshoot_results_oi'
    daynight_output = r'\\samba-1.quantchina.pro\quanyi4g\data\future\factor\term_structure\basis_momentum_daynight'

    output_folder3 = r'\\samba-1.quantchina.pro\quanyi4g\data\future\factor\term_structure\slope_monthly'
    output_folder4 = r'\\samba-1.quantchina.pro\quanyi4g\data\future\factor\term_structure\curvature_monthly'
    N_list = [10,22,66,252]
    for symbol in symbol_list:
            curvature_daily(symbol, source_folder1,output_folder4)