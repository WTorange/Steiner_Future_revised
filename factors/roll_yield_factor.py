# 远月合约：改用主力合约
# 合约到期日：统一规定为月的最后一天

# 因子值：用前收计算
# 分daily和daynight
# 主力合约snapshoot，近月合约snapshoot，prev_close

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



def roll_yield_daynight(symbol: str, source_folder1: str, source_folder2: str, output_folder: str):
    '''
    计算 daynight 数据的滚动收益率并保存到 CSV 文件。

    参数:
    symbol (str): 合约符号。
    source_folder1 (str): 存储 nearby 数据的文件夹路径。
    source_folder2 (str): 存储 main 数据的文件夹路径。
    output_folder (str): 保存输出文件的文件夹路径。

    输出:
    str: 如果找不到对应的 CSV 文件，返回错误信息，否则返回 None，数据保存在输出文件中。
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
    nearby['nearby_end_date'] = nearby['contract'].apply(lambda x: get_end_date(x))
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

    main.columns = main.columns.str.strip()
    main = main.groupby(['trading_date','daynight']).first().reset_index()
    print(main)
    # 提取 main_contract 的年和月，并生成 main_end_date
    main['main_end_date'] = main['contract'].apply(lambda x: get_end_date(x))
    # 重命名 contract 列为 main_contract，避免列名冲突
    main.rename(columns={'contract': 'main_contract'}, inplace=True)

    # 合并 nearby 和 main 数据，按照 trading_date 和 daynight 列
    merged = pd.merge(nearby, main[['trading_date', 'daynight', 'main_contract', 'prev_close']],
                      on=['trading_date', 'daynight'], how='left')

    def calculate_roll_yield(row):
        if row['days_diff'] == 0:
            return 0  # 当 days_diff 为 0 时，返回 0
        try:
            result = (np.log(row['nearby_prev_close']) - np.log(row['prev_close'])) * 365 / row['days_diff']
            if not isinstance(result, float):  # 检查结果是否为 float 类型
                print(f"Warning: Non-float value encountered for {symbol}_roll_yield_daynight: {result}")
            return result
        except Exception as e:
            print(f"Error calculating roll yield for row: {row}, error: {e}")
            return np.nan  # 出现错误时返回 NaN
    # 计算日期差值（单位：天）
    merged['main_contract']=merged['main_contract'].astype(str)

    merged['main_end_date'] = merged['main_contract'].apply(lambda x: get_end_date(x))
    merged['nearby_end_date'] = pd.to_datetime(merged['nearby_end_date'], format='%Y%m%d')
    merged['main_end_date'] = pd.to_datetime(merged['main_end_date'], format='%Y%m%d')
    merged['days_diff'] = (merged['main_end_date'] - merged['nearby_end_date']).dt.days

    merged[f'{symbol}_roll_yield_daynight'] = merged.apply(calculate_roll_yield, axis=1)
    # 计算 {symbol}_roll_yield_daynight 列
    # merged[f'{symbol}_roll_yield_daynight'] = (np.log(merged['nearby_prev_close']) -
    #                                            np.log(merged['prev_close'])) * 365 / merged['days_diff']

    # 选择所需的列
    result = merged[['trading_date', 'daynight', f'{symbol}_roll_yield_daynight']]

    # 保存结果到文件
    output_file = os.path.join(output_folder, f"{symbol}_roll_yield_daynight.csv")
    result.to_csv(output_file, index=False)

    print(f"Roll yield daynight data saved to {output_file}")

def roll_yield_daily(symbol: str, source_folder1: str,  output_folder: str):
    '''
    计算 daily 数据的滚动收益率并保存到 CSV 文件。

    参数:
    symbol (str): 合约符号。
    source_folder1 (str): 存储 nearby 数据的文件夹路径。
    output_folder (str): 保存输出文件的文件夹路径。

    输出:
    str: 如果找不到对应的 CSV 文件，返回错误信息，否则返回 None，数据保存在输出文件中。
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

    nearby['nearby_end_date'] = nearby['S_INFO_WINDCODE'].apply(lambda x: get_end_date(x))
    far['far_end_date'] = far['S_INFO_WINDCODE'].apply(lambda x: get_end_date(x))



    # 合并 nearby 和 main 数据，按照 trading_date 和 daynight 列
    merged = pd.merge(nearby[['TRADE_DT', 'nearby_prev_close', 'nearby_end_date']],far[['TRADE_DT', 'far_prev_close', 'far_end_date']],
                      on=['TRADE_DT'], how='left')

    def calculate_roll_yield(row):
        '''
        计算 daily 滚动收益率。

        参数:
        row (pd.Series): 包含合约日期差、nearby 和 far 的前一日收盘价。

        输出:
        float: 滚动收益率，若发生错误或日期差为 0，返回 0 或 NaN。
        '''
        if row['days_diff'] == 0:
            return 0  # 当 days_diff 为 0 时，返回 0
        try:
            result = (np.log(row['nearby_prev_close']) - np.log(row['far_prev_close'])) * 365 / row['days_diff']
            if not isinstance(result, float):  # 检查结果是否为 float 类型
                print(f"Warning: Non-float value encountered for {symbol}_roll_yield_daily: {result}")
            return result
        except Exception as e:
            print(f"Error calculating roll yield for row: {row}, error: {e}")
            return np.nan  # 出现错误时返回 NaN

    merged['nearby_end_date'] = pd.to_datetime(merged['nearby_end_date'], format='%Y%m%d')
    merged['far_end_date'] = pd.to_datetime(merged['far_end_date'], format='%Y%m%d')
    merged['days_diff'] = (merged['far_end_date'] - merged['nearby_end_date']).dt.days
    merged[f'{symbol}_roll_yield_daily'] = merged.apply(calculate_roll_yield, axis=1)
    merged.rename(columns={'TRADE_DT':'trading_date'},inplace=True)
    # 选择所需的列
    result = merged[['trading_date', f'{symbol}_roll_yield_daily']]

    # 保存结果到文件
    output_file = os.path.join(output_folder, f"{symbol}_roll_yield_daily.csv")
    print(output_file)
    result.to_csv(output_file, index=False)
# 辅助函数：提取 contract 列中的 yymm 并生成该月份的最后一天
def get_end_date(contract: str) -> str:
    '''
    计算合约的到期日，默认为到期月份的最后一天
    '''
    match = re.search(r'(\d{4})', contract)
    if match:
        yymm = match.group(1)
        year = '20' + yymm[:2]  # 提取年
        month = yymm[2:]  # 提取月
        # 生成该月份的最后一天
        end_date = pd.Timestamp(f'{year}-{month}-01') + pd.offsets.MonthEnd(0)
        return end_date.strftime('%Y%m%d')
    return None


if __name__ == '__main__':

    source_folder1 = r"\\samba-1.quantchina.pro\quanyi4g\data\future\trade_buffer\buffer_day_nearby"
    source_folder2 = r"\\samba-1.quantchina.pro\quanyi4g\temporary\Steiner\data_wash\linux_so\py311\snapshoot_results_oi"
    output_folder = r"\\samba-1.quantchina.pro\quanyi4g\data\future\factor\term_structure\roll_yield_daynight"
    output_folder2 = r"\\samba-1.quantchina.pro\quanyi4g\data\future\factor\roll_yield\roll_yield_daily"
    daily_source_folder1 = r'\\samba-1.quantchina.pro\quanyi4g\data\future\daybar'

    symbol_list = ["A", "AG", "AL", "AO", "AP", "AU", "B", "BB", "BC", "BR", "BU", "C", "CF", 'CJ', 'CS', 'CU', 'CY', 'EB',
                   'EC', 'EG', 'FB', 'FG', 'FU', 'HC', 'I', 'IC',
                   'IF', 'IH', 'IM', 'J', 'JD', 'JM', 'JR', 'L', 'LH', 'LU', 'M', 'MA', 'ME', 'NI', 'NR', 'OI', 'P', 'PB',
                   'PF', 'PG', 'PK', 'PM', 'PP', 'PX', 'RB', 'RI', 'RM', 'RO',
                   'RR', 'RS', 'RU', 'SA', 'SC', 'SF', 'SH', 'SM', 'SN', 'SP', 'SR', 'SS', 'T', 'TA', 'TC', 'TF', 'TL',
                   'TS', 'UR', 'V', 'WH', 'WR', 'WS', 'WT', 'Y', 'ZC', 'ZN']
    # symbol_list = ['JD']
    for symbol in symbol_list:
        roll_yield_daynight(symbol, source_folder1,source_folder2, output_folder)

    # AG = pd.read_csv(r"\\samba-1.quantchina.pro\quanyi4g\data\future\factor\roll_yield\reports\roll_yield_daynight\AG_roll_yield_daynight_detail.csv")
    # AG['log_value']=np.log(AG['theo_net_value2'])
    # AG['yield_sum']=AG['theo_yields'].cumsum()
    # plt.figure(figsize=(10, 6))
    # plt.plot(AG['date'], AG['yield_sum'], label='theo_yields')
    #
    # # 添加标题和标签
    # # plt.title('Log Value of AG over Time')
    # plt.xlabel('Date')
    # plt.ylabel('Log Value')
    #
    # # 添加网格和图例
    # plt.grid(True)
    # plt.legend()

    # 显示图像
    plt.show()
