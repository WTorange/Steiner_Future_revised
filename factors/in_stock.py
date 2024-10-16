import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
import math
# 库存、仓单因子，从wind的表格中获取，因此只有daily。
# 表格中只有中文，数据离散，需要重新对应和填充。从“合约信息大全”中匹
# IN_STOCK列有值率高

# 读取 in_stock.csv 文件
def combine_symbol():
    in_stock_df = pd.read_csv('in_stock.csv')
    #
    # 读取 合约信息大全.csv 文件
    contract_info_df = pd.read_csv('合约信息大全.csv')
    contract_info_df = contract_info_df.drop_duplicates(subset='S_INFO_NAME', keep='first')
    # 通过 FS_INFO_SCNAME 列和 S_INFO_NAME 列进行匹配，将 S_INFO_CODE 列的信息添加到 in_stock.csv 中
    merged_df = pd.merge(in_stock_df, contract_info_df[['S_INFO_NAME', 'S_INFO_CODE']],
                         left_on='FS_INFO_SCNAME', right_on='S_INFO_NAME', how='left')

    # 将 S_INFO_CODE 列重命名为 symbol
    merged_df.rename(columns={'S_INFO_CODE': 'symbol'}, inplace=True)

    # 保存结果到新的 CSV 文件
    merged_df.to_csv('in_stock_with_symbol.csv', index=False)

def in_stock_daily(symbol,output_folder, N):
    merged_df = pd.read_csv('in_stock_with_symbol.csv')
    merged_df['ANN_DATE'] = pd.to_datetime(merged_df['ANN_DATE'], format='%Y%m%d')
    all_dates = pd.Series(pd.to_datetime(merged_df['ANN_DATE'].unique())).sort_values()

    # 筛选出特定symbol的数据
    symbol_df = merged_df[(merged_df['symbol'] == symbol) & (merged_df['IN_STOCK']!=0)].copy()
    symbol_df = symbol_df[(symbol_df['ANN_DATE']>='2014-01-01') & (symbol_df['ANN_DATE']<='2024-07-10')]

    # 确保symbol的数据也有完整的日期索引，重新索引以填充日期缺失的行
    symbol_df = symbol_df.set_index('ANN_DATE').reindex(all_dates)
    symbol_df = symbol_df[symbol_df.index >= '2014-01-01']
    # 填充IN_STOCK中的缺失值
    symbol_df['IN_STOCK'] = symbol_df['IN_STOCK'].ffill()
    # symbol_df.to_csv('check.csv')
    # 计算前1个日期和前N+1个日期的IN_STOCK比率
    symbol_df['IN_STOCK_T-1'] = symbol_df['IN_STOCK'].shift(1)
    symbol_df['IN_STOCK_T-(N+1)'] = symbol_df['IN_STOCK'].shift(N+1)

    # 计算因子：前1个日期的IN_STOCK / 前N+1个日期的IN_STOCK
    symbol_df[f'{N}days_in_stock_daily'] = symbol_df['IN_STOCK_T-1'] / symbol_df['IN_STOCK_T-(N+1)'] -1
    symbol_df.reset_index(inplace=True)
    # print(symbol_df)
    symbol_df.rename(columns={'index':'trading_date'},inplace=True)
    output_df = symbol_df[['trading_date', f'{N}days_in_stock_daily']]



    output_file = f"{symbol}_{N}days_in_stock_daily.csv"
    output_path = os.path.join(output_folder, output_file)
    output_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved: {output_path}")



if __name__ == '__main__':
    symbol_list = ["A", "AG", "AL", "AO", "AP", "AU", "B", "BB", "BC", "BR", "BU", "C", "CF", 'CJ', 'CS', 'CU', 'CY', 'EB',
                   'EC', 'EG', 'FB', 'FG', 'FU', 'HC', 'I', 'IC',
                   'IF', 'IH', 'IM', 'J', 'JD', 'JM', 'JR', 'L', 'LH', 'LU', 'M', 'MA', 'ME', 'NI', 'NR', 'OI', 'P', 'PB',
                   'PF', 'PG', 'PK', 'PM', 'PP', 'PX', 'RB', 'RI', 'RM', 'RO',
                   'RR', 'RS', 'RU', 'SA', 'SC', 'SF', 'SH', 'SM', 'SN', 'SP', 'SR', 'SS', 'T', 'TA', 'TC', 'TF', 'TL',
                   'TS', 'UR', 'V', 'WH', 'WR', 'WS', 'WT', 'Y', 'ZC', 'ZN']

    daynight_output = r'\\samba-1.quantchina.pro\quanyi4g\data\future\factor\in_stock\in_stock_daily'
    N_list = [5,10,22,66,252]
    for symbol in symbol_list:
        for N in N_list:
            in_stock_daily(symbol, daynight_output, N)
