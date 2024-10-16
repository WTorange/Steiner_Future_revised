import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
import math
from pathlib import Path
## 计算因子值和之后N天的收益率（价格指数）的相关性，
# 因子值采用加权动量和截面动量的因子值，加权动量5,10,22,66，N天取1，3,5,10,20,30,50。
# 收益率采用snapshot中的close计算，由于股指期货都是daily，可以不用考虑夜盘问题。出于兼容性考虑，进行重采样后计算，可以直接使用basis_momentum中的代码
# 将所有的因子值和收益率的日期对齐合并到一个df中，进行相关性计算，输出图表。


#

def Ndays_return(symbol: str, source_folder: str, source_folder2: str,output_folder: str,factor:str):
    '''
    Holding_length.py
    计算交易信号与之后M天收益率的相关系数，确定不同因子的持有期。
    输入：期货品种名称，因子文件夹，价格数据文件夹，输出文件夹
    输出：相关性与相关稀释矩阵，png图片

    '''
    N_values = [1, 3, 5, 10, 20, 30, 50,70,90,110,126,180,240,300,360,420,480,504]
    source_path = Path(source_folder)

    source_path2  = Path(source_folder2)
    file_list2 = list(source_path2.glob(f'{symbol}_*.csv'))
    # 查找以 symbol 开头的文件
    file_list = list(source_path.glob(f'{symbol}_*.csv'))

    if not file_list:
        print(f"No files found for {symbol}")
        return

    # 读取第一个匹配的文件
    file_path = file_list[0]
    data = pd.read_csv(file_path)

    # 验证 symbol 列第一个值是否匹配
    if data['symbol'].iloc[0] != symbol:
        print(f"Symbol mismatch in file {file_path.name}")
        return

    # 过滤掉 'query_notrade' 不为 0 的行，并以交易日分组获取最后记录
    data = data[data['query_notrade'] == 0].groupby('trading_date').last().reset_index()

    momentum_df = pd.DataFrame(data['trading_date'])

    # 计算每个N的收益率，并添加到 DataFrame 中
    for N in N_values:
        momentum_df[f'{N}_days_return'] = data['last_prc'].shift(-N) / data['last_prc'] - 1

    # 计算 N 天后的收益率：利用 shift(-N) 来向下平移 N 天，直接计算收益率
    for file_path2 in file_list2:
        file_data = pd.read_csv(file_path2)

        # 合并时按trading_date对齐
        momentum_df = pd.merge(momentum_df, file_data, on='trading_date', how='left',
                               suffixes=('', f'_{file_path2.stem}'))

    new_columns = [col for col in momentum_df.columns if col not in ['trading_date'] + [f'{N}_days_return' for N in N_values]]

    # 转化为0,1信号
    for col in new_columns:
        momentum_df[col] = momentum_df[col].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # 计算每个N天收益率与新合并列的相关系数并计算平均值
    corr_results = {}
    avg_corr_results = {}

    for N in N_values:
        corr_values = []
        for col in new_columns:
            # 计算相关系数
            corr_value = momentum_df[f'{N}_days_return'].corr(momentum_df[col])
            print(corr_value)
            corr_values.append(corr_value)

        # 存储每个N的相关系数列表
        corr_results[f'{N}_days_return'] = corr_values

        # 计算并存储每个N的相关系数平均值
        avg_corr_results[f'{N}_days_return'] = np.mean(corr_values)

    # 将相关系数和平均值结果转换为 DataFrame
    corr_df = pd.DataFrame(corr_results, index=new_columns)

    corr_df = corr_df.sort_index(ascending=True)

    corr_df['Row_Avg'] = corr_df.mean(axis=1)

    # 计算每列的平均值
    col_avg = corr_df.mean(axis=0)

    # 将列平均值添加为最后一行
    corr_df.loc['Col_Avg'] = col_avg
    avg_corr_df = pd.DataFrame.from_dict(avg_corr_results, orient='index', columns=['Average Correlation'])

    # 保存结果
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存相关系数矩阵
    corr_output_file = output_path / f"{symbol}_{factor}_correlation_matrix.csv"
    corr_df.to_csv(corr_output_file)

    # 保存平均相关系数
    avg_corr_output_file = output_path / f"{symbol}_{factor}_average_correlation.csv"
    # avg_corr_df.to_csv(avg_corr_output_file)

    print(f"Correlation matrix saved to {corr_output_file}")
    print(f"Average correlation saved to {avg_corr_output_file}")

if __name__ == '__main__':
    # 使用时调用该函数，传入symbol和路径
    symbol_list = ['IC','IF','IH','IM']
    source_folder = r'\\samba-1.quantchina.pro\quanyi4g\temporary\Steiner\data_wash\linux_so\py311\snapshot_index_temp'
    source_folder2 = r'\\samba-1.quantchina.pro\quanyi4g\data\future\factor\momentum\cs_momentum\basic_cs_momentum'
    output_folder = r'\\samba-1.quantchina.pro\quanyi4g\temporary\Steiner\data_wash\linux_so\py311\corr'
    for symbol in symbol_list:
        Ndays_return(symbol, source_folder, source_folder2, output_folder,'basic_cs')

