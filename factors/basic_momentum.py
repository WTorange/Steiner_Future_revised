import os
import glob
import pandas as pd
import numpy as np

def calculate_basic_momentum(symbol: str, N: int, source_folder: str, output_folder:str):
    """
    计算基于前N个交易日的（长短）动量因子。

    :param symbol: 期货标的符号，例如"AU", "RB"。
    :param N: 前N个交易日作为参考。
    :param source_folder: 存储输入数据的文件夹路径。
    :param output_folder: 存储输出结果的文件夹路径。
    """
    # 查找以 symbol 开头的文件
    file_list = [f for f in os.listdir(source_folder) if f.startswith(symbol + "_") and f.endswith('.csv')]

    if not file_list:
        print(f"No files found for {symbol}")
        return

    # 读取第一个匹配的文件
    file_path = os.path.join(source_folder, file_list[0])
    data = pd.read_csv(file_path)

    # 核验 symbol 列的第一个值是否等于输入的 symbol
    if data['symbol'].iloc[0] != symbol:
        print(f"Symbol mismatch in file {file_list[0]}")
        return
    # 添加 daynight2 列
    data['trading_date'] = pd.to_datetime(data['trading_date'], format='%Y-%m-%d')
    data['query_time'] = pd.to_datetime(data['query_time'])
    data['daynight2'] = np.where(
        data['query_time'].dt.time < pd.Timestamp('08:00:00').time(),
        'night',
        np.where(data['query_time'].dt.time < pd.Timestamp('20:00:00').time(), 'day', 'night')
    )

    # 过滤出 query_notrade == 0 的数据
    filtered_data = data[data['query_notrade'] == 0]

    def calculate_momentum(row):
        current_index = row.name
        current_daynight = row['daynight2']

        # 获取row行及之前的数据
        filtered_data = data.loc[:current_index].copy()

        # 筛选daynight2相同的数据
        filtered_data = filtered_data[filtered_data['daynight2'] == current_daynight]

        # 找到最大的N+1个交易日期
        unique_trading_dates = filtered_data['trading_date'].drop_duplicates(keep='last')
        if len(unique_trading_dates) < N + 1:
            return np.nan

        selected_dates = unique_trading_dates.tail(N + 1)
        new_filtered_data = filtered_data[filtered_data['trading_date'].isin(selected_dates)]


        last_price_A = row['last_prc']
        last_price_B = new_filtered_data.iloc[0]['last_prc']
        # print(new_filtered_data.iloc[0])
        # print(filtered_days)

        return last_price_A / last_price_B - 1


    filtered_data[f'{N}days_basic_momentum'] = filtered_data.apply(calculate_momentum, axis=1)
    basic_mom = filtered_data[['trading_date', 'daynight2', f'{N}days_basic_momentum']]
    basic_mom.rename(columns={'daynight2':'daynight'},inplace=True)

    output_file = f"{symbol}_{N}days_basic_momentum.csv"
    output_path = os.path.join(output_folder, output_file)
    basic_mom.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved: {output_path}")

def basic_daily_momentum(symbol: str, N: int, source_folder: str, output_folder: str):
    """
    计算基于前N个交易日的长短动量因子。

    :param symbol: 期货标的符号，例如"AU", "RB"。
    :param N: 前N个交易日作为参考。
    :param source_folder: 存储输入数据的文件夹路径。
    :param output_folder: 存储输出结果的文件夹路径。
    """

    # 查找以 symbol 开头的文件
    file_list = [f for f in os.listdir(source_folder) if f.startswith(symbol + "_") and f.endswith('.csv')]

    if not file_list:
        print(f"No files found for {symbol}")
        return

    # 读取第一个匹配的文件
    file_path = os.path.join(source_folder, file_list[0])
    data = pd.read_csv(file_path)

    # 核验 symbol 列的第一个值是否等于输入的 symbol
    if data['symbol'].iloc[0] != symbol:
        print(f"Symbol mismatch in file {file_list[0]}")
        return
    filtered_data = data[data['query_notrade'] == 0]
    resampled_data = filtered_data.groupby('trading_date').last()
    resampled_prices = resampled_data['last_prc']
    def calculate_long_minus_short_momentum(current_index):

        # 获取当前日期及之前的数据
        filtered_data = resampled_prices.loc[:current_index].copy()

        # 如果数据少于N+1个交易日，返回 NaN
        if len(filtered_data) < N + 1:
            return np.nan

        # 取最后 N+1 个交易日的数据
        selected_data = filtered_data.tail(N + 1)
        if N == 5:
        # 计算长-短因子值
            factor_value = np.log(selected_data.iloc[-1-1] )- np.log(selected_data.iloc[0])
        elif N == 10:
            factor_value = np.log(selected_data.iloc[-1-2] )- np.log(selected_data.iloc[0])
        elif N==20:
            factor_value = np.log(selected_data.iloc[-1-5] )- np.log(selected_data.iloc[0])
        elif N==60:
            factor_value = np.log(selected_data.iloc[-1-10] )- np.log(selected_data.iloc[0])
        elif N == 120:
            factor_value = np.log(selected_data.iloc[-1-20] )- np.log(selected_data.iloc[0])
        elif N == 252:
            factor_value = np.log(selected_data.iloc[-1-60] )- np.log(selected_data.iloc[0])
        else:
            factor_value = np.log(selected_data.iloc[-1] )- np.log(selected_data.iloc[0])
        return factor_value

    def calculate_basic_momentum(current_index):

        # 获取当前日期及之前的数据
        filtered_data = resampled_prices.loc[:current_index].copy()

        # 如果数据少于N+1个交易日，返回 NaN
        if len(filtered_data) < N + 1:
            return np.nan

        # 取最后 N+1 个交易日的数据
        selected_data = filtered_data.tail(N + 1)
        factor_value = np.log(selected_data.iloc[-1]) - np.log(selected_data.iloc[0])

        return factor_value

    # 应用calculate_weighted_momentum函数到filtered_data中
    resampled_data[f'{symbol}_{N}days_long-short_momentum'] = resampled_data.index.to_series().apply(calculate_long_minus_short_momentum)
    resampled_data.reset_index(inplace=True)

    def filter(resampled_data):
        # 进行统计学过滤
        momentum_column = f'{symbol}_{N}days_long-short_momentum'
        absolute_values = resampled_data[momentum_column].abs()

        # 找出最接近 0 的 10% 数据的分位数阈值
        threshold = absolute_values.quantile(0.1)

        # 将绝对值小于等于该阈值的数值设置为 0
        resampled_data.loc[absolute_values <= threshold, momentum_column] = 0
        return resampled_data
    # resampled_data = filter(resampled_data)
    # 选择需要的列
    output_data = resampled_data[['trading_date', f'{symbol}_{N}days_long-short_momentum']]

    # 创建输出文件夹路径
    os.makedirs(output_folder, exist_ok=True)

    # 保存结果
    output_file = f"{symbol}_{N}days_long-short_momentum.csv"
    output_path = os.path.join(output_folder, output_file)
    output_data.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved: {output_path}")


if __name__ == '__main__':

    symbol_list = ["A", "AG", "AL", "AO", "AP", "AU", "B", "BB", "BC", "BR", "BU", "C", "CF",'CJ','CS','CU','CY','EB','EC','EG','ER','FB','FG','FU','HC','I','IC',
                   'IF','IH','IM','J','JD','JM','JR','L','LH','LU','M','MA','ME','NI','NR','OI','P','PB','PF','PG','PK','PM','PP','PX','RB','RI','RM','RO',
                   'RR','RS','RU','SA','SC','SF','SH','SM','SN','SP','SR','SS','T','TA','TC','TF','TL','TS','UR','V','WH','WR','WS','WT','Y','ZC','ZN']


    symbol_list2 = ['IC','IF','IH','IM']
    N_list = [5,10,20,30,40,50,60,70,80,90,100,110,120]
    N_list2 = [5,10,20,60,120,252]
    source_folder = r"\\samba-1.quantchina.pro\quanyi4g\temporary\Steiner\data_wash\linux_so\py311\snapshoot_results_index"
    output_folder = r"\\samba-1.quantchina.pro\quanyi4g\data\future\factor\momentum\long-short_daily_momentum"

    for symbol in symbol_list:
        for N in N_list2:
            basic_daily_momentum(symbol, N, source_folder, output_folder)

