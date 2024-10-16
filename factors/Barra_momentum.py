import os
import glob
import pandas as pd
import numpy as np

def weighted_momentum(symbol: str, source_folder: str, output_folder: str, N = 504):


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

    def calculate_weighted_momentum(row):
        current_index = row.name
        current_daynight = row['daynight2']

        # 获取row行及之前的数据
        filtered_data = data.loc[:current_index].copy()

        # 筛选daynight2相同的数据
        filtered_data = filtered_data[filtered_data['daynight2'] == current_daynight]

        # 找到最大的N+1个交易日期
        unique_trading_dates = filtered_data['trading_date'].drop_duplicates(keep='last')
        if len(unique_trading_dates) < N + 1 + 21:
            return np.nan

        selected_dates = unique_trading_dates[-(N + 1 + 21):-21]
        new_filtered_data = filtered_data[filtered_data['trading_date'].isin(selected_dates)]

        if new_filtered_data.empty:
            return np.nan

        # 重采样每个 trading_date，只保留最后一个 last_prc
        resampled_prices = new_filtered_data.groupby('trading_date')['last_prc'].last()

        # 计算收益率
        returns = resampled_prices.pct_change().dropna()

        # 如果不足N个交易日，返回NaN
        if len(returns) < N:
            return np.nan

        # 计算权重
        days_diff = np.arange(len(returns), 0, -1)
        half_life = 126
        weights = np.exp(-np.log(2) * days_diff / half_life)

        weights /= weights.sum()  # 归一化，使权重和为1

        # 计算加权动量因子
        weighted_momentum = np.sum(returns * weights)

        return weighted_momentum

    # 应用calculate_weighted_momentum函数到filtered_data中
    filtered_data[f'{symbol}_{N}days_barra_momentum'] = filtered_data.apply(calculate_weighted_momentum, axis=1)

    # 提取相关列
    weighted_mom = filtered_data[['trading_date', 'daynight2', f'{symbol}_{N}days_barra_momentum']]
    weighted_mom.rename(columns={'daynight2': 'daynight'}, inplace=True)

    # 创建输出文件夹路径
    os.makedirs(output_folder, exist_ok=True)

    # 保存结果
    output_file = f"{symbol}_{N}days_barra_momentum.csv"
    output_path = os.path.join(output_folder, output_file)
    weighted_mom.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved: {output_path}")


symbol_list = ["A", "AG", "AL", "AO", "AP", "AU", "B", "BB", "BC", "BR", "BU", "C", "CF", 'CJ', 'CS', 'CU', 'CY', 'EB',
               'EC', 'EG', 'ER', 'FB', 'FG', 'FU', 'HC', 'I', 'IC',
               'IF', 'IH', 'IM', 'J', 'JD', 'JM', 'JR', 'L', 'LH', 'LU', 'M', 'MA', 'ME', 'NI', 'NR', 'OI', 'P', 'PB',
               'PF', 'PG', 'PK', 'PM', 'PP', 'PX', 'RB', 'RI', 'RM', 'RO',
               'RR', 'RS', 'RU', 'SA', 'SC', 'SF', 'SH', 'SM', 'SN', 'SP', 'SR', 'SS', 'T', 'TA', 'TC', 'TF', 'TL',
               'TS', 'UR', 'V', 'WH', 'WR', 'WS', 'WT', 'Y', 'ZC', 'ZN']

N_list = [504]

source_folder = "/nas92/temporary/Steiner/data_wash/linux_so/py311/snapshot_results_index"
output_folder = "/nas92/data/future/factor/momentum/barra_momentum"

for symbol in symbol_list:
    weighted_momentum(symbol, source_folder, output_folder, N=504)