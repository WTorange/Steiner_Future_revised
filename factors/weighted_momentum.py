import os
import glob
import pandas as pd
import numpy as np

def weighted_momentum(symbol: str, N: int, source_folder: str, output_folder: str):
    # ????????????

    # ?????? symbol ??????????
    file_list = [f for f in os.listdir(source_folder) if f.startswith(symbol + "_") and f.endswith('.csv')]

    if not file_list:
        print(f"No files found for {symbol}")
        return

    # ????????????????????
    file_path = os.path.join(source_folder, file_list[0])
    data = pd.read_csv(file_path)

    # ???? symbol ?????????????????????????? symbol
    if data['symbol'].iloc[0] != symbol:
        print(f"Symbol mismatch in file {file_list[0]}")
        return
    # ???? daynight2 ??
    data['trading_date'] = pd.to_datetime(data['trading_date'], format='%Y-%m-%d')
    data['query_time'] = pd.to_datetime(data['query_time'])
    data['daynight2'] = np.where(
        data['query_time'].dt.time < pd.Timestamp('08:00:00').time(),
        'night',
        np.where(data['query_time'].dt.time < pd.Timestamp('20:00:00').time(), 'day', 'night')
    )


    filtered_data = data[data['query_notrade'] == 0]

    def calculate_weighted_momentum(row):
        current_index = row.name
        current_daynight = row['daynight2']

        filtered_data = data.loc[:current_index].copy()

        filtered_data = filtered_data[filtered_data['daynight2'] == current_daynight]

        unique_trading_dates = filtered_data['trading_date'].drop_duplicates(keep='last')
        if len(unique_trading_dates) < N + 1:
            return np.nan

        selected_dates = unique_trading_dates.tail(N + 1)
        new_filtered_data = filtered_data[filtered_data['trading_date'].isin(selected_dates)]

        if new_filtered_data.empty:
            return np.nan

        resampled_prices = new_filtered_data.groupby('trading_date')['last_prc'].last()


        returns = resampled_prices.pct_change().dropna()

        if len(returns) < N:
            return np.nan


        days_diff = np.arange(len(returns), 0, -1)
        half_life = N / 2.0
        weights = np.exp(-np.log(2) * days_diff / half_life)

        weights /= weights.sum()


        weighted_momentum = np.sum(returns * weights)

        return weighted_momentum


    filtered_data[f'{symbol}_{N}days_weighted_momentum'] = filtered_data.apply(calculate_weighted_momentum, axis=1)

    # ??????????
    weighted_mom = filtered_data[['trading_date', 'daynight2', f'{symbol}_{N}days_weighted_momentum']]
    weighted_mom.rename(columns={'daynight2': 'daynight'}, inplace=True)

    os.makedirs(output_folder, exist_ok=True)

    # ????????
    output_file = f"{symbol}_{N}days_weighted_momentum.csv"
    output_path = os.path.join(output_folder, output_file)
    weighted_mom.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved: {output_path}")



def weighted_daily_momentum(symbol: str, N: int, source_folder: str, output_folder: str):


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
    def filtered_weight(N):
        if N == 5:
            days_to_remove = 1
        elif N ==10:
            days_to_remove = 2
        elif N ==20:
            days_to_remove = 5
        elif N ==60:
            days_to_remove = 10
        elif N ==120:
            days_to_remove = 20
        elif N== 252:
            days_to_remove = 60
        else:
            days_to_remove = min(N // 5, N - 1)

        adjusted_N = N - days_to_remove
        # 提前计算权重

        days_diff = np.arange(adjusted_N+days_to_remove, days_to_remove, -1)
        half_life = adjusted_N / 2.0
        weights = np.exp(-np.log(2) * days_diff / half_life)
        weights /= weights.sum()  # 归一化，使权重和为1

        def apply_weighted_momentum(x):
            # 过滤掉 NaN 值
            x = x[~np.isnan(x)]
            if len(x) == N:
                return np.dot(x[-adjusted_N:], weights)
            else:
                return np.nan

        rolling_returns = resampled_prices.pct_change().rolling(window=N).apply(apply_weighted_momentum, raw=True)

        return rolling_returns

    def weight(N):
        adjusted_N =N
        days_diff = np.arange(N, 0, -1)
        half_life = adjusted_N / 2.0
        weights = np.exp(-np.log(2) * days_diff / half_life)
        weights /= weights.sum()  # 归一化，使权重和为1
        def apply_weighted_momentum(x):
            # 过滤掉 NaN 值
            x = x[~np.isnan(x)]
            if len(x) == N:
                return np.dot(x[-adjusted_N:], weights)
            else:
                return np.nan

        rolling_returns = resampled_prices.pct_change().rolling(window=N).apply(apply_weighted_momentum, raw=True)


        return rolling_returns

    # rolling_returns = filtered_weight(N)
    rolling_returns = weight(N)
    momentum_df = pd.DataFrame({
        'trading_date': resampled_prices.index,
        f'{symbol}_{N}days_weighted_momentum': rolling_returns
    }).dropna()
    # 应用calculate_weighted_momentum函数到filtered_data中
    # resampled_data[f'{symbol}_{N}days_weighted_momentum'] = returns
    # resampled_data.reset_index(inplace=True)
    #
    # # 选择需要的列
    # output_data = resampled_data[['trading_date', f'{symbol}_{N}days_weighted_momentum']]

    # 创建输出文件夹路径
    os.makedirs(output_folder, exist_ok=True)

    # 保存结果
    output_file = f"{symbol}_{N}days_weighted_momentum.csv"
    output_path = os.path.join(output_folder, output_file)
    momentum_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved: {output_path}")


symbol_list = ["A", "AG", "AL", "AO", "AP", "AU", "B", "BB", "BC", "BR", "BU", "C", "CF", 'CJ', 'CS', 'CU', 'CY', 'EB',
               'EC', 'EG', 'ER', 'FB', 'FG', 'FU', 'HC', 'I', 'IC',
               'IF', 'IH', 'IM', 'J', 'JD', 'JM', 'JR', 'L', 'LH', 'LU', 'M', 'MA', 'ME', 'NI', 'NR', 'OI', 'P', 'PB',
               'PF', 'PG', 'PK', 'PM', 'PP', 'PX', 'RB', 'RI', 'RM', 'RO',
               'RR', 'RS', 'RU', 'SA', 'SC', 'SF', 'SH', 'SM', 'SN', 'SP', 'SR', 'SS', 'T', 'TA', 'TC', 'TF', 'TL',
               'TS', 'UR', 'V', 'WH', 'WR', 'WS', 'WT', 'Y', 'ZC', 'ZN']
symbol_list2 =['IC',
               'IF', 'IH', 'IM']
N_list = [5,10,20,30,40,50,60,70,80,90,100,110,120,252]
N_list2 = [5,10,20,60,120,252]
source_folder = "/nas92/temporary/Steiner/data_wash/linux_so/py311/snapshot_results_index/"
output_folder = "/nas92/data/future/factor/momentum/weighted_daily_momentum"

for symbol in symbol_list:
    for N in N_list2:
        weighted_daily_momentum(symbol, N, source_folder, output_folder)