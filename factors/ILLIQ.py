import os
import glob
import pandas as pd
import numpy as np

# 计算因子值和出信号：用品种数据，流动性公式： （N天中每天的收益率/成交金额）的平均值
# 都从index中获取

def calculate_ILLIQ_daynight(symbol: str, N: int, source_folder: str, output_folder:str):
    """
    计算日夜分段的 ILLIQ 因子值

    参数：
    - symbol (str): 交易品种代码。
    - N (int): 回溯天数，用于计算因子。
    - source_folder (str): 源数据文件夹路径。
    - output_folder (str): 输出文件夹路径，用于保存计算结果。

    功能：
    从 `source_folder` 读取特定 `symbol` 的数据文件，通过过滤掉无交易的记录，根据回溯的 N 天数据，
    分别计算日间和夜间的 ILLIQ 因子，并将结果保存为 CSV 文件。

    输出：
    生成以 `{symbol}_{N}days_ILLIQ_daynight.csv` 命名的文件，包含 `trading_date`、`daynight` 和计算得到的 ILLIQ 因子值。

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
    data = data[data['query_notrade'] == 0]
    data['close'] = pd.to_numeric(data['close'], errors='coerce')
    data['prev_close'] = pd.to_numeric(data['prev_close'], errors='coerce')
    data['amount'] = pd.to_numeric(data['amount'], errors='coerce')
    data['return'] = data['close'] / data['prev_close'] - 1
    def calculate_ILLIQ(row):
        current_index = row.name
        current_daynight = row['daynight2']
        # 获取row行及之前的数据
        filtered_data = data.loc[:current_index].copy()

        # 找到最大的N+1个交易日期
        unique_trading_dates = filtered_data['trading_date'].drop_duplicates(keep='last')
        if len(unique_trading_dates) < N + 1:
            return np.nan

        selected_dates = unique_trading_dates.tail(N + 1)
        new_filtered_data = filtered_data[filtered_data['trading_date'].isin(selected_dates)]
        new_filtered_data = new_filtered_data.iloc[:-1]

        new_filtered_data.dropna(subset=['close', 'prev_close', 'amount'], inplace=True)
        new_filtered_data = new_filtered_data[new_filtered_data['amount'] != 0]
        if new_filtered_data.empty:
            return np.nan
        illiq_value = (new_filtered_data['return'] * 1000000 / new_filtered_data['amount']).sum() / N

        return illiq_value


    data[f'{N}days_ILLIQ_daynight'] =data.apply(calculate_ILLIQ, axis=1)
    skew_daynight = data[['trading_date', 'daynight2', f'{N}days_ILLIQ_daynight']]
    skew_daynight.rename(columns={'daynight2':'daynight'},inplace=True)

    output_file = f"{symbol}_{N}days_ILLIQ_daynight.csv"
    output_path = os.path.join(output_folder, output_file)
    skew_daynight.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved: {output_path}")
# daily:
def calculate_ILLIQ_daily(symbol: str, N: int, source_folder: str, output_folder: str):
    """
    计算日频率的 ILLIQ 因子值

    参数：
    - symbol (str): 交易品种代码。
    - N (int): 回溯天数，用于计算因子。
    - source_folder (str): 源数据文件夹路径。
    - output_folder (str): 输出文件夹路径，用于保存计算结果。

    功能：
    从 `source_folder` 读取特定 `symbol` 的数据文件，通过过滤掉无交易的记录，根据回溯的 N 天数据，
    计算每日 ILLIQ 因子，并将结果保存为 CSV 文件。

    输出：
    生成以 `{symbol}_{N}days_ILLIQ_daily.csv` 命名的文件，包含 `trading_date` 和计算得到的 ILLIQ 因子值。

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
    resampled_data ['close'] = pd.to_numeric(resampled_data ['close'], errors='coerce')
    resampled_data ['prev_close'] = pd.to_numeric(resampled_data ['prev_close'], errors='coerce')
    resampled_data ['amount'] = pd.to_numeric(resampled_data ['amount'], errors='coerce')
    resampled_data['return'] = resampled_data['close'] / resampled_data['prev_close'] - 1
    def calculate_ILLIQ(current_index):

        # 获取当前日期及之前的数据
        filtered_data = resampled_data.loc[:current_index].copy()

        # 如果数据少于N+1个交易日，返回 NaN
        if len(filtered_data) < N + 1:
            return np.nan

        # 取最后 N+1 个交易日的数据
        selected_data = filtered_data.tail(N)

        selected_data.dropna(subset=['close', 'prev_close', 'amount'], inplace=True)
        new_filtered_data = selected_data[selected_data['amount'] != 0]
        if new_filtered_data.empty:
            return np.nan

        # 计算因子值：最后一个 'last_prc' 除以第一个 'last_prc'
        illiq_value = (selected_data['return'] * 1000000 / selected_data['amount']).sum() / N

        return illiq_value



    # 应用calculate_weighted_momentum函数到filtered_data中
    resampled_data[f'{symbol}_{N}days_ILLIQ_daily'] = resampled_data.index.to_series().apply(calculate_ILLIQ)
    resampled_data.reset_index(inplace=True)

    # 选择需要的列
    output_data = resampled_data[['trading_date', f'{symbol}_{N}days_ILLIQ_daily']]

    # 创建输出文件夹路径
    os.makedirs(output_folder, exist_ok=True)

    # 保存结果
    output_file = f'{symbol}_{N}days_ILLIQ_daily'
    output_path = os.path.join(output_folder, output_file)
    output_data.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved: {output_path}")





symbol_list = ["A", "AG", "AL", "AO", "AP", "AU", "B", "BB", "BC", "BR", "BU", "C", "CF",'CJ','CS','CU','CY','EB','EC','EG','ER','FB','FG','FU','HC','I','IC',
               'IF','IH','IM','J','JD','JM','JR','L','LH','LU','M','MA','ME','NI','NR','OI','P','PB','PF','PG','PK','PM','PP','PX','RB','RI','RM','RO',
               'RR','RS','RU','SA','SC','SF','SH','SM','SN','SP','SR','SS','T','TA','TC','TF','TL','TS','UR','V','WH','WR','WS','WT','Y','ZC','ZN']

N_list = [5,10,22,66,252]

source_folder = "/nas92/data/future/trade_buffer/buffer_day_A/"
output_folder = "/nas92/data/future/factor/ILLIQ/ILLIQ_daynight/"
output_folder2 = "/nas92/data/future/factor/ILLIQ/ILLIQ_daily/"
for symbol in symbol_list:
    for N in N_list:
        calculate_ILLIQ_daynight(symbol, N, source_folder, output_folder)
        calculate_ILLIQ_daily(symbol,N, source_folder, output_folder2)
