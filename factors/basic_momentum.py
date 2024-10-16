import os
import glob
import pandas as pd
import numpy as np

# 不用quote数据，用处理好的snapshot和twap_vwap数据
# 但是一开始可以先用quote数据（用前收算
# 先回退N个trading_date，然后看日夜盘


# 计算因子值：品种wap文件中，往前N个trading_date， 然后看query_time, 输出因子值。
# 然后根据因子值调整信号，这样更为方便
r"Z:\temporary\Steiner\data_wash\linux_so\py311\snapshot_results_index"
# momentum类，里面是动量因子的函数，
# 参数：symbol, N
def calculate_basic_momentum(symbol: str, N: int, source_folder: str, output_folder:str):
    # 设置文件路径

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

        # print(filtered_days)

        return last_price_A / last_price_B - 1


    filtered_data[f'{symbol}_{N}days_basic_momentum'] = filtered_data.apply(calculate_momentum, axis=1)
    basic_mom = filtered_data[['trading_date', 'daynight2', f'{symbol}_{N}days_basic_momentum']]
    basic_mom.rename(columns={'daynight2':'daynight'},inplace=True)

    output_file = f"{symbol}_{N}days_basic_momentum.csv"
    output_path = os.path.join(output_folder, output_file)
    basic_mom.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved: {output_path}")


symbol_list = ["A", "AG", "AL", "AO", "AP", "AU", "B", "BB", "BC", "BR", "BU", "C", "CF",'CJ','CS','CU','CY','EB','EC','EG','ER','FB','FG','FU','HC','I','IC',
               'IF','IH','IM','J','JD','JM','JR','L','LH','LU','M','MA','ME','NI','NR','OI','P','PB','PF','PG','PK','PM','PP','PX','RB','RI','RM','RO',
               'RR','RS','RU','SA','SC','SF','SH','SM','SN','SP','SR','SS','T','TA','TC','TF','TL','TS','UR','V','WH','WR','WS','WT','Y','ZC','ZN']
# inplace_symbol_list = ['A', 'C', 'IF', 'NR', 'RR', 'T']出于后续修改数据的问题，需要重新跑一下
N_list = [5,10,22,66,252]

source_folder = "/nas92/temporary/Steiner/data_wash/linux_so/py311/snapshot_results_index/"
output_folder = "/nas92/data/future/factor/momentum/basic_momentum"

for symbol in symbol_list:
    for N in N_list:
        calculate_basic_momentum(symbol, N, source_folder, output_folder)

