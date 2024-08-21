import os
import glob
import pandas as pd
import numpy as np

# 先得到最大可交易合约
# 特殊情况：wap价格为0——涨跌停，以及通过开盘价和last_prc判断是否涨跌停，添加可交易标签.涨停不能买入，跌停不能卖出。
def determine_trade_contracts(df):
    # 确定每天交易的合约代码
    df['trade_contract'] = ''
    df['change_contract'] = ''

    # 初始交易合约为第一天的主力合约
    current_contract = df.at[0, 'main_contract']
    df.at[0, 'trade_contract'] = current_contract

    for i in range(1, len(df) - 1):
        if df.at[i, 'main_contract'] != current_contract and df.at[
            i - 1, 'main_contract'] != current_contract and df.at[i - 2, 'main_contract'] != current_contract\
                and df.at[i - 3, 'main_contract'] == current_contract:

            df.at[i, 'trade_contract'] = current_contract
            df.at[i, 'change_contract'] = df.at[i, 'main_contract']
            # 接下来的一天也在换仓期
            current_contract = df.at[i, 'main_contract']
        elif df.at[i, 'trade_contract'] == '':
            # 如果不是换仓期，交易前一天的合约
            df.at[i, 'trade_contract'] = current_contract
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.strftime('%Y%m%d')


    return df[['date', 'trade_contract', 'change_contract']]
# 读取parquet文件的函数
def read_parquet_files(file_path):
    all_files = glob.glob(os.path.join(file_path, "*.parquet"))
    df_list = [pd.read_parquet(file) for file in all_files]
    data = pd.concat(df_list, ignore_index=True)
    return data


# 划分日盘和夜盘并编号的函数
def split_day_night(data):
    data['resample_time'] = pd.to_datetime(data['resample_time'])
    # data['tradedate'] = data['resample_time'].dt.date
    data['daynight'] = np.where(
        data['resample_time'].dt.time < pd.Timestamp('08:00:00').time(),
        'night',
        np.where(data['resample_time'].dt.time < pd.Timestamp('20:00:00').time(), 'day', 'night')
    )

    # 给每个日盘和夜盘编号
    data['period'] = data.groupby(['trading_date', 'daynight']).ngroup()
    print('split_done')
    return data


# 生成布林带策略信号的函数
def generate_bollinger_signals(data, N, k):
    # 计算中枢（移动平均线）和标准差
    period_groups = data.groupby('period').first().reset_index()  # 取每个盘口的第一个记录
    period_groups['mean'] = period_groups['last_prc'].rolling(window=N).mean()
    period_groups['std'] = period_groups['last_prc'].rolling(window=N).std()

    # 计算布林带的上下轨
    period_groups['upper'] = period_groups['mean'] + k * period_groups['std']
    period_groups['lower'] = period_groups['mean'] - k * period_groups['std']

    # 初始化trade_df
    trade_df = pd.DataFrame(columns=['date', 'daynight',  'position'])

    # 初始持仓为0
    position = 0

    # 遍历每一个盘口
    for i in range(1, len(period_groups)):
        current_row = period_groups.iloc[i]
        previous_row = period_groups.iloc[i - 1]

        # 判断开仓条件
        if position == 0:
            if previous_row['last_prc'] <= previous_row['upper'] and current_row['last_prc'] > current_row['upper']:
                position = 1  # 满仓做多
            elif previous_row['last_prc'] >= previous_row['lower'] and current_row['last_prc'] < current_row['lower']:
                position = -1  # 满仓做空

        # 判断平仓条件
        if position == 1:
            if current_row['last_prc'] > current_row['mean'] + (k + 1) * current_row['std'] or current_row['last_prc'] < \
                    current_row['mean']:
                position = 0  # 平仓
        elif position == -1:
            if current_row['last_prc'] < current_row['mean'] - (k + 1) * current_row['std'] or current_row['last_prc'] > \
                    current_row['mean']:
                position = 0  # 平仓

        # 记录交易信号
        trade_df = pd.concat([trade_df, pd.DataFrame([{
            'date': current_row['trading_date'],
            'daynight': current_row['daynight'],
            'position': position
        }])], ignore_index=True)

    trade_df['daynight_order'] = trade_df['daynight'].apply(lambda x: 0 if x == 'night' else 1)
    trade_df = trade_df.sort_values(by=['date', 'daynight_order']).drop(columns=['daynight_order'])
    if 'Unnamed: 0' in trade_df.columns:
        trade_df.drop(columns=['Unnamed: 0'], inplace=True)

   # trade_df已经排序好
    return trade_df


# 使用示例
# file_path = 'Z:/data/future/BR'
# data = read_parquet_files(file_path)
# data = split_day_night(data)
# N = 10  # 设定N
# k = 0.3  # 设定k
# trade_df = generate_bollinger_signals(data, N, k)
#
# trade_df.to_csv('001.csv')