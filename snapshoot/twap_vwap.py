import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

def wap(symbol, start_date: int, end_date: int, start_time: int, end_time: int):
    """
    计算指定时间段内的时间加权平均价格（TWAP）和成交量加权平均价格（VWAP）。

    :param symbol: 合约标识符
    :param start_date: 开始日期（格式：YYYYMMDD）
    :param end_date: 结束日期（格式：YYYYMMDD）
    :param start_time: 开始时间（格式：HHMMSSSSS）
    :param end_time: 结束时间（格式：HHMMSSSSS）

    :return: 包含 TWAP 和 VWAP 的 DataFrame
    """
    result = pd.DataFrame(columns=['symbol', 'date', 'start_time', 'end_time', 'twap', 'vwap'])

    current_date = start_date
    # 循环处理每一天的数据，可以优化
    while current_date <= end_date:
        if not os.path.exists(sub_path):
            print(f'no data in dir {sub_path}')
            current_date += 1
            continue

        if os.path.exists(sub_path + file1):
            df = pd.read_parquet(sub_path + file1)
        elif os.path.exists(sub_path + file2):  # 分成两种文件名的情况
            df = pd.read_parquet(sub_path + file2)
        else:
            print(f'no data for {symbol} on {current_date}')
            current_date += 1
            continue

        # 处理跨午夜的情况（即如果时间范围跨越 24:00:00）
        if (start_time <= 240000000) & (end_time <= 23000000):
            mask1 = (df['time'] >= start_time) & (df['time'] <= 240000000)
            mask2 = (df['time'] >= 0) & (df['time'] < end_time)
            df.loc[mask1, 'time'] = df.loc[mask1, 'time'] - 120000000
            df.loc[mask2, 'time'] = df.loc[mask2, 'time'] + 120000000

            # 这里开始的一段和else之后的一段是完全一样的？写错了？需要检查修改。
        df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
        twap = df['last_prc'].mean()
        vwap = np.dot(df['last_prc'], df['t_volume']) / df['t_volume'].sum()

            # 创建临时行并且添加结果
        temp_row = pd.Series({'symbol': symbol, 'date': current_date, 'start_time': start_time,
                              'end_time': end_time, 'twap': twap, 'vwap': vwap})

        result = result.append(temp_row, ignore_index=True)

        current_date += 1

    return result

if __name__ == '__main__':
    symbol = 'CU2302.SHF'
    start_date = 20230101
    end_date = 20230106
    start_time = 92000000
    end_time = 100000000

    df_wap = wap(symbol, start_date, end_date, start_time, end_time)
    print(df_wap)