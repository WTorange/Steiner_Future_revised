
import os
import pandas as pd
pd.options.mode.chained_assignment = None
from glob import glob
import numpy as np
from datetime import datetime
import re
from datetime import datetime, timedelta

'''
从组内数据库的清洗过的quote数据中直接生成每天主力合约和次主力合约的快照&wap价格数据。便于选择读取quote数据文件夹范围进行更新。
兼容了公司quote数据涨跌停时刻累计成交量异常的情况，修正了wap数据没有trading_date列的问题。获取快照以及wap数据建议使用此程序，而不是直接从公司数据库读取的程序。
输入：quote数据文件夹，想要市场的期货品种代码，快照和wap输出文件夹
输出： 品种的快照和wap数据
'''




def get_quote(symbol, base_path="/nas92/data/future/index"):
    # 获取指定文件夹下所有符合条件的文件路径
    folder_path = os.path.join(base_path, symbol)
    file_pattern = f"{folder_path}/{symbol}_*_weightedaverage.parquet"

    files = glob(file_pattern)

    # 提取日期并排序
    files_with_dates = []
    for file in files:
        # 提取文件名中的日期部分
        date_str = os.path.basename(file).split('_')[1]
        date = datetime.strptime(date_str, "%Y%m%d")
        files_with_dates.append((date, file))

    # 按日期排序文件
    files_with_dates.sort()

    # 获取排序后的文件路径
    sorted_files = [file for date, file in files_with_dates]
    if len(sorted_files) <= 2:
        print(f"Insufficient files to process for symbol: {symbol}. Required at least 2 files, but got {len(sorted_files)}.")
        return
    # 遍历文件并两两合并
    for i in range(len(sorted_files) - 2):
        file1 = sorted_files[i]
        file2 = sorted_files[i + 1]
        file3 = sorted_files[i + 2]
        # 读取三个相邻的文件
        df1 = pd.read_parquet(file1)
        df2 = pd.read_parquet(file2)
        df3 = pd.read_parquet(file3)

        # 合并两个文件的数据
        combined_df = pd.concat([df1, df2, df3])

        yield combined_df


def wap(day_quote, symbol,start_datetime_str, *lengths):

    day_quote['resample_time'] = pd.to_datetime(day_quote['resample_time'])
    day_quote['time'] = day_quote['resample_time'].dt.strftime('%H%M%S%f').str.slice(0, 9)
    day_quote['date'] = day_quote['resample_time'].dt.strftime('%Y%m%d')
    result = pd.DataFrame(columns=[ 'date', 'start_time','trading_date'])

    start_datetime = pd.to_datetime(start_datetime_str, format='%Y-%m-%d %H:%M:%S')
    date = start_datetime.strftime('%Y%m%d')


    for length in lengths:
        # 解析长度
        if length.startswith('-'):
            is_backward = True
            length = length[1:]
        else:
            is_backward = False

        hours, minutes, seconds, milliseconds = map(int, length.split(':'))
        length_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)

        length_desc_parts = []
        if hours > 0:
            length_desc_parts.append(f"{hours}h")
        if minutes > 0 or hours > 0:
            length_desc_parts.append(f"{minutes}m")
        if seconds > 0 and ( minutes > 0 or hours > 0):
            length_desc_parts.append(f"{seconds}s")
        if milliseconds > 0 and ( seconds > 0 or minutes > 0 or hours > 0):
            length_desc_parts.append(f"{milliseconds}f")
        length_desc = ''.join(length_desc_parts)


        if is_backward:
            end_datetime = start_datetime - length_delta
        else:
            end_datetime = start_datetime + length_delta

        maskdate = day_quote['date'] == date

        masktime = (day_quote['resample_time'] >= min(start_datetime,end_datetime)) & (day_quote['resample_time'] <= max(start_datetime,end_datetime))
        df_period = day_quote[maskdate & masktime]

        if df_period.empty:
            print(f"无数据， date={date}, start_time={start_datetime}")
            return None

        if not df_period.empty:
            # 读取初始价格 init
            init = df_period['last_prc'].iloc[0]
            contract = df_period['symbol'].iloc[0]
            trading_date = df_period['trading_date'].iloc[0]
            # 过滤掉 middle_price, ask_price, bid_price 为 0 的行
            df_period = df_period[
                (df_period['middle_price'] != 0) & (df_period['ask_prc1'] != 0) & (df_period['bid_prc1'] != 0)]

            # 检查过滤后是否为空
            if df_period.empty:
                # 如果过滤后为空，设置 twap 和 vwap 为 init
                twap = vwap = init
                no_volume = 1
            else:
                # 计算 VWAP 时需要考虑成交量的差值
                df_period['volume_diff'] = df_period['volume'].diff().fillna(df_period['volume'])

                # 计算 TWAP
                twap = df_period['middle_price'].mean()

                # 计算 VWAP
                if df_period['volume_diff'].sum() != 0 and not np.isnan(df_period['volume_diff'].sum()):
                    vwap = np.dot(df_period['middle_price'], df_period['volume_diff']) / df_period['volume_diff'].sum()
                    no_volume = 0
                else:
                    vwap = df_period['middle_price'].iloc[0]
                    no_volume = 1

            # 继续处理或返回计算结果

            result.at[0, 'date'] = date
            result.at[0,'symbol'] = symbol
            result.at[0,'contract'] = contract

            result.at[0, 'start_time'] = str(start_datetime)
            result.at[0, 'datetime'] = start_datetime
            result.at[0, 'trading_date'] = trading_date
            result[f'{length_desc}_twap_{"pre_" if is_backward else "post_"}prc'] = twap
            result[f'{length_desc}_vwap_{"pre_" if is_backward else "post_"}prc'] = vwap
            result.at[0, 'no_volume'] = no_volume

    return result

def certain_snapshot( df, time, symbol):

    df['resample_time'] = pd.to_datetime(df['resample_time'])
    df['time'] = df['resample_time'].dt.strftime('%H%M%S%f').str.slice(0, 9)
    df['date'] = df['resample_time'].dt.strftime('%Y%m%d')

    query_time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')

    df_before_query = df[df['resample_time'] <= query_time]
    query_date = query_time.date()
    if df_before_query.empty:
        return None
    else:
        last_tick = df_before_query.iloc[-1]
        # 增加对夜盘缺失的判断
        last_tick_time = last_tick['resample_time'].time()

        if (query_time.time() >= datetime.strptime("21:05:00", '%H:%M:%S').time() and
                last_tick_time < datetime.strptime("15:05:00", '%H:%M:%S').time()):
            null_value = 1
        elif last_tick['resample_time'].date() != query_date:
            null_value = 1
        else:
            null_value = 0

    # 当天数据不存在，按要求输出数据
    if null_value == 1:
        high = low = open_price = prev_close = last_tick['last_prc']
        ask = bid = volume = amount = 0

    def is_day_time(time_str):
        time_obj = datetime.strptime(time_str.split('.')[0], '%H%M%S%f')
        return datetime.strptime("080000000000", '%H%M%S%f') <= time_obj <= datetime.strptime("200000000000",
                                                                                              '%H%M%S%f')
    last_tick = last_tick.copy()
    last_tick_time = last_tick['time']
    last_tick['daynight'] = 'day' if is_day_time(last_tick_time) else 'night'

    # 分日盘和夜盘讨论
    if null_value == 0:
        if last_tick['daynight'] == "night":
            # 由于有的夜盘会跨自然日，确定夜盘开始的日期时间
            if last_tick['resample_time'].time() >= datetime.strptime("200000", '%H%M%S').time():
                night_start_time = last_tick['resample_time'].replace(hour=20, minute=0, second=0, microsecond=0)
            else:
                night_start_time = (last_tick['resample_time'] - timedelta(days=1)).replace(hour=20, minute=0, second=0,
                                                                                       microsecond=0)
            # 根据last_prc计算高开低收, amount等
            high = df[(df['resample_time'] >= night_start_time) & (df['resample_time'] <= query_time)]['last_prc'].max()

            low = df[(df['resample_time'] >= night_start_time) & (df['resample_time'] <= query_time)]['last_prc'].min()

            ask = last_tick['ask_prc1']
            bid = last_tick['bid_prc1']
            volume = last_tick['volume']
            amount = last_tick['turnover']
            night_df = df[df['resample_time'] >= night_start_time]
            open_price = night_df.iloc[0]['last_prc'] if not night_df.empty else None

            # 根据last_tick的时间判断之前一个盘的时间区间（日盘or夜盘）
            if last_tick['resample_time'].time() >= datetime.strptime("200000", '%H%M%S').time():
                prev_close_time_start = last_tick['resample_time'].replace(hour=8, minute=0, second=0, microsecond=0)
                prev_close_time_end = last_tick['resample_time'].replace(hour=20, minute=0, second=0, microsecond=0)
            else:
                prev_close_time_start = (last_tick['resample_time'] - timedelta(days=1)).replace(hour=8, minute=0,
                                                                                            second=0,
                                                                                            microsecond=0)
                prev_close_time_end = (last_tick['resample_time'] - timedelta(days=1)).replace(hour=20, minute=0,
                                                                                          second=0,
                                                                                          microsecond=0)
            retry_count = 0
            while True:
                prev_close_df = df[
                    (df['resample_time'] >= prev_close_time_start) & (df['resample_time'] < prev_close_time_end)]
                if not prev_close_df.empty:
                    prev_close = prev_close_df.iloc[-1]['last_prc']
                    break
                # 如果上一个时间段没有数据，则往前推12小时
                prev_close_time_start -= timedelta(hours=12)
                prev_close_time_end -= timedelta(hours=12)
                retry_count += 1

                # 如果计数器达到20，跳出循环并设置 prev_close 为 None 或其他空值
                if retry_count >=30:
                    prev_close = None
                    break

        # 日盘的情况与夜盘相同
        elif last_tick['daynight'] == "day":
            day_start_time = last_tick['resample_time'].replace(hour=8, minute=0, second=0, microsecond=0)

            high = df[(df['resample_time'] >= day_start_time) & (df['resample_time'] <= query_time)]['last_prc'].max()
            low = df[(df['resample_time'] >= day_start_time) & (df['resample_time'] <= query_time)]['last_prc'].min()

            ask = last_tick['ask_prc1']
            bid = last_tick['bid_prc1']

            day_df = df[df['resample_time'] >= day_start_time]
            open_price = day_df.iloc[0]['last_prc'] if not day_df.empty else None

            # 计算amount
            last_volume = last_tick['volume']
            first_volume = day_df.iloc[0]['volume'] if not day_df.empty else 0
            last_amount = last_tick['turnover']
            first_amount = day_df.iloc[0]['turnover'] if not day_df.empty else 0

            volume = last_volume - first_volume if not day_df.empty else 0
            amount = last_amount - first_amount if not day_df.empty else 0
            prev_close_time_start = (last_tick['resample_time'] - timedelta(days=1)).replace(hour=20, minute=0, second=0,
                                                                                        microsecond=0)
            prev_close_time_end = last_tick['resample_time'].replace(hour=8, minute=0, second=0, microsecond=0)

            retry_count = 0
            while True:
                prev_close_df = df[
                    (df['resample_time'] >= prev_close_time_start) & (df['resample_time'] < prev_close_time_end)]
                if not prev_close_df.empty:
                    prev_close = prev_close_df.iloc[-1]['last_prc']
                    break
                retry_count += 1
                prev_close_time_start -= timedelta(hours=12)
                prev_close_time_end -= timedelta(hours=12)
                if retry_count >= 30:
                    prev_close = None
                    break

    result = {
        'symbol': symbol,
        'date': last_tick['date'],
        'trading_date': last_tick['trading_date'],
        'daynight': last_tick['daynight'],
        'time': last_tick['time'],
        'open': open_price,
        'high': high,
        'low': low,
        'last_prc': last_tick['last_prc'],
        'prev_close': prev_close,
        'open_interest': last_tick['open_interest'],
        'volume': volume,
        'amount': amount,
        'ask': ask,
        'bid': bid,
        'query_time': str(time),
        'query_notrade': null_value
    }

    result_df = pd.DataFrame([result])

    return result_df


def process_quotes(symbol, snapshot_results_folder, wap_results_folder):
    wap_results = []
    snapshot_results = []
    query_counter = 0

    for combined_df in get_quote(symbol):

        # 获取 trading_date 列中第二大的日期
        unique_dates = pd.to_datetime(combined_df['trading_date']).dt.date.unique()
        if len(unique_dates) < 2:
            continue  # 如果唯一日期少于2，跳过此组合
        dates = sorted(unique_dates)[-2]
        date_str = dates.strftime('%Y-%m-%d')

        # 生成查询的日期时间列表
        date_time_list = [f"{date_str} 14:30:00", f"{date_str} 22:30:00"]

        # 对每个日期时间进行查询
        for query_time in date_time_list:
            print(symbol,query_time)
            quote_data = combined_df  # 假设 combined_df 包含所需的 quote_data
            if quote_data is None:
                continue

            lengths = ['-00:01:00:000', '-00:03:00:000', '-00:05:00:000', '00:01:00:000', '00:03:00:000',
                       '00:05:00:000']
            snapshot_result = certain_snapshot(quote_data, query_time, symbol=symbol)
            # wap_result = wap(quote_data, datetime.strptime(query_time, '%Y-%m-%d %H:%M:%S'), *lengths)

            wap_result= wap(quote_data, symbol,datetime.strptime(query_time, '%Y-%m-%d %H:%M:%S'), *lengths)
            wap_results.append(wap_result)
            snapshot_results.append(snapshot_result)
            query_counter += 1

            # 每处理20次输出一次结果
            if query_counter % 5 == 0:
                if snapshot_results:
                    combined_snapshot_results = pd.concat(snapshot_results, ignore_index=True)
                    sorted_snapshot_results = combined_snapshot_results.sort_values(by=['query_time']).reset_index(
                        drop=True)

                    snapshot_output_file = os.path.join(snapshot_results_folder, f"{symbol}_snapshot_results.csv")
                    sorted_snapshot_results.to_csv(snapshot_output_file, index=False)


                if wap_results:
                    combined_wap_results = pd.concat(wap_results, ignore_index=True)
                    sorted_wap_results = combined_wap_results.sort_values(by=(['start_time'])).reset_index(drop=True)
                    wap_output_file = os.path.join(wap_results_folder, f"{symbol}_wap_results.csv")
                    sorted_wap_results.to_csv(wap_output_file, index=False)


    # 处理剩余的结果
    if snapshot_results:
        combined_snapshot_results = pd.concat(snapshot_results, ignore_index=True)
        sorted_snapshot_results = combined_snapshot_results.sort_values(by=['query_time']).reset_index(drop=True)
        snapshot_output_file = os.path.join(snapshot_results_folder, f"{symbol}_snapshot_results.csv")
        sorted_snapshot_results.to_csv(snapshot_output_file, index=False)

    if wap_results:
        combined_wap_results = pd.concat(wap_results, ignore_index=True)
        sorted_wap_results = combined_wap_results.sort_values(by=(['start_time'])).reset_index(drop=True)
        wap_output_file = os.path.join(wap_results_folder, f"{symbol}_wap_results.csv")
        sorted_wap_results.to_csv(wap_output_file, index=False)

def main_second(symbol: str, source_folder: str, output_folder: str):
    '''
    主力合约和次主力合约的wap数据
    '''

    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹

    all_files = []
    for root, dirs, files in os.walk(source_folder):
        if 'main_contract' in root or 'second' in root:
            for file in files:
                if file.startswith(symbol) and file[len(symbol)].isdigit():
                    try:
                        # file_path = os.path.join(root, file)
                        date_str = file.split('_')[1][:8]
                        date_index = pd.to_datetime(date_str, format='%Y%m%d')
                        file_path = os.path.join(root, file)
                        all_files.append((file_path, date_index))
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")
                        continue
    lengths = ['-00:01:00:000', '-00:03:00:000', '-00:05:00:000', '00:01:00:000', '00:03:00:000',
               '00:05:00:000']
    all_files.sort(key=lambda x: x[1])
    wap_results = []
    query_counter = 0
    for file_path, _ in all_files:
        print(file_path)
        day_quote = pd.read_parquet(file_path)
        if day_quote.empty:
            continue
        unique_dates = day_quote['date'].unique()
        for date_str in unique_dates:
            date_formatted = pd.to_datetime(date_str).strftime('%Y-%m-%d')
            date_time_list = [f"{date_formatted} 14:30:00", f"{date_formatted} 22:30:00"]


            for start_datetime_str in date_time_list:
                print(start_datetime_str)
                wap_result = wap(day_quote, symbol, start_datetime_str, *lengths)
                '''
                此处可添加snapshoot数据生成
                '''
                if wap_result is not None:
                    query_counter += 1
                    wap_results.append(wap_result)

                    # 每处理20次输出一次结果
                    if query_counter % 20 == 0:
                        if wap_results:
                            combined_wap_results = pd.concat(wap_results, ignore_index=True)
                            sorted_wap_results = combined_wap_results.sort_values(by=(['start_time'])).reset_index(
                                drop=True)
                            wap_output_file = os.path.join(wap_results_folder, f"{symbol}_wap_results.csv")
                            sorted_wap_results.to_csv(wap_output_file, index=False)


        if wap_results:
            combined_wap_results = pd.concat(wap_results, ignore_index=True)
            sorted_wap_results = combined_wap_results.sort_values(by=(['start_time'])).reset_index(drop=True)
            wap_output_file = os.path.join(wap_results_folder, f"{symbol}_wap_results.csv")
            sorted_wap_results.to_csv(wap_output_file, index=False)
# 测试代码
if __name__ == '__main__':

    # 指定文件夹
    source_folder = '/nas92/data/future/quote'
    snapshot_results_folder = "/nas92/temporary/Steiner/data_wash/linux_so/py311/snapshot_results_index/"
    wap_results_folder = "/nas92/temporary/Steiner/data_wash/linux_so/py311/wap_temp/"

    symbols = ["A", "AG", "AL", "AO", "AP", "AU", "B", "BB", "BC", "BR"]



    for symbol in symbols:
        process_quotes(symbol, source_folder, wap_results_folder)