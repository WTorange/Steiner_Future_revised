
import os
import pandas as pd
pd.options.mode.chained_assignment = None
from glob import glob
import numpy as np
from datetime import datetime
import re
from datetime import datetime, timedelta




def get_quote(symbol, base_path="/nas92/data/future/index"):
    """
    get_nearby_snapshot&wap.py
    从组内数据库清洗过的quote数据，生成每天价格指数的snapshot到/nas92/data/future/trade_buffer/ buffer_day_index_daybar/和wap数据到/nas92/temporary/Steiner/data_wash/linux_so/py311/wap_results_index/
    输入：期货品种代码 symbol和数据文件夹/nas92/data/future/index，数据输出文件夹
    输出：snapshot和wap数据
    """
    folder_path = os.path.join(base_path, symbol)
    file_pattern = f"{folder_path}/{symbol}_*_weightedaverage.parquet"

    files = glob(file_pattern)

    files_with_dates = []
    for file in files:

        date_str = os.path.basename(file).split('_')[1]
        date = datetime.strptime(date_str, "%Y%m%d")
        files_with_dates.append((date, file))

    files_with_dates.sort()


    sorted_files = [file for date, file in files_with_dates]
    if len(sorted_files) <= 2:
        print(f"Insufficient files to process for symbol: {symbol}. Required at least 2 files, but got {len(sorted_files)}.")
        return

    for i in range(len(sorted_files) - 2):
        file1 = sorted_files[i]
        file2 = sorted_files[i + 1]
        file3 = sorted_files[i + 2]
        df1 = pd.read_parquet(file1)
        df2 = pd.read_parquet(file2)
        df3 = pd.read_parquet(file3)


        combined_df = pd.concat([df1, df2, df3])

        yield combined_df


def wap(day_quote, symbol,start_datetime_str, *lengths):


    day_quote['resample_time'] = pd.to_datetime(day_quote['resample_time'])
    day_quote['time'] = day_quote['resample_time'].dt.strftime('%H%M%S%f').str.slice(0, 9)
    day_quote['date'] = day_quote['resample_time'].dt.strftime('%Y%m%d')
    result = pd.DataFrame(columns=[ 'date', 'start_time','trading_date'])

    start_datetime = start_datetime_str
    date = start_datetime.strftime('%Y%m%d')


    for length in lengths:

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
            print( f"empty period, date={date}, start_time={start_datetime}")

            return None

        if not df_period.empty:

            df_period['volume_diff'] = df_period['volume'].diff().fillna(df_period['volume'])

            twap = df_period['middle_price'].mean()
            if df_period['volume_diff'].sum() != 0 and not np.isnan(df_period['volume_diff'].sum()):
                vwap = np.dot(df_period['middle_price'], df_period['volume_diff']) / df_period['volume_diff'].sum()
            else:
                vwap = df_period['middle_price'].iloc[0]
            trading_date = df_period['trading_date'].iloc[0]
            result.at[0,'symbol'] = symbol
            result.at[0, 'date'] = date
            result.at[0, 'start_time'] = str(start_datetime)
            result.at[0, 'datetime'] = start_datetime
            result.at[0, 'trading_date'] = trading_date
            result[f'{length_desc}_twap_{"pre_" if is_backward else "post_"}prc'] = twap
            result[f'{length_desc}_vwap_{"pre_" if is_backward else "post_"}prc'] = vwap

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

        last_tick_time = last_tick['resample_time'].time()

        if (query_time.time() >= datetime.strptime("21:05:00", '%H:%M:%S').time() and
                last_tick_time < datetime.strptime("15:05:00", '%H:%M:%S').time()):
            null_value = 1
        elif last_tick['resample_time'].date() != query_date:
            null_value = 1
        else:
            null_value = 0


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


    if null_value == 0:
        if last_tick['daynight'] == "night":

            if last_tick['resample_time'].time() >= datetime.strptime("200000", '%H%M%S').time():
                night_start_time = last_tick['resample_time'].replace(hour=20, minute=0, second=0, microsecond=0)
            else:
                night_start_time = (last_tick['resample_time'] - timedelta(days=1)).replace(hour=20, minute=0, second=0,
                                                                                       microsecond=0)

            high = df[(df['resample_time'] >= night_start_time) & (df['resample_time'] <= query_time)]['last_prc'].max()

            low = df[(df['resample_time'] >= night_start_time) & (df['resample_time'] <= query_time)]['last_prc'].min()

            ask = last_tick['ask_prc1']
            bid = last_tick['bid_prc1']
            volume = last_tick['volume']
            amount = last_tick['turnover']
            night_df = df[df['resample_time'] >= night_start_time]
            open_price = night_df.iloc[0]['last_prc'] if not night_df.empty else None


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

                prev_close_time_start -= timedelta(hours=12)
                prev_close_time_end -= timedelta(hours=12)
                retry_count += 1


                if retry_count >=30:
                    prev_close = None
                    break


        elif last_tick['daynight'] == "day":
            day_start_time = last_tick['resample_time'].replace(hour=8, minute=0, second=0, microsecond=0)

            high = df[(df['resample_time'] >= day_start_time) & (df['resample_time'] <= query_time)]['last_prc'].max()
            low = df[(df['resample_time'] >= day_start_time) & (df['resample_time'] <= query_time)]['last_prc'].min()

            ask = last_tick['ask_prc1']
            bid = last_tick['bid_prc1']

            day_df = df[df['resample_time'] >= day_start_time]
            open_price = day_df.iloc[0]['last_prc'] if not day_df.empty else None

            # ????amount
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
    snapshot_results = []
    wap_results = []
    query_counter = 0
    processed_times = set()
    for combined_df in get_quote(symbol):
        unique_dates = pd.to_datetime(combined_df['trading_date']).dt.date.unique()
        if len(unique_dates) < 2:
            continue
        combined_df['resample_time'] = pd.to_datetime(combined_df['resample_time'])
        combined_df = combined_df.sort_values('resample_time')

        # 提取日期部分
        combined_df['date'] = combined_df['resample_time'].dt.date


        for date, group in combined_df.groupby('date'):
            # 筛选早上八点到晚上八点的记录
            day_time_df = group[(group['resample_time'].dt.time >= pd.to_datetime('08:00:00').time()) &
                                (group['resample_time'].dt.time < pd.to_datetime('20:00:00').time())]

            if not day_time_df.empty:
                # 获取此时间段的最后一条记录
                last_day_time = day_time_df['resample_time'].iloc[-1]
                if last_day_time not in processed_times:
                    processed_times.add(last_day_time)
                    print(symbol, last_day_time)
                    quote_data = combined_df  # 保留完整数据集
                    snapshot_result_day = certain_snapshot(quote_data, last_day_time.strftime('%Y-%m-%d %H:%M:%S'),
                                                             symbol=symbol)
                    snapshot_results.append(snapshot_result_day)
                    query_counter += 1

            # 筛选晚上八点到第二天早上八点的记录
            night_time_df = group[(group['resample_time'].dt.time >= pd.to_datetime('20:00:00').time()) |
                                        (group['resample_time'].dt.time < pd.to_datetime('08:00:00').time())]

            if not night_time_df.empty:
                # 获取此时间段的最后一条记录
                last_night_time = night_time_df['resample_time'].iloc[-1]
                if last_night_time not in processed_times:
                    processed_times.add(last_night_time)
                    print(symbol, last_night_time)
                    quote_data = combined_df  # 保留完整数据集
                    snapshot_result_night = certain_snapshot(quote_data,
                                                               last_night_time.strftime('%Y-%m-%d %H:%M:%S'),
                                                               symbol=symbol)
                    snapshot_results.append(snapshot_result_night)
                    query_counter += 1

            # 每5次查询后合并并保存结果
            if query_counter % 10 == 0:
                if snapshot_results:
                    combined_snapshot_results = pd.concat(snapshot_results, ignore_index=True)
                    sorted_snapshot_results = combined_snapshot_results.sort_values(by=['query_time']).reset_index(
                        drop=True)
                    snapshot_output_file = os.path.join(snapshot_results_folder, f"{symbol}_daynight.csv")
                    sorted_snapshot_results.to_csv(snapshot_output_file, index=False)

    # 最后合并并保存结果
    if snapshot_results:
        combined_snapshot_results = pd.concat(snapshot_results, ignore_index=True)
        sorted_snapshot_results = combined_snapshot_results.sort_values(by=['query_time']).reset_index(drop=True)
        snapshot_output_file = os.path.join(snapshot_results_folder, f"{symbol}_daynight.csv")
        sorted_snapshot_results.to_csv(snapshot_output_file, index=False)

if __name__ == '__main__':


    snapshot_results_folder = "/nas92/data/future/trade_buffer/buffer_day_index_daybar/"
    wap_results_folder = "/nas92/temporary/Steiner/data_wash/linux_so/py311/wap_results_index/"

    # symbols = ["A", "AG", "AL", "AO", "AP", "AU", "B", "BB", "BC", "BU"]
    symbols= ['A']
    for symbol in symbols:
        process_quotes(symbol, snapshot_results_folder, wap_results_folder)