from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
from datetime import datetime
import re
from datetime import datetime, timedelta
import logging

from ApiClient import ApiClient
from logbook import Logger
from Common.logger import get_logger
import time as t
from quote_wash2 import DataProcessor

# 思路
def get_logger_snapshot(name, debug=False):

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        handler = logging.FileHandler('log_file.log')
        handler.setLevel(logging.DEBUG if debug else logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class get_snapshot(ApiClient):


    def __init__(self, api_client, contract: str, time: str):

        self.api_client = api_client
        self.contract = contract
        self.time = time
        self.future_info = pd.read_csv('future_information.csv')
        # 交易所日历
        self.calendar = pd.read_csv('calendar.csv').rename(
            columns={'TRADE_DAYS': 'trade_days', 'S_INFO_EXCHMARKET': 'exchmarket'})
        self.calendar['trade_days'] = pd.to_datetime(self.calendar['trade_days'], format='%Y%m%d').dt.date
        self.logger = get_logger_snapshot(name="get_snapshot", debug=False)
        # 期货开盘时间
        self.opening_hours_df = pd.read_csv('future_information.csv')


    def extract_contract(self, contract):

        match = re.match(r'^([A-Z]{1,2})', contract)
        return match.group(1) if match else contract

    def correct_czc_code(self, contract, query_date):

        if contract.endswith('.CZC'):
            # 提取期货代码中的字母和数字部分
            match = re.match(r'^([A-Z]+)(\d+)\.CZC$', contract)
            if match:
                letters, numbers = match.groups()
                # 提取查询时间的年份
                string_date = query_date.strftime('%Y')
                year = int(string_date)
                # year = int(query_date[:4])
                # 修正数字部分，在第一位加上年份
                if len(numbers) == 3 and numbers[0] != '9' and year >= 2019:
                    corrected_numbers = '2' + numbers
                elif len(numbers) == 3:
                    corrected_numbers = '1' + numbers
                else:
                    corrected_numbers = numbers
                return f"{letters}{corrected_numbers}.CZC"
        return contract


    def get_quote_data(self):

        entries = [self.contract]
        entry_type = 'quote'

        future_code = self.extract_contract(self.contract)


        query_datetime = datetime.strptime(self.time, '%Y-%m-%d %H:%M:%S')
        query_date = query_datetime.date()

        calendar = self.calendar
        calendar['trade_days']: str = calendar['trade_days'].astype(str)

        # 根据calendar获取对应交易所的开盘时间
        exchange = self.contract.split('.')[-1][:2]
        exchange_rows = calendar[calendar['exchmarket'].str.startswith(exchange)].copy()
        exchange_rows['trade_days'] = pd.to_datetime(exchange_rows['trade_days'], format='%Y-%m-%d')

        # 根据交易日历获取start_date和end_date
        def get_start_end_dates(query_date, exchange_rows):
            query_date = pd.to_datetime(query_date)
            exchange_rows = exchange_rows.sort_values(by='trade_days').reset_index(drop=True)

            # 取查询时间前两个交易日作为开始日期
            previous_days = exchange_rows[exchange_rows['trade_days'] < query_date].tail(1)
            start_date = previous_days.iloc[0]['trade_days']
            # 取查询日期后一个交易日作为结束日期
            next_day = exchange_rows[exchange_rows['trade_days'] > query_date].head(1)
            end_date = next_day.iloc[0]['trade_days']

            return start_date, end_date

        start_date, end_date = get_start_end_dates(query_date, exchange_rows)

        start_time = start_date.strftime('%Y-%m-%d')
        end_time = end_date.strftime('%Y-%m-%d')

        # 使用处理后的（尤其是郑商所）的期货合约代码
        corrected_contract = self.correct_czc_code(self.contract, query_date)
        print(corrected_contract)
        entries = [corrected_contract]

        # 由数据库获取quote数据
        day_quote = self.api_client.query_history(entries, entry_type, start_time, end_time)
        day_quote = day_quote[0]

        # 当库内没有对应数据时，会返回一个空的list或者Nonetype, 打日志并跳过
        if day_quote is None or len(day_quote) == 0:
            self.logger.warning(
                f"不存在 {start_date} 到 {end_date} 的数据 contract={contract}, trading_date={query_date}")
            print('day_quote is None')
            return None

        # 保留特定的列
        required_columns = ['symbol', 'datetime', 'last_prc',
                            'volume', 'turnover', 'ask_prc1', 'bid_prc1', 'trading_date', 'open_interest']

        missing_columns = [col for col in required_columns if col not in day_quote.columns]

        # 如果关键字段缺失，打日志+跳过
        if any(col in ['symbol', 'datetime', 'volume', 'turnover', 'last_prc', 'ask_prc1', 'bid_prc1'] for col in
               missing_columns):
            self.logger.warning(
                f"数据字段有缺失 contract={contract}, trading_date={query_date}, 缺失字段={missing_columns}")
            return None

        # 如果date_time在missing_columns 里面，新建一个time列
        # if 'date' in missing_columns or 'time' in missing_columns:
        day_quote['datetime'] = pd.to_datetime(day_quote['datetime'])
        # day_quote['time'] = day_quote['datetime'].dt.strftime('%H%M%S%f').str.slice(0, 9)
        # day_quote['date'] = day_quote['datetime'].dt.strftime('%Y%m%d')

        day_quote = day_quote[required_columns]

        # 使用数据清洗程序quote_wash清洗
        data_processor = DataProcessor(future_index=future_code, debug=True)
        day_quote = data_processor.process(day_quote, future_code)


        if day_quote is None or len(day_quote) == 0:
            self.logger.warning(
                f"data after wash is None {start_date} 到 {end_date} 的数据 contract={contract}, trading_date={query_date}")
            print('day_quote is None')
            return None
        day_quote['time'] = day_quote['time'].astype(str)
        day_quote['date'] = day_quote['date'].astype(str)
        day_quote['trading_date'] = day_quote['trading_date'].astype(str)

        return day_quote
    def wap(day_quote, contract, start_datetime_str, *lengths):
        """
        计算指定时间段内的时间加权平均价格（TWAP）和成交量加权平均价格（VWAP）。

        :param date: 日期（格式：YYYY-MM-DD）
        :param start_time: 开始时间（格式：HHMMSSfff）
        :param lengths: 时间长度（格式：HH:MM:SS:SSS）

        :return: 包含 TWAP 和 VWAP 的 DataFrame
        """

        result = pd.DataFrame(columns=['contract', 'date', 'start_time'])

        # start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%d %H:%M:%S')
        start_datetime = start_datetime_str
        date = start_datetime.strftime('%Y%m%d')

        day_quote['resample_time'] = pd.to_datetime(day_quote['resample_time'])
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

            df_period['middle_price'] = np.where(
                (df_period['ask_prc1'] != 0) & (df_period['bid_prc1'] != 0),
                (df_period['ask_prc1'] + df_period['bid_prc1']) / 2,
                0
            )
            if df_period.empty:
                print( f"无数据， date={date}, start_time={start_datetime}")

                return None

            if not df_period.empty:
                # 计算 VWAP 时需要考虑成交量的差值
                df_period['volume_diff'] = df_period['volume'].diff().fillna(df_period['volume'])

                twap = df_period['middle_price'].mean()
                if df_period['volume_diff'].sum() != 0 and not np.isnan(df_period['volume_diff'].sum()):
                    vwap = np.dot(df_period['middle_price'], df_period['volume_diff']) / df_period['volume_diff'].sum()
                else:
                    vwap = df_period['middle_price'].iloc[0]

                result.at[0, 'contract'] = contract
                result.at[0, 'date'] = date
                result.at[0, 'start_time'] = str(start_datetime)
                result.at[0, 'datetime'] = start_datetime
                result[f'{length_desc}_twap_{"pre_" if is_backward else "post_"}prc'] = twap
                result[f'{length_desc}_vwap_{"pre_" if is_backward else "post_"}prc'] = vwap

        return result

    def certain_snapshot(self, df, time):

        query_time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')


        df['time'] = df['time'].apply(lambda x: str(x).zfill(9))
        df['date'] = df['datetime'].dt.strftime('%Y%m%d')

        df['datetime'] = pd.to_datetime(df['date'] + df['time'], format='%Y%m%d%H%M%S%f')
        # contract = self.extract_contract(self.contract)

        future_code = self.extract_contract(self.contract)
        # 获取 contract 后面的交易所信息
        exchange = self.contract.split('.')[-1][:2]


        # 找到查询时间之前的最后一个tick
        df_before_query = df[df['datetime'] <= query_time]
        query_date = query_time.date()
        if df_before_query.empty:
            self.logger.warning(
                f"无数据，contract={contract}, trading_date={query_date},")
            return None
        else:
            last_tick = df_before_query.iloc[-1]
            # 增加对夜盘缺失的判断
            last_tick_time = last_tick['datetime'].time()

            if (query_time.time() >= datetime.strptime("21:05:00", '%H:%M:%S').time() and
                    last_tick_time < datetime.strptime("15:05:00", '%H:%M:%S').time()):
                null_value = 1
            elif last_tick['datetime'].date() != query_date:
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
                if last_tick['datetime'].time() >= datetime.strptime("200000", '%H%M%S').time():
                    night_start_time = last_tick['datetime'].replace(hour=20, minute=0, second=0, microsecond=0)
                else:
                    night_start_time = (last_tick['datetime'] - timedelta(days=1)).replace(hour=20, minute=0, second=0,
                                                                                           microsecond=0)
                # 根据last_prc计算高开低收, amount等
                high = df[(df['datetime'] >= night_start_time) & (df['datetime'] <= query_time)]['last_prc'].max()

                low = df[(df['datetime'] >= night_start_time) & (df['datetime'] <= query_time)]['last_prc'].min()

                ask = last_tick['ask_prc1']
                bid = last_tick['bid_prc1']
                volume = last_tick['volume']
                amount = last_tick['turnover']
                night_df = df[df['datetime'] >= night_start_time]
                open_price = night_df.iloc[0]['last_prc'] if not night_df.empty else None

                # 根据last_tick的时间判断之前一个盘的时间区间（日盘or夜盘）
                if last_tick['datetime'].time() >= datetime.strptime("200000", '%H%M%S').time():
                    prev_close_time_start = last_tick['datetime'].replace(hour=8, minute=0, second=0, microsecond=0)
                    prev_close_time_end = last_tick['datetime'].replace(hour=20, minute=0, second=0, microsecond=0)
                else:
                    prev_close_time_start = (last_tick['datetime'] - timedelta(days=1)).replace(hour=8, minute=0,
                                                                                                second=0,
                                                                                                microsecond=0)
                    prev_close_time_end = (last_tick['datetime'] - timedelta(days=1)).replace(hour=20, minute=0,
                                                                                              second=0,
                                                                                              microsecond=0)
                retry_count = 0
                while True:
                    prev_close_df = df[
                        (df['datetime'] >= prev_close_time_start) & (df['datetime'] < prev_close_time_end)]
                    if not prev_close_df.empty:
                        prev_close = prev_close_df.iloc[-1]['last_prc']
                        break
                    # 如果上一个时间段没有数据，则往前推12小时
                    prev_close_time_start -= timedelta(hours=12)
                    prev_close_time_end -= timedelta(hours=12)
                    retry_count += 1

                    # 如果计数器达到20，跳出循环并设置 prev_close 为 None 或其他空值
                    if retry_count >= 20:
                        prev_close = None
                        break

            # 日盘的情况与夜盘相同
            elif last_tick['daynight'] == "day":
                day_start_time = last_tick['datetime'].replace(hour=8, minute=0, second=0, microsecond=0)

                high = df[(df['datetime'] >= day_start_time) & (df['datetime'] <= query_time)]['last_prc'].max()
                low = df[(df['datetime'] >= day_start_time) & (df['datetime'] <= query_time)]['last_prc'].min()

                ask = last_tick['ask_prc1']
                bid = last_tick['bid_prc1']

                day_df = df[df['datetime'] >= day_start_time]
                open_price = day_df.iloc[0]['last_prc'] if not day_df.empty else None

                # 计算amount
                last_volume = last_tick['volume']
                first_volume = day_df.iloc[0]['volume'] if not day_df.empty else 0
                last_amount = last_tick['turnover']
                first_amount = day_df.iloc[0]['turnover'] if not day_df.empty else 0

                volume = last_volume - first_volume if not day_df.empty else 0
                amount = last_amount - first_amount if not day_df.empty else 0
                prev_close_time_start = (last_tick['datetime'] - timedelta(days=1)).replace(hour=20, minute=0, second=0,
                                                                                            microsecond=0)
                prev_close_time_end = last_tick['datetime'].replace(hour=8, minute=0, second=0, microsecond=0)

                retry_count = 0
                while True:
                    prev_close_df = df[
                        (df['datetime'] >= prev_close_time_start) & (df['datetime'] < prev_close_time_end)]
                    if not prev_close_df.empty:
                        prev_close = prev_close_df.iloc[-1]['last_prc']
                        break
                    retry_count += 1
                    prev_close_time_start -= timedelta(hours=12)
                    prev_close_time_end -= timedelta(hours=12)
                    if retry_count >= 20:
                        prev_close = None
                        break

        result = {
            'contract': self.extract_contract(self.contract),
            'contract': self.contract,
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

        exchange_rows = self.calendar[self.calendar['exchmarket'].str.startswith(exchange)]

        last_tick_date = datetime.strptime(last_tick['date'], '%Y%m%d').date()
        if query_date in exchange_rows['trade_days'].values:
            if query_date != last_tick_date:
                self.logger.warning(
                    f"未找到交易信息：contract={self.contract}, trading_date={query_date}, 交易所开盘但无数据")

        result_df = pd.DataFrame([result])

        return result_df



# 测试代码
if __name__ == '__main__':

    # 指定你的存放主力合约的文件夹
    maincontract_folder = "/nas92/temporary/Steiner/data_wash/linux_so/py311/12/"
    snapshot_results_folder = "/nas92/temporary/Steiner/data_wash/linux_so/py311/snapshot_results_oi/"
    wap_results_folder = "/nas92/temporary/Steiner/data_wash/linux_so/py311/wap_results_oi/"


    api_client = ApiClient("zlt_research1", "zlt_research1", "./config/api.json", get_logger(name="ApiClientLog", debug=False))
    if not api_client.init():
        raise Exception("ApiClient Error")
    file_names = sorted([file_name for file_name in os.listdir(maincontract_folder) if file_name.endswith(".csv")])
    # 遍历所有主力合约
    for file_name in file_names:
        if file_name.endswith(".csv"):
            file_path = os.path.join(maincontract_folder, file_name)
            contracts_df = pd.read_csv(file_path)
            contracts_df.dropna(inplace=True)
        print(file_name)
        contracts_df['date'] = pd.to_datetime(contracts_df['date'])

        wap_results = []
        snapshot_results = []
        start_date = pd.Timestamp('2004-01-01')
        end_date = pd.Timestamp('2024-07-10')
        filtered_contracts_df = contracts_df[(contracts_df['date'] >= start_date) & (contracts_df['date'] <= end_date)]
        reversed_contracts_df = filtered_contracts_df.iloc[::-1]
        query_counter = 0

        for _, row in reversed_contracts_df.iterrows():
            contract = row['contract']
            dates = row['date']
            print(contract)

            # 生成查询的日期时间列表
            date_str = dates.strftime('%Y-%m-%d')
            date_time_list = [f"{date_str} 14:30:00" ,f"{date_str} 22:30:00"]

            snapshot = get_snapshot(api_client, contract, " ")

            # 对每个日期时间进行查询
            for query_time in date_time_list:
                print(query_time)
                snapshot.time = query_time
                quote_data = snapshot.get_quote_data()
                # print(quote_data)
                if quote_data is None:
                    continue
                lengths = ['-00:01:00:000', '-00:03:00:000', '-00:05:00:000', '00:01:00:000', '00:03:00:000',
                           '00:05:00:000']
                snapshot_result = snapshot.certain_snapshot(quote_data, query_time)

                # wap_result = snapshot.wap(day_quote=quote_data, contract=contract, start_datetime_str= query_time, *lengths )
                wap_result = get_snapshot.wap(quote_data, contract, datetime.strptime(query_time, '%Y-%m-%d %H:%M:%S'), *lengths)


                wap_results.append(wap_result)
                snapshot_results.append(snapshot_result)
                query_counter += 1
                if query_counter % 20 == 0:
                    if snapshot_results:
                        combined_snapshot_results = pd.concat(snapshot_results, ignore_index=True)
                        sorted_snapshot_results = combined_snapshot_results.sort_values(by=['query_time']).reset_index(drop=True)
                        snapshot_output_file = os.path.join(snapshot_results_folder, f"{os.path.splitext(file_name)[0]}_results.csv")
                        sorted_snapshot_results.to_csv(snapshot_output_file, index=False)
                    if wap_results:
                        combined_wap_results = pd.concat(wap_results, ignore_index=True)
                        sorted_wap_results = combined_wap_results.sort_values(by=['start_time']).reset_index(drop=True)
                        wap_output_file = os.path.join(wap_results_folder, f"{os.path.splitext(file_name)[0]}_results.csv")
                        sorted_wap_results.to_csv(wap_output_file, index=False)
        if snapshot_results:
            combined_snapshot_results = pd.concat(snapshot_results, ignore_index=True)
            sorted_snapshot_results = combined_snapshot_results.sort_values(by=['query_time']).reset_index(drop=True)
            snapshot_output_file = os.path.join(snapshot_results_folder,
                                                 f"{os.path.splitext(file_name)[0]}_results.csv")
            sorted_snapshot_results.to_csv(snapshot_output_file, index=False)
        if wap_results:
            combined_wap_results = pd.concat(wap_results, ignore_index=True)
            sorted_wap_results = combined_wap_results.sort_values(by=['start_time']).reset_index(drop=True)
            wap_output_file = os.path.join(wap_results_folder, f"{os.path.splitext(file_name)[0]}_results.csv")
            sorted_wap_results.to_csv(wap_output_file, index=False)
