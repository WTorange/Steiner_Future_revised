from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pandas as pd
import numpy as np
from datetime import datetime
import re
from datetime import datetime, timedelta
import logging

from ApiClient import ApiClient
from logbook import Logger
from Common.logger import get_logger
import time as t

start_time = t.time()
from quote_wash import DataProcessor
end_time = t.time()
print(f"Importing DataProcessor took {end_time - start_time} seconds")

# 思路
def get_logger_snapshoot(name, debug=False):
    """
    初始化一个日志记录器。

    参数:
    - name (str): 日志记录器的名称。

    返回:
    - logger (logging.Logger): 配置好的日志记录器。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        handler = logging.FileHandler('log_file.log')
        handler.setLevel(logging.DEBUG if debug else logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class get_snapshoot(ApiClient):
    """
    从ApiClient类继承，用于获取期货品种快照的类。

    参数:
    - future_index (str): 期货品种索引。
    - time (str): 查询的时间，格式为 'YYYY-MM-DD HH:MM:SS'。
    首先根据期货的合约名称确定期货种类和交易所，
    """

    def __init__(self, future_index: str, time: str):
        """
       初始化get_snapshoot类。

       参数:
       - future_index (str): 期货品种索引。
       - time (str): 查询的时间，格式为 'YYYY-MM-DD HH:MM:SS'。
       """
        super().__init__("zlt_read", "zlt_read", "./config/api.json", get_logger(name="ApiClientLog", debug=False))
        if not self.init():
            raise Exception("初始化失败")

        self.future_index = future_index
        self.time = time
        self.future_info = pd.read_csv('future_information.csv')
        self.calendar = pd.read_csv('calendar.csv').rename(
            columns={'TRADE_DAYS': 'trade_days', 'S_INFO_EXCHMARKET': 'exchmarket'})
        self.calendar['trade_days'] = pd.to_datetime(self.calendar['trade_days'], format='%Y%m%d').dt.date
        self.logger = get_logger_snapshoot(name="get_snapshoot", debug=False)
        self.opening_hours_df = pd.read_csv('future_information.csv')

    def extract_contract(self, future_index):
        """
        提取合约开头的一个或两个大写字母。

        参数:
        - future_index (str): 期货品种索引。

        返回:
        - match (str): 合约开头的大写字母。
        """
        match = re.match(r'^([A-Z]{1,2})', future_index)
        return match.group(1) if match else future_index

    def correct_czc_code(self, contract_code, query_date):
        """
        修正.CZC结尾的期货合约代码。

        参数:
        - contract_code (str): 期货合约代码。
        - query_date (str): 查询的时间，格式为 'YYYY-MM-DD HH:MM:SS'。

        返回:
        - corrected_code (str): 修正后的期货合约代码。
        """
        if contract_code.endswith('.CZC'):
            # 提取期货代码中的字母和数字部分
            match = re.match(r'^([A-Z]+)(\d+)\.CZC$', contract_code)
            if match:
                letters, numbers = match.groups()
                # 提取查询时间的年份
                year = int(query_date[:4])
                # 修正数字部分
                if len(numbers) == 3 and numbers.startswith('0'):
                    if year >= 2019:
                        corrected_numbers = '2' + numbers
                    else:
                        corrected_numbers = '1' + numbers
                else:
                    corrected_numbers = numbers
                return f"{letters}{corrected_numbers}.CZC"
        return contract_code

    def get_quote_data(self):
        """
        根据输入的时间获取相应的期货品种数据。

        返回:
        - day_quote (DataFrame): 包含期货品种数据的DataFrame。
        """
        # 根据输入的时间time，取yyyy-mm-dd，命名为day_time.
        entries = [self.future_index]
        entry_type = 'quote'

        contract_code = self.extract_contract(self.future_index)

        # todo 如需获取期货品种信息
        # future_details = self.future_info[self.future_info['code'] == contract_code]
        # if future_details.empty:
        #     raise Exception("未找到期货品种信息")
        # 判断是否有夜盘
        # has_night_session = future_details.iloc[0]['daynight'] == 1
        part1_time = t
        query_datetime = datetime.strptime(self.time, '%Y-%m-%d %H:%M:%S')
        query_date = query_datetime.date()

        calendar = self.calendar
        calendar['trade_days']: str = calendar['trade_days'].astype(str)

        # 获取对应交易所的开盘时间
        exchange = self.future_index.split('.')[-1][:2]
        exchange_rows = calendar[calendar['exchmarket'].str.startswith(exchange)].copy()
        exchange_rows['trade_days'] = pd.to_datetime(exchange_rows['trade_days'], format='%Y-%m-%d')

        # 根据交易日历获取start_date和end_date
        def get_start_end_dates(query_date, exchange_rows):
            query_date = pd.to_datetime(query_date)
            exchange_rows = exchange_rows.sort_values(by='trade_days').reset_index(drop=True)

            previous_days = exchange_rows[exchange_rows['trade_days'] < query_date].tail(2)
            start_date = previous_days.iloc[0]['trade_days']
            next_day = exchange_rows[exchange_rows['trade_days'] > query_date].head(1)
            end_date = next_day.iloc[0]['trade_days']

            return start_date, end_date

        start_date, end_date = get_start_end_dates(query_date, exchange_rows)

        start_time = start_date.strftime('%Y-%m-%d')
        end_time = end_date.strftime('%Y-%m-%d')

        corrected_future_index = self.correct_czc_code(self.future_index, query_date)
        entries = [corrected_future_index]

        ## 获取quote数据
        day_quote = self.query_history(entries, entry_type, start_time, end_time)
        day_quote = day_quote[0]

        if day_quote is None or len(day_quote) == 0:
            self.logger.warning(
                f"不存在 {start_date} 到 {end_date} 的数据 future_index={future_index}, trading_date={query_date}")
            print('day_quote is None')
            return None

        # 保留特定的列
        required_columns = ['symbol', 'date', 'time', 'datetime', 'last_prc',
                            'volume', 'turnover', 'ask_prc1', 'bid_prc1', 'trading_date', 'open_interest']

        missing_columns = [col for col in required_columns if col not in day_quote.columns]

        # 如果关键字段缺失，打日志+跳过
        if any(col in ['symbol', 'datetime', 'volume', 'turnover', 'past_prc', 'ask_prc1', 'bid_prc1'] for col in
               missing_columns):
            self.logger.warning(
                f"数据字段有缺失 future_index={future_index}, trading_date={query_date}, 缺失字段={missing_columns}")
            return None

        # 如果date_time在missing_columns 里面，新建一个time列
        # if 'date' in missing_columns or 'time' in missing_columns:
        day_quote['datetime'] = pd.to_datetime(day_quote['datetime'])
        day_quote['time'] = day_quote['datetime'].dt.strftime('%H%M%S%f').str.slice(0, 9)
        day_quote['date'] = day_quote['datetime'].dt.strftime('%Y%m%d')

        day_quote = day_quote[required_columns]


        # 使用数据清洗程序
        data_processor = DataProcessor(future_index=contract_code, debug=True)
        day_quote = data_processor.process(day_quote, contract_code)

        day_quote['time'] = day_quote['time'].astype(str)
        day_quote['date'] = day_quote['date'].astype(str)
        day_quote['trading_date'] = day_quote['trading_date'].astype(str)

        # print( day_quote['time'])




        # contract_hours = self.opening_hours_df.loc[self.opening_hours_df['code'] == contract_code, 'hours'].values[0]

        # def time_to_ms(hour, minute, second, ms):
        #     return hour * 3600 * 1000 + minute * 60 * 1000 + second * 1000 + ms
        #
        # def filter_trading_data(df):
        #     trading_time = str(df['time']).zfill(9)  # 获取交易时间，补全到9位
        #     trading_hour = int(trading_time[:2])  # 提取小时部分，转换为整数
        #     trading_minute = int(trading_time[2:4])  # 提取分钟部分，转换为整数
        #     trading_second = int(trading_time[4:6])  # 秒部分
        #     trading_ms = int(trading_time[6:9])
        #     trading_total_ms = time_to_ms(trading_hour, trading_minute, trading_second, trading_ms)
        #
        #     # 拆分开盘时间段，处理多个时间段
        #     for period in contract_hours.split():
        #         start, end = period.split('-')
        #         start_hour, start_minute = map(int, start.split(':'))
        #         end_hour, end_minute = map(int, end.split(':'))
        #
        #         start_second, end_second = 0, 0  # 开始和结束时间的秒数默认为0
        #         start_ms, end_ms = 0, 0
        #         start_total_ms = time_to_ms(start_hour, start_minute, start_second, start_ms)
        #
        #         if end_hour < start_hour or (end_hour == start_hour and end_minute < start_minute):
        #             end_total_ms = time_to_ms(end_hour + 24, end_minute, end_second, end_ms)
        #         else:
        #             end_total_ms = time_to_ms(end_hour, end_minute, end_second, end_ms)
        #
        #         # 判断交易时间是否在任何一个开盘时间段内
        #         if start_total_ms <= trading_total_ms <= end_total_ms:
        #             return True
        #
        #     return False
        #
        # day_quote = day_quote[day_quote.apply(filter_trading_data, axis=1)]

        # 转换时间格式并增加'daynight'列

        def convert_time(row):
            time_str =  str(row['time']).split('.')[0]
            time_obj = datetime.strptime(time_str, '%H%M%S%f')
            if datetime.strptime("080000000000", '%H%M%S%f') <= time_obj <= datetime.strptime(
                    "200000000000", '%H%M%S%f'):
                return "day"
            else:
                return "night"

        day_quote['daynight'] = day_quote.apply(convert_time, axis=1)
        return day_quote

    def certain_snapshoot(self, df, time):
        """
        根据特定时间获取期货品种的快照数据。

        参数:
        - df (DataFrame): 包含期货品种数据的DataFrame。
        - time (str): 查询的时间，格式为 'YYYY-MM-DD HH:MM:SS'。

        返回:
        - result_df (DataFrame): 包含特定时间快照数据的DataFrame。
        """
        # 将传入的时间字符串转换为datetime对象
        query_time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')

        df['time'] = df['time'].apply(lambda x: str(x).zfill(9))  # 确保时间字符串是9位，包括毫秒
        df['date'] = df['datetime'].dt.strftime('%Y%m%d')

        df['datetime'] = pd.to_datetime(df['date'] + df['time'], format='%Y%m%d%H%M%S%f')
        # future_index = self.extract_contract(self.future_index)

        contract_code = self.extract_contract(self.future_index)
        exchange = self.future_index.split('.')[-1][:2]  # 获取 future_index 后面的交易所信息

        # 找到查询时间之前的最后一个tick
        df_before_query = df[df['datetime'] <= query_time]
        query_date = query_time.date()
        if df_before_query.empty:
            self.logger.warning(
                f"无数据，future_index={future_index}, trading_date={query_date},")
            return None
        else:
            last_tick = df_before_query.iloc[-1]
            if (query_time.time() >= datetime.strptime("210500", '%H%M%S').time() and
                    last_tick['datetime'].time() < datetime.strptime("150500", '%H%M%S').time()):
                null_value = 1
            elif last_tick['datetime'].date() != query_date:
                # 若当天数据不存在，向前搜寻最后一行数据
                null_value = 1
            else:
                # 当天数据存在，取最后一个 tick
                null_value = 0

        # 当天数据不存在，按要求输出横线
        if null_value == 1:
            high = low = open_price = prev_close = last_tick['last_prc']
            ask = bid = volume = amount = 0

        # todo daynight也可以加在这里，或许增加兼容性以及提高速度
        # def convert_time(row):
        #     time_str = row['time']
        #     time_obj = datetime.strptime(time_str, '%H%M%S%f')
        #     if datetime.strptime("080000000000", '%H%M%S%f') <= time_obj <= datetime.strptime(
        #             "200000000000", '%H%M%S%f'):
        #         return "day"
        #     else:
        #         return "night"

        # 分日盘和夜盘讨论
        if null_value == 0:
            if last_tick['daynight'] == "night":
                if last_tick['datetime'].time() >= datetime.strptime("200000", '%H%M%S').time():
                    night_start_time = last_tick['datetime'].replace(hour=20, minute=0, second=0, microsecond=0)
                else:
                    night_start_time = (last_tick['datetime'] - timedelta(days=1)).replace(hour=20, minute=0, second=0,
                                                                                           microsecond=0)

                high = df[(df['datetime'] >= night_start_time) & (df['datetime'] <= query_time)]['last_prc'].max()
                low = df[(df['datetime'] >= night_start_time) & (df['datetime'] <= query_time)]['last_prc'].min()

                ask = last_tick['ask_prc1']
                bid = last_tick['bid_prc1']
                volume = last_tick['volume']
                amount = last_tick['turnover']
                night_df = df[df['datetime'] >= night_start_time]
                open_price = night_df.iloc[0]['last_prc'] if not night_df.empty else None

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
                    prev_close_time_start -= timedelta(hours=12)
                    prev_close_time_end -= timedelta(hours=12)
                    retry_count += 1

                    # 如果计数器达到20，跳出循环并设置 prev_close 为 None 或其他空值
                    if retry_count >= 20:
                        prev_close = None
                        break

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
            'future_index': self.extract_contract(self.future_index),
            'contract': self.future_index,
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
                    f"未找到交易信息：future_index={self.future_index}, trading_date={query_date}, 交易所开盘但无数据")

        result_df = pd.DataFrame([result])

        return result_df


# 测试代码
if __name__ == '__main__':

    # 指定你的存放主力合约的文件夹
    maincontract_folder = r"C:\Users\maki\Desktop\quantchina\Futures-main\data_wash\maincontract"
    calendar = pd.read_csv('calendar.csv').rename(
        columns={'TRADE_DAYS': 'trade_days', 'S_INFO_EXCHMARKET': 'exchmarket'})

    calendar['trade_days'] = calendar['trade_days'].astype(str)

    # 遍历所有主力合约
    for file_name in os.listdir(maincontract_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(maincontract_folder, file_name)
            contracts_df = pd.read_csv(file_path)
        print(file_name)

        # todo 方便测试该品种是否能顺利运行
        contracts_df = pd.read_csv(
            r"C:\Users\maki\Desktop\quantchina\Futures-main\data_wash\maincontract\RU_20091201_20240628.csv")
        results = []
        exchange = contracts_df['contract'][0].split('.')[-1][:2]

        reversed_contracts_df = contracts_df.iloc[::-1]
        for _, row in reversed_contracts_df.iterrows():
            future_index = row['contract']
            start_date = str(row['start'])
            end_date = str(row['end'])

            # 筛选出在日期区间内的交易日期
            exchange_rows = calendar[calendar['exchmarket'].str.startswith(exchange)].copy()
            exchange_rows.loc[:, 'trade_days'] = exchange_rows['trade_days'].astype(str)
            # 指定日期
            filtered_calendar = exchange_rows[(exchange_rows['trade_days'] >= '20240624') &
                                              (exchange_rows['trade_days'] >= start_date) &
                                              (exchange_rows['trade_days'] <= end_date)]
            reversed_filtered_calendar = filtered_calendar.iloc[::-1]

            if reversed_filtered_calendar is None:
                continue

            # 生成查询的日期时间列表
            date_time_list = []
            for trade_day in reversed_filtered_calendar['trade_days']:
                trade_day_str = pd.to_datetime(str(trade_day)).strftime('%Y-%m-%d')
                date_time_list.append(pd.to_datetime(str(trade_day)).strftime('%Y-%m-%d') + ' 14:30:00')
                date_time_list.append(pd.to_datetime(str(trade_day)).strftime('%Y-%m-%d') + ' 22:30:00')

            snapshoot = get_snapshoot(future_index, " ")

            # 对每个日期时间进行查询
            for query_time in date_time_list:
                snapshoot.time = query_time
                quote_data = snapshoot.get_quote_data()
                # print(quote_data)
                if quote_data is None:
                    continue

                result = snapshoot.certain_snapshoot(quote_data, query_time)
                print(query_time)
                results.append(result)
        if results:
            combined_results = pd.concat(results, ignore_index=True)
            sorted_results = combined_results.sort_values(by=['query_time']).reset_index(drop=True)
            output_file = os.path.join(maincontract_folder, f"{os.path.splitext(file_name)[0]}_results.csv")
            sorted_results.to_csv(output_file, index=False)

# todo 暂时先保留，后续参考修改
# 以下是多线程 & 多进程的尝试，效果不佳
# from concurrent.futures import ThreadPoolExecutor, as_completed
#
# # 定义 process_future_index 函数
# def process_future_index(row):
#     future_index = row['contract']
#     start_date = pd.to_datetime(str(row['start'])).date()
#     end_date = pd.to_datetime(str(row['end'])).date()
#
#     # 筛选出在日期区间内的交易日期
#     exchange_rows = calendar[calendar['exchmarket'].str.startswith(exchange)].copy()
#     exchange_rows['trade_days'] = pd.to_datetime(exchange_rows['trade_days'], format='%Y%m%d').dt.date
#     # exchange_rows['trade_days'] = pd.to_datetime(exchange_rows['trade_days']).dt.date
#
#     # print('1',type(exchange_rows['trade_days']),'2',type(pd.to_datetime('20240615').date()),'3',type(start_date))
#
#     filtered_calendar = exchange_rows[(exchange_rows['trade_days'] >= pd.to_datetime('20201130').date()) &  # 指定日期之后的条件
#                                       (exchange_rows['trade_days'] >= start_date) &
#                                       (exchange_rows['trade_days'] <= end_date)]
#     reversed_filtered_calendar = filtered_calendar.iloc[::-1]
#
#     # 生成查询的日期时间列表
#     date_time_list = []
#     for trade_day in reversed_filtered_calendar['trade_days']:
#         trade_day_str = pd.to_datetime(str(trade_day)).strftime('%Y-%m-%d')
#         date_time_list.append(pd.to_datetime(str(trade_day)).strftime('%Y-%m-%d') + ' 14:30:00')
#         date_time_list.append(pd.to_datetime(str(trade_day)).strftime('%Y-%m-%d') + ' 22:30:00')
#
#     # 创建 get_snapshoot 实例
#     snapshoot = get_snapshoot(future_index, "")
#
#     # 对每个日期时间进行查询
#     results = []
#     for query_time in date_time_list:
#         snapshoot.time = query_time
#         quote_data = snapshoot.get_quote_data()
#         result = snapshoot.certain_snapshoot(quote_data, query_time)
#         results.append(result)
#         print(query_time)
#     return results
#
#
# contracts_df = pd.read_csv(
#     r"C:\Users\maki\Desktop\quantchina\Futures-main\data_wash\maincontract\RU_20091201_20260624.csv")
# calendar = pd.read_csv('calendar.csv').rename(
#     columns={'TRADE_DAYS': 'trade_days', 'S_INFO_EXCHMARKET': 'exchmarket'})
# calendar['trade_days'] = calendar['trade_days'].astype(str)
#
# exchange = contracts_df['contract'][0].split('.')[-1][:2]
# reversed_contracts_df = contracts_df.iloc[::-1]
#
# # 使用线程池处理每个 future_index
# results = []
# with ThreadPoolExecutor() as executor:
#     futures = [executor.submit(process_future_index, row) for _, row in reversed_contracts_df.iterrows()]
# # with ProcessPoolExecutor() as executor:
# #     futures = [executor.submit(process_future_index, row) for _, row in reversed_contracts_df.iterrows()]
#
#     for future in as_completed(futures):
#         results.extend(future.result())
#
# # 合并结果并输出到 CSV 文件
# combined_results = pd.concat(results, ignore_index=True)
# sorted_results = combined_results.sort_values(by=['query_time'])
# sorted_results.to_csv('RU.csv', index=False)

# 打印过滤后的数据
filtered_data.to_csv('filter_test.csv')
