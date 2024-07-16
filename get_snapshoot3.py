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
from quote_wash import DataProcessor

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
    - contract (str): 期货品种索引。
    - time (str): 查询的时间，格式为 'YYYY-MM-DD HH:MM:SS'。

    步骤：
    1. 根据期货的合约名称确定期货种类和交易所，在main_contract文件夹中查找到对应品种的主力合约和日期，calendar.csv表格查找到交易所对应的交易所开盘日期。
    2. 修正.CZC 的期货合约代码，根据输入的时刻和对应的交易所开盘日期在数据库中查找quote数据。
    3. 选择根据quote数据的字段，对time和date字段缺失情况进行补足
    4. 使用quote_wash程序进行数据清洗
    4.1. 根据期货品种在future_information.csv文件中查找对应的开盘时间，删除开盘时间外的数据。
    4.2. 在开盘时间内以0.5s为间隔重采样，方法为pad(ffill)，生成每0.5s的快照数据，新建一列resample_time，为快照数据的时刻。其余数据得到保留
    5. 日盘和夜盘进行分隔，计算要求的高开低收, volume, amount等，向前查找last_prc。
    6. 最后输出对应时刻的快照数据

    """

    def __init__(self, api_client, contract: str, time: str):
        """
       初始化get_snapshoot类。

       参数:
       - contract (str): 期货品种索引。
       - time (str): 查询的时间，格式为 'YYYY-MM-DD HH:MM:SS'。
       """
        # super().__init__("zlt_read", "zlt_read", "./config/api.json", get_logger(name="ApiClientLog", debug=False))
        # if not self.init():
        #     raise Exception("初始化失败")

        self.api_client = api_client
        self.contract = contract
        self.time = time
        self.future_info = pd.read_csv('future_information.csv')
        # 交易所日历
        self.calendar = pd.read_csv('calendar.csv').rename(
            columns={'TRADE_DAYS': 'trade_days', 'S_INFO_EXCHMARKET': 'exchmarket'})
        self.calendar['trade_days'] = pd.to_datetime(self.calendar['trade_days'], format='%Y%m%d').dt.date
        self.logger = get_logger_snapshoot(name="get_snapshoot", debug=False)
        # 期货开盘时间
        self.opening_hours_df = pd.read_csv('future_information.csv')

    def extract_contract(self, contract):
        """
        提取合约开头的一个或两个大写字母。

        参数:
        - contract (str): 期货品种索引。

        返回:
        - match (str): 合约开头的大写字母。
        """
        match = re.match(r'^([A-Z]{1,2})', contract)
        return match.group(1) if match else contract

    def correct_czc_code(self, contract, query_date):
        """
        修正.CZC结尾的期货合约代码。

        参数:
        - future_code (str): 期货合约代码。
        - query_date (str): 查询的时间，格式为 'YYYY-MM-DD HH:MM:SS'。

        返回:
        - corrected_code (str): 修正后的期货合约代码。
        """
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
                if len(numbers) == 3  and numbers[0] != '9':
                    if year >= 2019 :
                        corrected_numbers = '2' + numbers
                    else:
                        corrected_numbers = '1' + numbers
                else:
                    corrected_numbers = numbers
                return f"{letters}{corrected_numbers}.CZC"
        return contract

    def get_quote_data(self):
        """
        根据输入的时间获取相应的期货品种数据。

        返回:
        - day_quote (DataFrame): 包含期货品种数据的DataFrame。
        """

        # 根据输入的时间time，取yyyy-mm-dd，命名为day_time.
        entries = [self.contract]
        entry_type = 'quote'

        # 获取期货的大写字母代码
        future_code = self.extract_contract(self.contract)

        # todo 如需获取期货品种信息
        # future_details = self.future_info[self.future_info['code'] == future_code]
        # if future_details.empty:
        #     raise Exception("未找到期货品种信息")
        # 判断是否有夜盘
        # has_night_session = future_details.iloc[0]['daynight'] == 1


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

        # 转换时间格式并增加'daynight'列

        # def convert_time(row):
        #     time_str =  str(row['time']).split('.')[0]
        #     time_obj = datetime.strptime(time_str, '%H%M%S%f')
        #     if datetime.strptime("080000000000", '%H%M%S%f') <= time_obj <= datetime.strptime(
        #             "200000000000", '%H%M%S%f'):
        #         return "day"
        #     else:
        #         return "night"
        #
        # day_quote['daynight'] = day_quote.apply(convert_time, axis=1)
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


        # 确保时间字符串是9位，包括毫秒
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

            # if last_tick['datetime'].date() != query_date:
            #     null_value = 1
            # else:
            #     null_value = 0

        # 当天数据不存在，按要求输出数据
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

                low = df[(df['datetime'] >= night_start_time) & (df['datetime'] <= query_time)].min()

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
    maincontract_folder = "/nas92/temporary/Steiner/data_wash/linux_so/py311/maincontract/"
    results_folder = "/nas92/temporary/Steiner/data_wash/linux_so/py311/results/"

    calendar = pd.read_csv('calendar.csv').rename(
        columns={'TRADE_DAYS': 'trade_days', 'S_INFO_EXCHMARKET': 'exchmarket'})

    calendar['trade_days'] = calendar['trade_days'].astype(str)
    api_client = ApiClient("zlt_read", "zlt_read", "./config/api.json", get_logger(name="ApiClientLog", debug=False))
    if not api_client.init():
        raise Exception("ApiClient Error")
    file_names = sorted([file_name for file_name in os.listdir(maincontract_folder) if file_name.endswith(".csv")])
    # 遍历所有主力合约
    for file_name in file_names:
        if file_name.endswith(".csv"):
            file_path = os.path.join(maincontract_folder, file_name)
            contracts_df = pd.read_csv(file_path)
        print(file_name)

        # todo 方便测试该品种是否能顺利运行
        # contracts_df = pd.read_csv(
        #     "/nas92/temporary/Steiner/data_wash/linux_so/py311/maincontract/AP_20091201_20240628.csv")
        results = []
        exchange = contracts_df['contract'][0].split('.')[-1][:2]

        reversed_contracts_df = contracts_df.iloc[::-1]
        query_counter = 0

        for _, row in reversed_contracts_df.iterrows():
            contract = row['contract']
            start_date = str(row['start'])
            end_date = str(row['end'])

            # 筛选出在日期区间内的交易日期
            exchange_rows = calendar[calendar['exchmarket'].str.startswith(exchange)].copy()

            exchange_rows['trade_days'] = pd.to_datetime(exchange_rows['trade_days'], format='%Y%m%d')
            exchange_rows = exchange_rows.sort_values(by='trade_days')
            # exchange_rows.loc[:, 'trade_days'] = exchange_rows['trade_days'].astype(str)
            # 指定日期
            filtered_calendar = exchange_rows[(exchange_rows['trade_days'] >= pd.to_datetime('20100101')) &
                                              (exchange_rows['trade_days'] <= pd.to_datetime('20240710')) &
                                              (exchange_rows['trade_days'] >= pd.to_datetime(start_date)) &
                                              (exchange_rows['trade_days'] <= pd.to_datetime(end_date))]
            reversed_filtered_calendar = filtered_calendar.iloc[::-1]

            if reversed_filtered_calendar is None:
                continue

            # 生成查询的日期时间列表
            date_time_list = []
            for trade_day in reversed_filtered_calendar['trade_days']:
                trade_day_str = pd.to_datetime(str(trade_day)).strftime('%Y-%m-%d')
                date_time_list.append(pd.to_datetime(str(trade_day)).strftime('%Y-%m-%d') + ' 14:30:00')
                date_time_list.append(pd.to_datetime(str(trade_day)).strftime('%Y-%m-%d') + ' 22:30:00')

            snapshoot = get_snapshoot(api_client, contract, " ")

            # 对每个日期时间进行查询
            for query_time in date_time_list:
                print(query_time)
                snapshoot.time = query_time
                quote_data = snapshoot.get_quote_data()
                # print(quote_data)
                if quote_data is None:
                    continue

                result = snapshoot.certain_snapshoot(quote_data, query_time)

                results.append(result)
                query_counter += 1
                if query_counter % 100 == 0:
                    if results:
                        combined_results = pd.concat(results, ignore_index=True)
                        sorted_results = combined_results.sort_values(by=['query_time']).reset_index(drop=True)
                        output_file = os.path.join(results_folder, f"{os.path.splitext(file_name)[0]}_results.csv")
                        sorted_results.to_csv(output_file, index=False)
        if results:
            combined_results = pd.concat(results, ignore_index=True)
            sorted_results = combined_results.sort_values(by=['query_time']).reset_index(drop=True)
            output_file = os.path.join(results_folder, f"{os.path.splitext(file_name)[0]}_results.csv")
            sorted_results.to_csv(output_file, index=False)

# todo 暂时先保留，后续参考修改
# 以下是多线程 & 多进程的尝试，效果不佳
# from concurrent.futures import ThreadPoolExecutor, as_completed
#
# # 定义 process_contract 函数
# def process_contract(row):
#     contract = row['contract']
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
#     snapshoot = get_snapshoot(contract, "")
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
# # 使用线程池处理每个 contract
# results = []
# with ThreadPoolExecutor() as executor:
#     futures = [executor.submit(process_contract, row) for _, row in reversed_contracts_df.iterrows()]
# # with ProcessPoolExecutor() as executor:
# #     futures = [executor.submit(process_contract, row) for _, row in reversed_contracts_df.iterrows()]
#
#     for future in as_completed(futures):
#         results.extend(future.result())
#
# # 合并结果并输出到 CSV 文件
# combined_results = pd.concat(results, ignore_index=True)
# sorted_results = combined_results.sort_values(by=['query_time'])
# sorted_results.to_csv('RU.csv', index=False)

# 打印过滤后的数据
# filtered_data.to_csv('filter_test.csv')
