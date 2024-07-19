import os
import pandas as pd
import numpy as np
from datetime import datetime
from logbook import Logger
from Common.logger import get_logger
import logging
from datetime import datetime, timedelta
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
def get_logger_quote_wash(name, debug=False):
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


# 数据清洗需要做的：
# 明确开始和结束的时间，通过读取CSV文件获得，每个品种读取一次然后作为全局变量。删除在该时间段以外的数据。删除时间异常的数据。 已完成
# 保留特定的列。
# 开盘时间段内的数据时间间隔统一为0.5秒，缺失的部分由之前的数据填充。
# 如果是.CZC，需要补全合约代码

class DataProcessor:
    def __init__(self, future_index, opening_hours_file='future_information.csv', debug=False):
        self.api_client = api_client
        self.future_index = future_index
        self.opening_hours_df = pd.read_csv(opening_hours_file)
        self.logger = get_logger_quote_wash('QuoteProcessor', debug)
        self.opening_hours_df = pd.read_csv('future_information.csv')

    def quote_wash(self, day_quote, future_index):
        if day_quote is None or len(day_quote) == 0:
            return None
        # 保留特定的列
        # required_columns = ['symbol', 'date', 'time', 'datetime', 'last_prc',
        #                     'volume', 'turnover', 'ask_prc1', 'bid_prc1', 'trading_date', 'open_interest']
        required_columns = ['datetime', 'date', 'time']
        missing_columns = [col for col in required_columns if col not in day_quote.columns]

        # if missing_columns:
        #     return None

        # 如果date_time在missing_columns 里面，新建一个time列
        day_quote['datetime'] = pd.to_datetime(day_quote['datetime'])
        if 'date' in missing_columns or 'time' in missing_columns:
            day_quote['datetime'] = pd.to_datetime(day_quote['datetime'])
            day_quote['time'] = day_quote['datetime'].dt.strftime('%H%M%S%f').str.slice(0, 9)
            day_quote['date'] = day_quote['datetime'].dt.strftime('%Y%m%d')

        contract_code = future_index
        # 根据contract_code找到对应的开盘时间

        contract_hours = self.opening_hours_df.loc[self.opening_hours_df['code'] == contract_code, 'hours'].values[0]

        # 过滤时间
        def time_to_ms(hour, minute, second, ms):
            return hour * 3600 * 1000 + minute * 60 * 1000 + second * 1000 + ms

        def filter_trading_data(df):
            trading_time = str(df['time']).zfill(9)  # 获取交易时间，补全到9位
            trading_hour = int(trading_time[:2])  # 提取小时部分，转换为整数
            trading_minute = int(trading_time[2:4])  # 提取分钟部分，转换为整数
            trading_second = int(trading_time[4:6])  # 秒部分
            trading_ms = int(trading_time[6:9])
            trading_total_ms = time_to_ms(trading_hour, trading_minute, trading_second, trading_ms)

            # 拆分开盘时间段，处理多个时间段
            for period in contract_hours.split():
                start, end = period.split('-')
                start_hour, start_minute = map(int, start.split(':'))
                end_hour, end_minute = map(int, end.split(':'))

                start_second, end_second = 0, 0  # 开始和结束时间的秒数默认为0
                start_ms, end_ms = 0, 0
                start_total_ms = time_to_ms(start_hour, start_minute, start_second, start_ms)

                if end_hour < start_hour or (end_hour == start_hour and end_minute < start_minute):
                    end_total_ms = time_to_ms(end_hour + 24, end_minute, end_second, end_ms)
                else:
                    end_total_ms = time_to_ms(end_hour, end_minute, end_second, end_ms)

                # 判断交易时间是否在任何一个开盘时间段内
                if start_total_ms <= trading_total_ms <= end_total_ms:
                    return True

            return False

        day_quote = day_quote[day_quote.apply(filter_trading_data, axis=1)]

        return day_quote

    def resample_data(self, data, contract_code,  freq='500L'):

        data = data.copy()
        # data.loc[:, 'datetime'] = pd.to_datetime(data['datetime'])
        data['datetime'] = pd.to_datetime(data['datetime'])
        # data.loc[:, 'date'] = data['datetime'].dt.date
        data.loc[:, 'trading_date'] = pd.to_datetime(data['trading_date'], format='%Y%m%d')
        data['trading_date'] = pd.to_datetime(data['trading_date']).dt.date
        # data.to_csv('step1.csv')
        resampled_data = pd.DataFrame()
        # 循环处理每一个日期
        for date in data['trading_date'].unique():
            # print(type( data['trading_date'][1]))
            opening_hours_str = self.opening_hours_df.loc[self.opening_hours_df['code'] == contract_code, 'hours'].values[0]
            periods = opening_hours_str.split()

            date_data = data[data['trading_date'] == date].copy()
            original_datetime = date_data['datetime'].copy()
            date_data['resample_time'] = original_datetime

            for period in periods:
                start_time_str, end_time_str = period.split('-')
                start_datetime = datetime.strptime(date.strftime('%Y-%m-%d') + start_time_str, '%Y-%m-%d%H:%M')
                end_datetime = datetime.strptime(date.strftime('%Y-%m-%d') + end_time_str, '%Y-%m-%d%H:%M')

                if (20 <= start_datetime.hour < 24) or (20 <= end_datetime.hour < 24):
                    start_datetime -= timedelta(days=1)


                if 20 <= end_datetime.hour < 24:
                    end_datetime -= timedelta(days=1)

                date_data = date_data.drop_duplicates(subset=['resample_time'])
                all_times = pd.date_range(start=start_datetime, end=end_datetime, freq=freq)
                resampled = date_data.set_index('resample_time').reindex(all_times, method='pad')

                # resampled['date'] = date
                resampled['resample_time'] = resampled.index
                resampled = resampled[resampled['last_prc'].notna()]
                resampled_data = pd.concat([resampled_data, resampled], ignore_index=True)
            resampled_data = resampled_data.sort_values(by='resample_time',ignore_index=True)
            # 添加 middle_price 列
            resampled_data['middle_price'] = np.where(
                (resampled_data['ask_prc1'] != 0) & (resampled_data['bid_prc1'] != 0),
                (resampled_data['ask_prc1'] + resampled_data['bid_prc1']) / 2,
                0
            )
            if 'Unnamed: 0' in resampled_data.columns:
                resampled_data.drop(columns=['Unnamed: 0'], inplace=True)
        return resampled_data

    def process(self, day_quote,future_index):
        washed_quote = self.quote_wash(day_quote,future_index)
        if washed_quote is not None:
            resampled_quote = self.resample_data(washed_quote,future_index)
            return resampled_quote
        return None

def get_logger_snapshoot(name, debug=False):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        handler = logging.FileHandler('data_wash.log')
        handler.setLevel(logging.DEBUG if debug else logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class get_snapshoot(ApiClient):
    def __init__(self, api_client, contract: str, time: str):
        self.api_client = api_client
        self.contract = contract
        self.time = time
        self.future_info = pd.read_csv('future_information.csv')
        # 交易所日历
        # self.calendar = pd.read_csv('calendar.csv').rename(
        #     columns={'TRADE_DAYS': 'trade_days', 'S_INFO_EXCHMARKET': 'exchmarket'})
        # self.calendar['trade_days'] = pd.to_datetime(self.calendar['trade_days'], format='%Y%m%d').dt.date
        self.logger = get_logger_snapshoot(name="get_snapshoot", debug=False)
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
                if len(numbers) == 3 and numbers[0] != '9':
                    if year >= 2019:
                        corrected_numbers = '2' + numbers
                    else:
                        corrected_numbers = '1' + numbers
                else:
                    corrected_numbers = numbers
                return f"{letters}{corrected_numbers}.CZC"
        return contract

    def get_quote_data(self):

        entries = [self.contract]
        entry_type = 'quote'

        query_datetime = datetime.strptime(self.time, '%Y-%m-%d')
        query_date = query_datetime.date()

        # calendar = self.calendar
        # calendar['trade_days']: str = calendar['trade_days'].astype(str)

        # 根据calendar获取对应交易所的开盘时间
        # exchange = self.contract.split('.')[-1][:2]
        # exchange_rows = calendar[calendar['exchmarket'].str.startswith(exchange)].copy()
        # exchange_rows['trade_days'] = pd.to_datetime(exchange_rows['trade_days'], format='%Y-%m-%d')

        def get_start_end_dates(query_date):
            start_date = query_date
            end_date = query_date

            return start_date, end_date

        start_date, end_date = get_start_end_dates(query_date)

        start_time = start_date.strftime('%Y-%m-%d')
        end_time = end_date.strftime('%Y-%m-%d')

        # 使用处理后的（尤其是郑商所）的期货合约代码
        corrected_contract = self.correct_czc_code(self.contract, query_date)
        # print(corrected_contract)
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
                f"数据字段有缺失 contract={self.contract}, trading_date={query_date}, 缺失字段={missing_columns}")
            return None
        day_quote['datetime'] = pd.to_datetime(day_quote['datetime'])

        day_quote = day_quote[required_columns]

        return day_quote

if __name__ == '__main__':

    maincontract_folder = "/nas92/temporary/Steiner/data_wash/linux_so/py311/essentialcontract/"
    results_folder = "/nas92/data/future/"

    api_client = ApiClient("zlt_read", "zlt_read", "./config/api.json", get_logger(name="ApiClientLog", debug=False))
    if not api_client.init():
        raise Exception("ApiClient Error")

    file_names = sorted([file_name for file_name in os.listdir(maincontract_folder) if file_name.endswith(".csv")])
    # 遍历所有主力合约
    for file_name in file_names:
        if file_name.endswith(".csv"):
            file_path = os.path.join(maincontract_folder, file_name)
            contracts_df = pd.read_csv(file_path)
            # contracts_df =pd.read_csv("/nas92/temporary/Steiner/data_wash/linux_so/py311/essentialcontract/RU_20091201_20240710.csv")
            print(file_name)

        results = []

        reversed_contracts_df = contracts_df.iloc[::-1]
        query_counter = 0

        for _, row in reversed_contracts_df.iterrows():

            date = row['date']
            if datetime.strptime(date, '%Y-%m-%d').year < 2014:
                continue
            print(date)
            future_index = row['future_code']
            contracts = ['main_contract', 'second', 'third', 'nearby_contract']
            for contract_type in contracts:
                contract = row[contract_type]
                snapshoot = get_snapshoot(api_client, contract, date)
                day_quote = snapshoot.get_quote_data()
                print(contract)
                if day_quote is not None:
                    processor = DataProcessor(future_index)
                    cleaned_quote = processor.process(day_quote, future_index)

                    date_folder = os.path.join(results_folder,
                                               datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d'))
                    if not os.path.exists(date_folder):
                        os.makedirs(date_folder)

                    contract_folder = os.path.join(date_folder, contract_type)
                    if not os.path.exists(contract_folder):
                        os.makedirs(contract_folder)


                output_file = os.path.join(contract_folder, f"{contract}_{datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')}.csv")
                cleaned_quote.to_csv(output_file, index=False)

            # exchange_rows['trade_days'] = pd.to_datetime(exchange_rows['trade_days'], format='%Y%m%d')
            # exchange_rows = exchange_rows.sort_values(by='trade_days')
            #
            #
            # day_quote = get_quote_data(api_client, contract, " ")
            #
            # # day_quote = pd.read_csv('RU2005.SHF.csv')
            # future_index = 'RU'
            # processor = DataProcessor(future_index)
            # cleaned_quote = processor.process(day_quote, future_index)
            # cleaned_quote.to_csv("check.csv")
            # #
            # #
            #     # 对每个日期时间进行查询
            #     for query_time in date_time_list:
            #         print(query_time)
            #         snapshoot.time = query_time
            #         quote_data = snapshoot.get_quote_data()
            #         # print(quote_data)
            #         if quote_data is None:
            #             continue
            #
            #         result = snapshoot.certain_snapshoot(quote_data, query_time)
            #
            #         results.append(result)
            #         query_counter += 1
            #         if query_counter % 100 == 0:
            #             if results:
            #                 combined_results = pd.concat(results, ignore_index=True)
            #                 sorted_results = combined_results.sort_values(by=['query_time']).reset_index(drop=True)
            #                 output_file = os.path.join(results_folder, f"{os.path.splitext(file_name)[0]}_results.csv")
            #                 sorted_results.to_csv(output_file, index=False)
            # if results:
            #     combined_results = pd.concat(results, ignore_index=True)
            #     sorted_results = combined_results.sort_values(by=['query_time']).reset_index(drop=True)
            #     output_file = os.path.join(results_folder, f"{os.path.splitext(file_name)[0]}_results.csv")
            #     sorted_results.to_csv(output_file, index=False)
    #
