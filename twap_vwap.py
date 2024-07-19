import os
from datetime import datetime
import re
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from ApiClient import ApiClient
from logbook import Logger
from Common.logger import get_logger
from quote_wash_revised import DataProcessor


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
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
        handler = logging.FileHandler('wap.log')
        handler.setLevel(logging.DEBUG if debug else logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

class get_wap(ApiClient):
    def __init__(self, api_client, contract: str, time: str):
        """
       初始化get_snapshoot类。

       参数:
       - contract (str): 期货品种索引。
       - time (str): 查询的时间，格式为 'YYYY-MM-DD HH:MM:SS'。
       """

        self.api_client = api_client
        self.contract = contract
        self.time = time
        self.future_info = pd.read_csv('future_information.csv')
        # 交易所日历
        self.calendar = pd.read_csv('calendar.csv').rename(
            columns={'TRADE_DAYS': 'trade_days', 'S_INFO_EXCHMARKET': 'exchmarket'})
        self.calendar['trade_days'] = pd.to_datetime(self.calendar['trade_days'], format='%Y%m%d').dt.date
        self.logger = get_logger_snapshoot(name="get_wap", debug=False)
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
            previous_days = exchange_rows[exchange_rows['trade_days'] <= query_date].tail(1)
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

        if day_quote is None or len(day_quote) == 0:
            self.logger.warning(
                f"不存在 {start_date} 到 {end_date} 的数据 contract={contract}, trading_date={query_date}")
            print('day_quote is None')
            return None

        # 保留特定的列
        required_columns = ['symbol',  'datetime', 'last_prc',
                            'volume', 'turnover',  'trading_date', 'ask_prc1','ask_vol1','bid_prc1','bid_vol1']

        missing_columns = [col for col in required_columns if col not in day_quote.columns]

        # 如果关键字段缺失，打日志+跳过
        if any(col in ['symbol', 'datetime', 'volume', 'turnover', 'last_prc','ask_prc1','ask_vol1','bid_prc1','bid_vol1' ] for col in
               missing_columns):
            self.logger.warning(
                f"数据字段有缺失 contract={contract}, trading_date={query_date}, 缺失字段={missing_columns}")
            return None

        day_quote['datetime'] = pd.to_datetime(day_quote['datetime'])
        day_quote['time'] = day_quote['datetime'].dt.strftime('%H%M%S%f').str.slice(0, 9)
        day_quote['date'] = day_quote['datetime'].dt.strftime('%Y%m%d')

        day_quote = day_quote[required_columns]

        # 使用数据清洗程序quote_wash清洗
        data_processor = DataProcessor(future_index=future_code, debug=True)
        day_quote = data_processor.process(day_quote, future_code)

        day_quote['time'] = day_quote['time'].astype(str)
        day_quote['date'] = day_quote['date'].astype(str)
        day_quote['trading_date'] = day_quote['trading_date'].astype(str)

        return day_quote


    def wap( day_quote, contract, start_datetime_str, *lengths):
        """
        计算指定时间段内的时间加权平均价格（TWAP）和成交量加权平均价格（VWAP）。

        :param date: 日期（格式：YYYY-MM-DD）
        :param start_time: 开始时间（格式：HHMMSSfff）
        :param lengths: 时间长度（格式：HH:MM:SS:SSS）

        :return: 包含 TWAP 和 VWAP 的 DataFrame
        """
        # print(day_quote)
        # print(date)
        # print(start_time)
        result = pd.DataFrame(columns=['contract', 'date', 'start_time'])
        #
        # start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%d %H:%M:%S')
        start_datetime = start_datetime_str
        date = start_datetime.strftime('%Y-%m-%d')

        #
        # start_time_str = start_time[:2] + ':' + start_time[2:4] + ':' + start_time[4:6] + '.' + start_time[6:]
        #
        # start_datetime_str = f"{date} {start_time_str}"
        # print(start_datetime_str)
        # start_datetime = datetime.strptime(start_datetime_str, '%Y%m%d %H:%M:%S.%f')
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
            # length_desc = f"{hours}h{minutes}m{seconds}s{milliseconds}f"


            # current_date = start_date

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

                    # 创建临时行并且添加结果
                # temp_row = pd.Series({'date': date,
                #                       'start_time': start_datetime.strftime('%H%M%S%f'),
                #                       'datetime': start_datetime,
                #                       f'{length_desc}_twap_{"pre_" if is_backward else "post_"}prc': twap,
                #                       f'{length_desc}_vwap_{"pre_" if is_backward else "post_"}prc': vwap})
                result.at[0, 'contract'] = contract
                result.at[0, 'date'] = date
                result.at[0, 'start_time'] = str(start_datetime)
                result.at[0, 'datetime'] = start_datetime
                result[f'{length_desc}_twap_{"pre_" if is_backward else "post_"}prc'] = twap
                result[f'{length_desc}_vwap_{"pre_" if is_backward else "post_"}prc'] = vwap

                # results = pd.concat([result, temp_row.to_frame().T], ignore_index=True)


        return result

if __name__ == '__main__':

    # day_quote = pd.read_csv('check2.csv')
    # # date = '2013-11-22'
    # start_day_time = '2013-11-22 14:30:00'
    # lengths = ['-00:01:00:000', '-00:03:00:000', '-00:05:00:000', '00:01:00:000', '00:03:00:000', '00:05:00:000']
    # result = get_wap.wap(day_quote, start_day_time,*lengths)
    # print(result)
    # result.to_csv('wap.csv')
    maincontract_folder = "/nas92/temporary/Steiner/data_wash/linux_so/py311/maincontract2/"
    results_folder = "/nas92/temporary/Steiner/data_wash/linux_so/py311/wap_results/"

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

            snapshoot = get_wap(api_client,  contract, " ")

            # 对每个日期时间进行查询
            for query_time in date_time_list:
                print(query_time)

                snapshoot.time = query_time
                quote_data = snapshoot.get_quote_data()
                # print(quote_data)
                if quote_data is None:
                    continue
                lengths = ['-00:01:00:000', '-00:03:00:000', '-00:05:00:000', '00:01:00:000', '00:03:00:000',
                           '00:05:00:000']
                result = get_wap.wap(quote_data, contract, datetime.strptime(query_time, '%Y-%m-%d %H:%M:%S'),  *lengths)
                # print(result)
                results.append(result)
                query_counter += 1
                if query_counter % 100 == 0:
                    if results:
                        combined_results = pd.concat(results, ignore_index=True)
                        sorted_results = combined_results.sort_values(by=['start_time']).reset_index(drop=True)
                        output_file = os.path.join(results_folder, f"{os.path.splitext(file_name)[0]}_results.csv")
                        sorted_results.to_csv(output_file, index=False)
        if results:
            combined_results = pd.concat(results, ignore_index=True)
            sorted_results = combined_results.sort_values(by=['start_time']).reset_index(drop=True)
            output_file = os.path.join(results_folder, f"{os.path.splitext(file_name)[0]}_results.csv")
            sorted_results.to_csv(output_file, index=False)

