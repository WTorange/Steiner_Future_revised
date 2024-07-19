import os
import pandas as pd
import numpy as np
from datetime import datetime
from logbook import Logger
from Common.logger import get_logger
import logging
from datetime import datetime, timedelta

def get_logger_quote_wash(name, debug=False):

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

        if 'last_prc' in day_quote.columns and 'open_interest' in day_quote.columns:
            day_quote = day_quote[(day_quote['last_prc'] != 0) & (day_quote['open_interest'] != 0)]
        # if missing_columns:
        #     return None

        # 新建time列和date列保持格式统一
        day_quote['datetime'] = pd.to_datetime(day_quote['datetime'])
        # if 'date' in missing_columns or 'time' in missing_columns:
        #     day_quote['datetime'] = pd.to_datetime(day_quote['datetime'])
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
        data.loc[:, 'datetime'] = pd.to_datetime(data['datetime'])
        data.loc[:, 'date'] = data['datetime'].dt.date
        data = data[data['datetime'].dt.year >= 2009]

        resampled_data = pd.DataFrame()
        # 循环处理每一个日期
        for date in data['date'].unique():

            opening_hours_str = self.opening_hours_df.loc[self.opening_hours_df['code'] == contract_code, 'hours'].values[0]
            periods = opening_hours_str.split()

            date_data = data[data['date'] == date].copy()
            original_datetime = date_data['datetime'].copy()
            date_data['resample_time'] = original_datetime

            for period in periods:
                start_time_str, end_time_str = period.split('-')
                start_datetime = datetime.strptime(date.strftime('%Y-%m-%d') + start_time_str, '%Y-%m-%d%H:%M')
                end_datetime = datetime.strptime(date.strftime('%Y-%m-%d') + end_time_str, '%Y-%m-%d%H:%M')

                if end_datetime < start_datetime:
                    end_datetime += timedelta(days=1)

                all_times = pd.date_range(start=start_datetime, end=end_datetime, freq=freq)

                date_data = date_data.drop_duplicates(subset='resample_time')

                resampled = date_data.set_index('resample_time').reindex(all_times, method='pad')

                resampled['date'] = date
                resampled['resample_time'] = resampled.index
                resampled = resampled[resampled['last_prc'].notna()]
                resampled_data = pd.concat([resampled_data, resampled], ignore_index=True)
        return resampled_data

    def process(self, day_quote,future_index):
        washed_quote = self.quote_wash(day_quote,future_index)
        if washed_quote is not None:
            resampled_quote = self.resample_data(washed_quote,future_index)
            return resampled_quote
        return None


if __name__ == '__main__':
    # opening_hours_df = pd.read_csv('future_information.csv')

    future_index = 'RU'
    processor = DataProcessor(future_index)

    # 根据contract_code找到对应的开盘时间
    day_quote = pd.read_csv('test1.csv')
    cleaned_quote = processor.process(day_quote, future_index)
    cleaned_quote.to_csv("check.csv")

