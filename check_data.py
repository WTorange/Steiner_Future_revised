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

class get_snapshoot(ApiClient):
    def __init__(self, future_index: str, time: str):
        super().__init__("zlt_read", "zlt_read", "./config/api.json", get_logger(name="ApiClientLog", debug=False))
        if not self.init():
            raise Exception("初始化失败")

        self.future_index = future_index
        self.time = time
        self.future_info = pd.read_csv('future_information.csv')
        self.calendar = pd.read_csv('calendar.csv').rename(columns={'TRADE_DAYS': 'trade_days', 'S_INFO_EXCHMARKET': 'exchmarket'})
        # self.calendar['trade_days'] = pd.to_datetime(self.calendar['trade_days'], format='%Y%m%d')
        self.calendar['trade_days'] = pd.to_datetime(self.calendar['trade_days'], format='%Y%m%d').dt.date
        self.logger = get_logger(name="get_snapshoot", debug=False)


    def extract_contract(self, future_index):
        # 提取合约开头的一个或两个大写字母
        match = re.match(r'^([A-Z]{1,2})', future_index)
        return match.group(1) if match else future_index

    def get_quote_data(self):
        # 根据输入的时间time，取yyyy-mm-dd，命名为day_time.
        # day_time = self.time.split(" ")[0]

        entries = [self.future_index]
        entry_type = 'quote'

        contract_code = self.extract_contract(self.future_index)

        # 获取期货品种信息
        future_details = self.future_info[self.future_info['code'] == contract_code]
        if future_details.empty:
            raise Exception("未找到期货品种信息")
        # if not self.init():
        #     raise Exception("初始化失败")

        # 判断是否有夜盘
        has_night_session = future_details.iloc[0]['daynight'] == 1

        query_datetime = datetime.strptime(self.time, '%Y-%m-%d %H:%M:%S')
        query_date = query_datetime.date()
        start_date = query_date
        end_date = query_date

        start_time = start_date.strftime('%Y-%m-%d')
        end_time = end_date.strftime('%Y-%m-%d')

        # 获取当天的quote数据
        day_quote = self.query_history(entries, entry_type, start_time, end_time)
        day_quote = day_quote[0]

        # 保留特定的列
        required_columns =['symbol', 'date', 'time', 'acc_open', 'acc_high', 'acc_low', 'last_prc',
                           'volume', 'turnover','ask_prc1', 'bid_prc1','trading_date']
        day_quote = day_quote[required_columns]

        day_quote['time'] = day_quote['time'].astype(str)
        day_quote['date'] = day_quote['date'].astype(str)
        day_quote['trading_date'] = day_quote['trading_date'].astype(str)
        # 转换时间格式并增加'daynight'列

        return day_quote

if __name__ == '__main__':

    contracts_df = pd.read_csv(
        r"C:\Users\maki\Desktop\quantchina\Futures-main\data_wash\maincontract\RU_20091201_20260624.csv")
    contracts_df = contracts_df
    calendar = pd.read_csv('calendar.csv').rename(
        columns={'TRADE_DAYS': 'trade_days', 'S_INFO_EXCHMARKET': 'exchmarket'})

    calendar['trade_days'] = calendar['trade_days'].astype(str)
    start_time = t.time()
    # times = date_time_list

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

        filtered_calendar = exchange_rows[  # 指定日期之后的条件
                                          (exchange_rows['trade_days'] >= start_date) &
                                          (exchange_rows['trade_days'] <= end_date)]
        reversed_filtered_calendar = filtered_calendar.iloc[::-1]
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
            result = snapshoot.certain_snapshoot(quote_data, query_time)
            print(result)
            results.append(result)

    end_time = t.time()
    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time:.2f} 秒")
    combined_results = pd.concat(results, ignore_index=True)
    sorted_results = combined_results.sort_values(by=['query_time'])
    sorted_results.reset_index()
    sorted_results.to_csv('RU.csv')
