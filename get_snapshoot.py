import os
import pandas as pd
import numpy as np
from datetime import datetime
import re

from ApiClient import ApiClient
from logbook import Logger
from Common.logger import get_logger

# 输出： 日夜，品种，合约，日期，时刻，当天截止最高价，开盘，截止最低价，收盘。
class get_snapshoot(ApiClient):
    def __init__(self, future_index: str, time: str):
        super().__init__("zlt_read", "zlt_read", "./config/api.json", get_logger(name="ApiClientLog", debug=False))
        if not self.init():
            raise Exception("初始化失败")
        self.future_index = future_index  # 例 T2403CFE
        self.time = time

    def get_quote_data(self):
        # 根据输入的时间time，取yyyy-mm-dd，命名为day_time.
        day_time = self.time.split(" ")[0]
        start_time = day_time
        end_time = day_time
        entries = [self.future_index]
        entry_type = 'quote'

        # if not self.init():
        #     raise Exception("初始化失败")

        # 获取当天的quote数据
        day_quote = self.query_history(entries, entry_type, start_time, end_time)
        day_quote = day_quote[0]

        # 保留特定的列
        required_columns = ['symbol', 'time', 'acc_open', 'acc_high', 'acc_low', 'last_prc']
        day_quote = day_quote[required_columns]
        day_quote['time'] = day_quote['time'].astype(str)

        # 转换时间格式并增加'daynight'列
        def convert_time(row):
            time_str = row['time']
            time_obj = datetime.strptime(time_str, '%H%M%S%f')
            if time_obj >= datetime.strptime("090000000000", '%H%M%S%f') and time_obj <= datetime.strptime(
                    "210000000000", '%H%M%S%f'):
                return "day"
            else:
                return "night"

        day_quote['daynight'] = day_quote.apply(convert_time, axis=1)

        return day_quote

    def extract_contract(self, future_index):
        # 提取合约开头的一个或两个大写字母
        match = re.match(r'^([A-Z]{1,2})', future_index)
        return match.group(1) if match else future_index

    def certain_snapshoot(self, df, time):
        # 将传入的时间字符串转换为datetime对象
        query_time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        df['datetime'] = df['time'].apply(lambda x: datetime.strptime(x, '%H%M%S%f').replace(
            year=query_time.year, month=query_time.month, day=query_time.day))

        # 找到查询时间之前的最后一个tick
        df_before_query = df[df['datetime'] <= query_time]
        if df_before_query.empty:
            return None

        last_tick = df_before_query.iloc[-1]

        result = {
            'daynight': last_tick['daynight'],
            'future_index': self.extract_contract(self.future_index),
            'contract': self.future_index,
            'date': query_time.strftime('%Y-%m-%d'),
            'time': last_tick['time'],
            'high': df_before_query['acc_high'].max(),
            'open': df_before_query['acc_open'].iloc[0],
            'low': df_before_query['acc_low'].min(),
            'close': last_tick['last_prc']
        }

        result_df = pd.DataFrame([result])
        print(result_df)
        csv_filename = f'snapshoot_result_{self.future_index}_{query_time.strftime("%Y%m%d_%H%M%S")}.csv'
        result_df.to_csv(csv_filename, index=False)
        print(f'Result saved to {csv_filename}')
        return result_df


# 测试代码
if __name__ == '__main__':
    future_index = 'RU2305.SHF'
    time = '2023-01-11 16:30:24'

    snapshoot = get_snapshoot(future_index, time)
    quote_data = snapshoot.get_quote_data()
    test_snapshoot = snapshoot.certain_snapshoot(quote_data, time)

# 上海期货交易所（SHF）
shfe_dict = {
    "AL": "铝",
    "CU": "铜",
    "BC": "铜",
    "ZN": "锌",
    "PB": "铅",
    "NI": "镍",
    "SN": "锡",
    "AU": "黄金",
    "AG": "白银",
    "RB": "螺纹钢",
    "HC": "热轧卷板",
    "SS": "不锈钢",
    "SC": "原油",
    "FU": "燃料油",
    "BU": "石油沥青",
    "RU": "天然橡胶",
    "SP": "纸浆",
    "NR": "20号胶"
}

# 大连商品交易所（DCE）
dce_dict = {
    "A": "黄大豆1号",
    "B": "黄大豆2号",
    "C": "玉米",
    "CS": "淀粉",
    "M": "豆粕",
    "Y": "豆油",
    "P": "棕榈油",
    "L": "乙烯",
    "V": "聚氯乙烯",
    "PP": "聚丙烯",
    "J": "焦炭",
    "JM": "焦煤",
    "I": "铁矿石",
    "JD": "鸡蛋",
    "EB": "苯乙烯",
    "PG": "液化石油气"
}

# 郑州商品交易所（CZCE）
czce_dict = {
    "SR": "白糖",
    "CF": "棉花",
    "ZC": "动力煤",
    "FG": "玻璃",
    "TA": "PTA",
    "MA": "甲醇",
    "RM": "菜粕",
    "OI": "菜油",
    "RS": "油菜籽",
    "CY": "棉纱",
    "AP": "苹果",
    "CJ": "红枣",
    "UR": "尿素",
    "SA": "纯碱",
    "SF": "硅铁",
    "SM": "锰硅"
}

# 中国金融期货交易所（CFFEX）
cffex_dict = {
    "IF": "沪深300指数",
    "IH": "上证50指数",
    "IC": "中证500指数",
    "IM": "中证1000指数",
    "T": "10年期国债",
    "TF": "5年期国债",
    "TS": "2年期国债"
}

# 汇总所有交易所的字典
futures_dict = {
    "SHFE": shfe_dict,
    "DCE": dce_dict,
    "CZCE": czce_dict,
    "CFFEX": cffex_dict
}

# # 打印结果
# for exchange, codes in futures_dict.items():
#     print(f"{exchange}:")
#     for code, name in codes.items():
#         print(f"  {code}: {name}")