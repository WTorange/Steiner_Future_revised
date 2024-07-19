import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\maki\Desktop\quantchina\Futures-main\data_wash\2009.csv")
df = pd.DataFrame(data)
df = df['datetime']
# 开盘时间段的定义
opening_times = '9:00-11:30 13:30-15:00 21:00-23:00'
opening_times_list = opening_times.split()

# 定义一个函数将时间转换成目标格式
def convert_to_target_format(time_str):
    minutes, seconds = time_str.split(':')
    minutes = int(minutes)
    seconds = float(seconds)
    formatted_time = f"{minutes:02}{int(seconds * 1000):03}"
    return formatted_time


# 定义一个函数来匹配开盘时间段并生成新的 'time' 列
def create_new_time_column(row):
    time_str = row['datetime']
    for opening_time in opening_times_list:
        start_time, end_time = opening_time.split('-')
        start_hour, start_minute = map(int, start_time.split(':'))
        end_hour, end_minute = map(int, end_time.split(':'))

        # 解析时间并进行适当的调整
        minutes, seconds = time_str.split(':')
        minutes = int(minutes)
        seconds = float(seconds)

        # 确定小时数和分钟数范围
        total_start_minutes = start_hour * 60 + start_minute
        total_end_minutes = end_hour * 60 + end_minute
        total_minutes = minutes + start_hour * 60

        if total_start_minutes <= total_minutes <= total_end_minutes:
            # 匹配分钟数以 00、15、30 开头的时间
            if minutes % 15 == 0:
                return f"{start_hour:02}{convert_to_target_format(time_str)}"

    return None


# 创建新的 'time' 列
df['time'] = df.apply(create_new_time_column)

# 去除包含 None 的行
df = df.dropna(subset=['time'])

print(df)