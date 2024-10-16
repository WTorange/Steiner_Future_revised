# 价值，基差和持仓量
# 五年前的现货价格(spot_price)定义为账面价值，除以当前现货价格，定义为价值因子，将4.5-5.5年之间现货价格均值作为账面价值

# 先针对symbol_list，读取nearby中的parquet文件，合并，针对每一个日期，计算过去4.5-5.5年间的平均close作为之前的现货价格，除以上一天的close，作为当天的因子值
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import math

def correct_czc_code(contract, trading_date):
    # 如果期货合约以 '.CZC' 结尾
    if contract.endswith('.CZC'):
        # 匹配期货代码中的字母和数字部分
        match = re.match(r'^([A-Z]+)(\d+)\.CZC', contract)
        if match:
            letters, numbers = match.groups()
            # 提取查询时间的年份
            if isinstance(trading_date, int):
                # 假设 trading_date 是 yyyymmdd 格式的整数
                string_date = str(trading_date)
                year = int(string_date[:4])
            elif isinstance(trading_date, str):
                # 假设 trading_date 是 yyyymmdd 格式的字符串
                year = int(trading_date[:4])
            else:
                # 假设 trading_date 是 datetime 对象
                year = trading_date.strftime('%Y')
                year = int(year)

            # 修正数字部分，在第一位加上年份
            if len(numbers) == 3 and numbers[0] != '9' and year >= 2019:
                corrected_numbers = '2' + numbers
            elif len(numbers) == 3:
                corrected_numbers = '1' + numbers
            else:
                corrected_numbers = numbers
            return f"{letters}{corrected_numbers}.CZC"
    return contract

def value_factor(symbol: str, source_folder: str, output_folder: str):
    # 第一步：读取并合并期货数据
    print(symbol)
    all_files = []

    # 遍历source_folder中的所有日期文件夹
    for root, dirs, files in os.walk(source_folder):
        # 找到名称为nearby_contract的文件夹
        if 'nearby_contract' in root:
            for file in files:
                # 找到以symbol+数字开头的文件
                if file.startswith(symbol) and file[len(symbol)].isdigit():
                    # 解析文件名中的日期
                    try:
                        date_str = file.split('_')[1][:8]
                        date = pd.to_datetime(date_str, format='%Y%m%d')
                        file_path = os.path.join(root, file)
                        all_files.append((file_path, date))
                    except:
                        continue

    # 按照日期排序
    all_files.sort(key=lambda x: x[1])

    # 读取所有文件并合并
    data_frames = []
    for file_path, _ in all_files:
        print(file_path)
        df = pd.read_parquet(file_path)  # 假设文件格式为CSV，你可以根据实际格式调整
        if df.empty:
            continue
        else:
            data_frames.append(df)
    if not data_frames:
        print(f"No valid data found for symbol: {symbol}, skipping...")
        return
    # 合并所有数据
    quote_data = pd.concat(data_frames, ignore_index=True)

    # 第二步：打上day night标签
    # 确保resample_time是datetime类型
    quote_data['resample_time'] = pd.to_datetime(quote_data['resample_time'])

    # 打上day和night标签
    quote_data['daynight'] = np.where(
        quote_data['resample_time'].dt.time < pd.Timestamp('08:00:00').time(),
        'night',
        np.where(quote_data['resample_time'].dt.time < pd.Timestamp('20:00:00').time(), 'day', 'night')
    )



    def resample(data):

        # 确保trading_date和resample_time列是datetime类型
        data['trading_date'] = pd.to_datetime(data['trading_date'])
        data['resample_time'] = pd.to_datetime(data['resample_time'])

        # 创建一个新的DataFrame来存储重采样的结果
        resampled_data = data.groupby(['trading_date', 'daynight']).agg(
            open=('last_prc', 'first'),
            close=('last_prc', 'last'),
            volume=('volume', 'last'),
            turnover=('turnover', 'last'),
            open_interest=('open_interest', 'last')
        ).reset_index()
        return  resampled_data
    resampled_data = resample(quote_data)

    # 保存重采样后的数据
    output_file_path = os.path.join(output_folder, f"{symbol}_nearby.csv")
    resampled_data.to_csv(output_file_path, index=False)


def calculate_value_daily_factor(symbol,folder_path, save_folder):
    file_found = False
    for filename in os.listdir(folder_path):
        if filename.startswith(f"{symbol}_") and filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            file_found = True
            break
    if not file_found:
        return f"No CSV file found for symbol: {symbol}"
    df = pd.read_csv(file_path)

    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
    df = df[(df['S_DQ_CLOSE'] != 0) & (df['S_DQ_CLOSE'].notna())]

    df['S_INFO_WINDCODE'] = df.apply(lambda row: correct_czc_code(row['S_INFO_WINDCODE'], row['TRADE_DT']), axis=1)
    df = df.sort_values(by=['S_INFO_WINDCODE', 'TRADE_DT'])
    df['prev_close'] = df.groupby('S_INFO_WINDCODE')['S_DQ_CLOSE'].shift(1)
    df['contract_num'] = df['S_INFO_WINDCODE'].str.extract(r'(\d+)').astype(int)
    nearby_contracts = df.loc[df.groupby('TRADE_DT')['contract_num'].idxmin()][
        ['TRADE_DT', 'S_INFO_WINDCODE', 'S_DQ_CLOSE', 'prev_close']]

    def calculate_value(row):
        # 获取当前行的交易日期
        current_date = row['TRADE_DT']

        # 计算前4.5年到5.5年范围内的日期
        start_date = current_date - timedelta(days=int(5.5 * 365.25))  # 前5.5年
        end_date = current_date - timedelta(days=int(4.5 * 365.25))

        # 在这个日期范围内筛选数据
        date_range_data = nearby_contracts[
            (nearby_contracts['TRADE_DT'] >= start_date) & (nearby_contracts['TRADE_DT'] <= end_date)]

        # 计算时间跨度
        if not date_range_data.empty and pd.notna(row['prev_close']):
            time_span = (date_range_data['TRADE_DT'].max() - date_range_data['TRADE_DT'].min()).days

            if time_span >= 30 * 11:  # 至少 11 个月的数据
                mean_close = date_range_data['S_DQ_CLOSE'].mean()
                return math.log(mean_close / row['prev_close'])

        return None
    nearby_contracts['value_daily'] = nearby_contracts.apply(calculate_value, axis=1)

    # 选择需要的列并保存为新的 CSV 文件
    result_df = nearby_contracts[['TRADE_DT',  'value_daily']]
    result_df = result_df.rename(columns={'value_daily': f'{symbol}_value_daily','TRADE_DT':'trading_date'})
    result_df.dropna(subset=[f'{symbol}_value_daily'], inplace=True)  # 删除 value_daynight 为空的行

    # 保存结果到 CSV 文件
    output_file = os.path.join(save_folder, f"{symbol}_value_daily_factor.csv")
    result_df.to_csv(output_file, index=False)
    print(output_file)




def value_daily_factor(symbol,folder_path,save_folder):
    df = find_csv(symbol,folder_path)

    value_factor_df =calculate_value_daily_factor(symbol, df)

    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f'{symbol}_value_daily_factor.csv')

    value_factor_df.to_csv(save_path, index=False)

    print(f"Value factor data saved to {save_path}")

def value_daynight(symbol, source_folder, save_folder):
    file_list = [f for f in os.listdir(source_folder) if f.startswith(symbol + "_") and f.endswith('.csv')]

    if not file_list:
        print(f"No files found for {symbol}")
        return

    # 读取第一个匹配的文件
    file_path = os.path.join(source_folder, file_list[0])
    data = pd.read_csv(file_path)

    data['prev_close'] = data['close'].shift(1)

    # 转换 trading_date 为 datetime 格式，方便计算时间差
    data['trading_date'] = pd.to_datetime(data['trading_date'], format='%Y-%m-%d')

    def calculate_value_daynight(row):
        # 当前行的 trading_date
        current_date = row['trading_date']

        # 计算前4.5年到5.5年的日期范围
        start_date = current_date - timedelta(days=int(5.5 * 365.25))  # 前5.5年
        end_date = current_date - timedelta(days=int(4.5 * 365.25))

        # 获取日期范围内的所有数据行
        date_range_data = data[(data['trading_date'] >= start_date) & (data['trading_date'] <= end_date)]

        if not date_range_data.empty and pd.notna(row['prev_close']):
            time_span = (date_range_data['trading_date'].max() - date_range_data['trading_date'].min()).days

            if time_span >= 30 * 11 and pd.notna(row['prev_close']):

                mean_close = date_range_data['close'].mean()
                return math.log(mean_close / row['prev_close'])


        return None  # 如果日期范围数据为空或 prev_close 为空，返回 None

        # 计算 value_daynight 因子值并创建新列

    data['value_daynight'] = data.apply(calculate_value_daynight, axis=1)

    # 选择需要的列并保存为新的 CSV 文件
    result_df = data[['trading_date', 'daynight', 'value_daynight']]
    result_df = result_df.rename(columns={'value_daynight': f'{symbol}_value_daynight'})
    result_df.dropna(subset=[f'{symbol}_value_daynight'], inplace=True)  # 删除 value_daynight 为空的行

    # 保存结果到 CSV 文件
    output_file = os.path.join(save_folder, f"{symbol}_value_daynight_factor.csv")
    result_df.to_csv(output_file, index=False)

    print(f"Saved value_daynight factor to {output_file}")


def cs_value_factor():
    # Step 1: 读取并合并所有包含“5days”的CSV文件

    input_folder = f'/nas92/data/future/factor/value/value_daily'
    all_files = glob.glob(os.path.join(input_folder, "*.csv"))
    output_folder1 = "/nas92/data/future/factor/value/cs_value/rank_n_weight"
    output_folder2 = f"/nas92/data/future/factor/value/cs_value"
    # 初始化空的DataFrame以存储合并后的数据
    cs_df = pd.DataFrame()

    # 读取每个CSV文件并合并
    for file in all_files:
        df = pd.read_csv(file)
        df['trading_date'] = pd.to_datetime(df['trading_date'], format='%Y-%m-%d')
        df.set_index('trading_date', inplace=True)
        # 重命名列名为第一个下划线之前的字母
        new_col_name = df.columns[0].split('_')[0]
        df.rename(columns={df.columns[0]: new_col_name}, inplace=True)

        cs_df = pd.concat([cs_df, df], axis=1)
    # 重置索引并将trading_date作为第一列
    cs_df.reset_index(inplace=True)
    # 对每一行（每个交易日）计算排名
    rank_df = cs_df.set_index('trading_date').rank(axis=1, method='min', ascending=True)

    # 计算权重
    weight_df = pd.DataFrame(index=rank_df.index, columns=rank_df.columns)

    for date in rank_df.index:
        valid_ranks = rank_df.loc[date].dropna()
        s_t = len(valid_ranks)
        if s_t > 0:  # 确保有有效的排名值
            # 计算未调整的权重
            weights = valid_ranks.apply(lambda rank: rank / ((1 + s_t) / 2) - 1)

            weight_df.loc[date, weights.index] = weights
    print(weight_df)
    # 将trading_date列重置为第一列
    rank_df.reset_index(inplace=True)
    weight_df.reset_index(inplace=True)
    #
    for symbol in weight_df.columns:
        if symbol != 'trading_date':  # 忽略 'trading_date' 列
            symbol_df = weight_df[['trading_date', symbol]].dropna()  # 删除NaN值的行
            output_file = os.path.join(output_folder2, f"{symbol}_cs_value.csv")
            symbol_df.to_csv(output_file, index=False)
            print(f"已保存: {output_file}")

    # na_ratio = rank_df.isna().mean()  # 计算每列的空值占比
    # na_ratio_sorted = na_ratio.sort_values(ascending=False).head(10)  # 获取占比前五的列
    # print("Top 5 columns with highest NaN ratio:")
    # for col, ratio in na_ratio_sorted.items():
    #     print(f"{col}: {ratio:.2%}")

    os.makedirs(output_folder1, exist_ok=True)

    # 保存结果
    # 保存 rank_df 和 weight_df 到文件
    rank_output_file = os.path.join(output_folder1, f"rank_cs_value.csv")
    weight_output_file = os.path.join(output_folder1, f"weight_cs_value.csv")

    rank_df.to_csv(rank_output_file, index=False)
    weight_df.to_csv(weight_output_file, index=False)
    print(f"Rank and weight data saved to {output_folder1}")
if __name__ =='__main__':

    save_folder1 = "/nas92/data/future/factor/value/value_daynight/"
    source_folder1 = "/nas92/data/future/trade_buffer/buffer_day_nearby/"
    save_folder2 = "/nas92/data/future/factor/value/value_daily/"
    source_folder2 = "/nas92/data/future/daybar/"



    save_folder = "/nas92/data/future/trade_buffer/buffer_day_nearby/"
    source_folder = "/nas92/data/future/quote"
# #
    # symbol_list = ["A", "AG", "AL", "AO", "AP", "AU", "B", "BB", "BC", "BR", "BU", "C", "CF", 'CJ', 'CS', 'CU', 'CY', 'EB',
    #                'EC', 'EG', 'ER', 'FB', 'FG', 'FU', 'HC', 'I', 'IC',
    #                'IF', 'IH', 'IM', 'J', 'JD', 'JM', 'JR', 'L', 'LH', 'LU', 'M', 'MA', 'ME', 'NI', 'NR', 'OI', 'P', 'PB',
    #                'PF', 'PG', 'PK', 'PM', 'PP', 'PX', 'RB', 'RI', 'RM', 'RO',
    #                'RR', 'RS', 'RU', 'SA', 'SC', 'SF', 'SH', 'SM', 'SN', 'SP', 'SR', 'SS', 'T', 'TA', 'TC', 'TF', 'TL',
    #                'TS', 'UR', 'V', 'WH', 'WR', 'WS', 'WT', 'Y', 'ZC', 'ZN']

    symbol_list = ['A']
    # for symbol in symbol_list:
    #     value_factor(symbol, source_folder, save_folder)

    cs_value_factor()


