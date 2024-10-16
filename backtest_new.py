import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpld3
import plotly.graph_objs as go
import plotly.io as pio
from io import BytesIO
import pdfkit
from connect_wind import ConnectDatabase

from plotly.subplots import make_subplots
import re
# 调整小数位数
# 加上index
# 写动量代码，调用回测
# 补充手续费字典

comm_dict = {
    # 类型，开仓，平仓，平今手续费
    'I': (1, 0.00012, 0.00006, 0.00012), 'AG': (1, 0.00005, 0.00005, 0.00005),
    'IF': (1, 0.000023, 0.000023, 0.00023), 'IH': (1, 0.000023, 0.000023, 0.00023),
    'IC': (1, 0.000023, 0.000023, 0.00023), 'IM': (1, 0.000023, 0.000023, 0.00023), 'MA': (1, 0.0001, 0.0001, 0.0001),
    'SA': (1, 0.0002, 0.0002, 0.0002), 'SH': (1, 0.0001, 0.0001, 0), 'PX': (1, 0.0001, 0.0001, 0),
    'UR': (1, 0.0001, 0.0001, 0.0001), 'BB': (1, 0.0001, 0.0001, 0.0001), 'FB': (1, 0.0001, 0.0001, 0.0001),
    'J': (1, 0.0001, 0.0001, 0.00014), 'JD': (1, 0.00015, 0.00015, 0.00015), 'JM': (1, 0.0001, 0.0001, 0.0003),
    'LH': (1, 0.0001, 0.0001, 0.0002), 'SI': (1, 0.0001, 0.0001, 0.0001), 'LC': (1, 0.00008, 0.00008, 0.00008),
    'BC': (1, 0.0001, 0.0001, 0), 'LU': (1, 0.0001, 0.0001, 0.0001), 'NR': (1, 0.0002, 0.0002, 0),
    'EC': (1, 0.00005, 0.00005, 0.00005), 'AO': (1, 0.0001, 0.0001, 0.0001), 'BU': (1, 0.00005, 0.00005, 0),
    'CU': (1, 0.00005, 0.00005, 0.0001), 'FU': (1, 0.00005, 0.00005, 0.00005), 'HC': (1, 0.0001, 0.0001, 0.0001),
    'PB': (1, 0.00004, 0.00004, 0.00004), 'RB': (1, 0.00002, 0.00002, 0.00002), 'BR': (1, 0.00002, 0.00002, 0.00002),
    'SP': (1, 0.00005, 0.00005, 0), 'WR': (1, 0.00004, 0.00004, 0.00004),

    'TL': (2, 3, 3, 0), 'T': (2, 3, 3, 0), 'TF': (2, 3, 3, 0),
    'TS': (2, 3, 3, 0), 'JR': (2, 3, 3, 3), 'CJ': (2, 3, 3, 3),
    'AP': (2, 5, 5, 20), 'CF': (2, 4.3, 4.3, 0), 'CY': (2, 4, 4, 0),
    'FG': (2, 6, 6, 6), 'LR': (2, 3, 3, 3), 'OI': (2, 2, 2, 2),
    'PF': (2, 3, 3, 3), 'PK': (2, 4, 4, 4), 'PM': (2, 30, 30, 30),
    'RI': (2, 2.5, 2.5, 2.5), 'RM': (2, 1.5, 1.5, 1.5), 'RS': (2, 2, 2, 2),
    'SF': (2, 3, 3, 0), 'SM': (2, 3, 3, 0), 'SR': (2, 3, 3, 0),
    'TA': (2, 3, 3, 0), 'WH': (2, 30, 30, 30), 'ZC': (2, 150, 150, 150),
    'A': (2, 2, 2, 2), 'B': (2, 1, 1, 1), 'C': (2, 1.2, 1.2, 1.2),
    'CS': (2, 1.5, 1.5, 1.5), 'EB': (2, 3, 3, 3), 'EG': (2, 3, 3, 3),
    'L': (2, 1, 1, 1), 'M': (2, 1.5, 1.5, 1.5), 'P': (2, 2.5, 2.5, 2.5),
    'PG': (2, 6, 6, 6), 'PP': (2, 1, 1, 1), 'RR': (2, 1, 1, 1),
    'V': (2, 1, 1, 1), 'Y': (2, 2.5, 2.5, 2.5), 'SC': (2, 20, 20, 20),
    'AL': (2, 3, 3, 3), 'AU': (2, 2, 2, 2), 'NI': (2, 3, 3, 3),
    'RU': (2, 3, 3, 3), 'SN': (2, 3, 3, 3), 'SS': (2, 2, 2, 0), 'ZN': (2, 3, 3, 0),
'TC': (2, 8, 8, 4),
}

def correct_czc_code(contract, trading_date):
    if contract.endswith('.CZC'):
        # 提取期货代码中的字母和数字部分
        match = re.match(r'^([A-Z]+)(\d+)\.CZC$', contract)
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
class backtest(ConnectDatabase):
    def __init__(self, symbol: str, initial_capital: int, leverage: float, signal_df, factor_name, output_path,
                 start_date=None, end_date=None):

        self.margin_sql = f'''
                       SELECT S_INFO_WINDCODE, MARGINRATIO, TRADE_DT
                       FROM CFUTURESMARGINRATIO
                       WHERE TRADE_DT >= 20100101
                       AND S_INFO_WINDCODE LIKE '{symbol}%'
                       '''
        self.multi_sql = f'''
                             SELECT S_INFO_CODE, S_INFO_PUNIT
                             FROM  CFUTURESCONTPRO
                             WHERE S_INFO_CODE LIKE '{symbol}%'
                             '''

        super().__init__(self.margin_sql)
        self.margin_df = super().get_data()

        self.sql = self.multi_sql
        multi_df = super().get_data()

        multi_df = multi_df[multi_df['S_INFO_CODE'] == symbol]
        if not multi_df.empty:
            self.multi = int(multi_df.iloc[0]['S_INFO_PUNIT'])
            if symbol in ['T', 'TF', 'TS','TL']:
                self.multi = 100
            print(f'{symbol} multi = ', self.multi)
        else:
            raise ValueError(f"No matching S_INFO_CODE found for symbol: {symbol}")
        self.symbol = symbol

        first_row = signal_df.iloc[0].copy()
        first_row['position'] = 0
        signal_df = pd.concat([pd.DataFrame([first_row]), signal_df], ignore_index=True)

        self.df = signal_df
        self.leverage = leverage
        # self.margin = margin_dict[symbol]
        self.comm_dict = comm_dict
        # self.multi = multi_dict[symbol]
        self.future_type = comm_dict[symbol][0]

        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.factor_name = factor_name
        self.output_path = output_path

    def calculate_operation(self):
        """
        输入：包含信号的dataframe,start_date:yyyymmdd,end_date: yyyymmdd.
        根据初始和结束日期选择信号df中的片段，计算每天的operation
        输出：计算每天的operation，新的df
        """
        # 将date列转换为datetime类型
        self.df['date'] = pd.to_datetime(self.df['date'])
        if self.start_date and self.end_date:
            self.df = self.df[(self.df['date'] >= self.start_date) & (self.df['date'] <= self.end_date)]
        # 计算operation列
        self.df['operation'] = self.df['position'].diff().fillna(0).astype(int)
        if not self.df.empty:  # 确保数据框不为空
            self.df.at[self.df.index[0], 'operation'] = self.df.at[self.df.index[0], 'position']
        return self.df

    def determine_trade_contracts(self):
        """
        根据主力合约数据获取每天要交易的合约和换仓
        """
        # 确定每天交易的合约代码
        symbol = self.symbol
        folder_path = '/nas92/temporary/Steiner/data_wash/linux_so/py311/essentialcontract/'
        file_pattern = os.path.join(folder_path, f"{symbol}_*.csv")
        matching_files = glob.glob(file_pattern)
        if len(matching_files) == 0:
            print(f"No matching daybar file found for symbol: {symbol}")
            return None
        elif len(matching_files) > 1:
            print(f"Multiple matching daybar files found for symbol: {symbol}.")
            print("Please ensure there is only one matching file or refine the search criteria.")
            return None
        else:
            # 读取唯一匹配的文件
            contract_file = matching_files[0]
            print(f"Using daybar file: {contract_file}")
            contract = pd.read_csv(contract_file)
        contract['trade_contract'] = ''
        contract['change_contract'] = ''

        # 初始交易合约为第一天的主力合约
        current_contract = contract.at[0, 'main_contract']
        contract.at[0, 'trade_contract'] = current_contract

        for i in range(1, len(contract) - 1):
            if contract.at[i, 'main_contract'] != current_contract and contract.at[
                i - 1, 'main_contract'] != current_contract and contract.at[i - 2, 'main_contract'] != current_contract \
                    and contract.at[i - 3, 'main_contract'] == current_contract:
                    # and contract.at[i, 'main_contract'] == contract.at[i - 1, 'main_contract'] == contract.at[i - 2, 'main_contract']:

                contract.at[i, 'trade_contract'] = current_contract
                contract.at[i, 'change_contract'] = contract.at[i, 'main_contract']
                # 接下来的一天也在换仓期
                current_contract = contract.at[i, 'main_contract']
            elif contract.at[i, 'trade_contract'] == '':
                # 如果不是换仓期，交易前一天的合约
                contract.at[i, 'trade_contract'] = current_contract
        contract['date'] = pd.to_datetime(contract['date'], format='%Y-%m-%d').dt.strftime('%Y%m%d')

        contract = contract[['date', 'trade_contract', 'change_contract']]
        return contract

    def process_trades(self, contract_df):
        """
        输入：包含信号和日期的df,每天要交易和换约的合约代码df
        输出：匹配每天要交易的合约，包括换约
        """
        df = self.df
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        contract_df['date'] = pd.to_datetime(contract_df['date'], format='%Y%m%d')

        # 初始化结果DataFrame
        results = []

        # 合并两个数据框，基于日期进行匹配
        merged_df = pd.merge(df, contract_df, on='date', how='left')

        for _, row in merged_df.iterrows():
            date = row['date']
            daynight = row['daynight']
            position = row['position']
            operation = row['operation']
            trade_contract = row['trade_contract']
            change_contract = row['change_contract']
            if pd.notna(change_contract) and change_contract.strip() != '' and daynight == 'day':
                # if pd.notna(change_contract) and daynight == 'day' and position != 0:
                # 换约操作
                # 平掉原合约
                results.append({
                    'date': date,
                    'daynight': daynight,
                    'position': 0,
                    'operation': -position,
                    'contract': trade_contract,
                    'roll_signal': 1
                })
                # 开新合约
                results.append({
                    'date': date,
                    'daynight': daynight,
                    'position': position,
                    'operation': position,
                    'contract': change_contract,
                    'roll_signal': 1
                })
            else:
                # 非换约操作
                results.append({
                    'date': date,
                    'daynight': daynight,
                    'position': position,
                    'operation': operation,
                    'contract': trade_contract,
                    'roll_signal': 0
                })
        results = pd.DataFrame(results)
        i = 0
        while i < len(results) - 1:
            current_row = results.iloc[i]
            next_row = results.iloc[i + 1]

            # 满足条件：contract 不等于下一行 contract 且 roll_signal == 0
            if (current_row['contract'] != next_row['contract']) and (current_row['roll_signal'] == 0):
                # 在 current_row 和 next_row 之间插入一行
                new_row = {
                    'date': current_row['date'],
                    'daynight': current_row['daynight'],
                    'position': 0,
                    'operation': -current_row['position'],
                    'contract': current_row['contract'],
                    'roll_signal': 1
                }

                # 插入新行，保持顺序
                results = pd.concat(
                    [results.iloc[:i + 1], pd.DataFrame([new_row]), results.iloc[i + 1:]]).reset_index(drop=True)

                # 修改下一行的 roll_signal 为 1
                results.at[i + 2, 'roll_signal'] = 1
                results.at[i + 2, 'operation'] = next_row['position']
                # 跳过刚插入的新行，继续处理
                i += 2
            else:
                # 继续到下一个
                i += 1

        if 'Unnamed: 0' in results.columns:
            results.drop(columns=['Unnamed: 0'], inplace=True)

        self.df = results
        return self.df

    def query_trade_prices(self):
        """
        输入：上一步处理好的df
        输出：匹配好每天每个合约交易价格的df
        """
        # 不区分日夜盘
        # 定义wap结果表格所在目录
        symbol = self.symbol
        df = self.df
        wap_dir = '/nas92/temporary/Steiner/data_wash/linux_so/py311/wap_results_oi/'

        index_wap_dir = '/nas92/temporary/Steiner/data_wash/linux_so/py311/wap_results_index/'
        # 查找符合symbol的wap结果文件
        wap_file = [file for file in os.listdir(wap_dir) if
                    file.startswith(symbol + '_') and file.endswith('_results.csv')]
        if not wap_file:
            raise FileNotFoundError(f"No matching files found for symbol {symbol} in directory {wap_dir}")

        index_wap_file = [file for file in os.listdir(index_wap_dir) if
                          file.startswith(symbol + '_') and file.endswith('_results.csv')]

        if not index_wap_file:
            raise FileNotFoundError(f"No matching files found for symbol {symbol} in directory {index_wap_dir}")

        wap_filepath = os.path.join(wap_dir, wap_file[0])
        wap_df = pd.read_csv(wap_filepath)

        index_wap_filepath = os.path.join(index_wap_dir, index_wap_file[0])
        index_wap_df = pd.read_csv(index_wap_filepath)

        # 将日期格式转换为一致的格式
        wap_df['date'] = pd.to_datetime(wap_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        wap_df['trading_date'] = pd.to_datetime(wap_df['trading_date'], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
        wap_df['time'] = pd.to_datetime(wap_df['start_time']).dt.time

        index_wap_df['date'] = pd.to_datetime(index_wap_df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        index_wap_df['trading_date'] = pd.to_datetime(index_wap_df['trading_date'], format='%Y-%m-%d').dt.strftime(
            '%Y-%m-%d')
        index_wap_df['time'] = pd.to_datetime(index_wap_df['start_time']).dt.time

        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        # 定义匹配时间
        day_time = pd.to_datetime('14:30:00').time()
        night_time = pd.to_datetime('22:30:00').time()

        # 创建新列 'trade_price'，初始化为空值
        df['trade_price'] = None

        # 定义一个辅助函数进行匹配
        def match_trade_price(row, wap_df):
            """
            输入：wap价格表格，df每行
            输出：根据日期和合约代码匹配交易价格trading_price
            """
            trading_date = row['date']
            contract = row['contract']
            daynight = row['daynight']

            # 筛选对应的交易数据
            try:
                matched_wap_df = wap_df[(wap_df['trading_date'] == trading_date) & (wap_df['contract'] == contract)]

            except KeyError:
                matched_wap_df = wap_df[(wap_df['trading_date'] == trading_date)]

            # 根据 daynight 列的值选择合适的 time 进行匹配
            if daynight == 'day':
                matched_row = matched_wap_df[matched_wap_df['time'] == day_time]
            else:
                matched_row = matched_wap_df[matched_wap_df['time'] == night_time]

            if not matched_row.empty:
                return matched_row['5m_vwap_post_prc'].values[0]
            return None

        # 应用辅助函数匹配 'trade_price'
        df['trade_price'] = df.apply(match_trade_price, axis=1, wap_df=wap_df)
        df['index_trade_price'] = df.apply(match_trade_price, axis=1, wap_df=index_wap_df)
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        self.df = df
        return self.df

    def add_snapshot(self):
        """
        根据symbol添加snapshot数据库中的信息
        输出：添加了可交易标签、base_price，开盘价，前收和收盘价
        """
        symbol = self.symbol
        df = self.df
        # 设置文件夹路径
        snapshot_dir = '/nas92/temporary/Steiner/data_wash/linux_so/py311/snapshot_oi_temp/'
        index_snapshot_dir = '/nas92/temporary/Steiner/data_wash/linux_so/py311/snapshot_index_temp/'

        # 找到符合条件的 CSV 文件
        files = [f for f in os.listdir(snapshot_dir) if f.startswith(symbol + '_') and f.endswith('_results.csv')]
        if not files:
            raise FileNotFoundError(f"No CSV file found for symbol: {symbol}")

        index_files = [f for f in os.listdir(index_snapshot_dir) if
                       f.startswith(symbol + '_') and f.endswith('_results.csv')]
        if not index_files:
            raise FileNotFoundError(f"No CSV file found for symbol: {symbol}")

        # 读取对应的 snapshot_df
        snapshot_df = pd.read_csv(os.path.join(snapshot_dir, files[0]))
        index_snapshot_df = pd.read_csv(os.path.join(index_snapshot_dir, index_files[0]))

        # 过滤掉 query_notrade 为 1 的行
        snapshot_df = snapshot_df[snapshot_df['query_notrade'] != 1]

        snapshot_df['time'] = pd.to_datetime(snapshot_df['query_time']).dt.time
        index_snapshot_df['time'] = pd.to_datetime(index_snapshot_df['query_time']).dt.time
        # 定义匹配时间
        day_time = pd.to_datetime('14:30:00').time()
        night_time = pd.to_datetime('22:30:00').time()

        # 创建新列 'base_price'，初始化为空值
        df['base_price'] = None
        df['open'] = None
        df['prev_close'] = None
        df['tradable'] = None
        df['close'] = None
        # 定义一个辅助函数进行匹配
        def match_snapshot_data(row, snapshot_df):
            """
            根据日期和交易合约匹配快照文件中的数据
            """
            trading_date = row['date']
            contract = row['contract']
            daynight = row['daynight']
            close = row['daynight']
            # 筛选对应的交易数据
            try:
                # 尝试根据交易日期和合约进行匹配
                contract = row['contract']
                matched_snapshot_df = snapshot_df[
                    (snapshot_df['trading_date'] == trading_date) & (snapshot_df['contract'] == contract)]

                # 根据 daynight 列的值选择合适的 time 进行匹配
                if daynight == 'day':
                    matched_row = matched_snapshot_df[matched_snapshot_df['time'] == day_time]
                else:
                    matched_row = matched_snapshot_df[matched_snapshot_df['time'] == night_time]

                if not matched_row.empty:
                    return matched_row[['last_prc', 'open', 'prev_close', 'close','tradable']].iloc[0]
                return pd.Series([None, None, None,None, None], index=['last_prc', 'open', 'prev_close', 'close','tradable'])

            except KeyError:
                # 如果 contract 列不存在，说明是指数snapshot。只根据交易日期进行匹配
                matched_snapshot_df = snapshot_df[snapshot_df['trading_date'] == trading_date]

                # 根据 daynight 列的值选择合适的 time 进行匹配
                if daynight == 'day':
                    matched_row = matched_snapshot_df[matched_snapshot_df['time'] == day_time]
                else:
                    matched_row = matched_snapshot_df[matched_snapshot_df['time'] == night_time]

                if not matched_row.empty:
                    return matched_row[['last_prc', 'open', 'prev_close','close']].iloc[0]
                return pd.Series([None, None, None,None], index=['last_prc', 'open', 'prev_close','close'])

        # 应用辅助函数匹配 'base_price'，open, pre_close, close
        snapshot_data = df.apply(match_snapshot_data, axis=1, snapshot_df=snapshot_df)

        index_snapshot_data = df.apply(match_snapshot_data, axis=1, snapshot_df=index_snapshot_df)
        # print(snapshot_data)
        df[['base_price', 'open', 'prev_close','close', 'tradable']] = snapshot_data[
            ['last_prc', 'open', 'prev_close', 'close','tradable']]

        df[['index_base_price', 'index_open', 'index_prev_close','index_close']] = index_snapshot_data[
            ['last_prc', 'open', 'prev_close','close']]

        df['prev_close'] = df.groupby('contract')['close'].shift(1)
        df['index_prev_close'] = df.groupby('contract')['index_close'].shift(1)

        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

        self.df = df
        return self.df

    def add_settle_prc(self):
        """
        根据symbol匹配合适的daybar文件，添加settle列
        输出：df
        """
        # 先处理日期格式，使其统一
        symbol = self.symbol
        df = self.df
        folder_path ='/nas92/data/future/daybar'
        file_pattern = os.path.join(folder_path, f"{symbol}_*.csv")

        matching_files = glob.glob(file_pattern)
        if len(matching_files) == 0:
            print(f"No matching daybar file found for symbol: {symbol}")
            return None
        elif len(matching_files) > 1:
            print(f"Multiple matching daybar files found for symbol: {symbol}.")
            print("Please ensure there is only one matching file or refine the search criteria.")
            return None
        else:
            # 读取唯一匹配的文件
            daybar_file = matching_files[0]
            print(f"Using daybar file: {daybar_file}")
            day_bar = pd.read_csv(daybar_file)

        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
        day_bar['TRADE_DT'] = day_bar['TRADE_DT'].astype(str)
        # 重命名列以便合并
        df.rename(columns={'date': 'TRADE_DT', 'contract': 'S_INFO_WINDCODE'}, inplace=True)

        # 执行合并操作
        merged_df = pd.merge(df, day_bar[['TRADE_DT', 'S_INFO_WINDCODE', 'S_DQ_SETTLE']],
                             on=['TRADE_DT', 'S_INFO_WINDCODE'], how='left')

        # 将合并结果中的'S_DQ_SETTLE'列赋值给signal_df的'settle_prc'列
        df['settle_prc'] = merged_df['S_DQ_SETTLE']

        df.rename(columns={'TRADE_DT': 'date', 'S_INFO_WINDCODE': 'contract'}, inplace=True)
        # signal_df = signal_df[['date','daynight',	'position',	'operation',	'contract',	'trade_price','settle_prc']]
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        self.df = df
        return self.df

    def correct_trade(self):
        """
        根据 'tradable' 列的值修正 'corrected_position' 和 'corrected_operation' 列。
        返回值:
        df: 添加了 'corrected_position' 和 'corrected_operation' 列的 DataFrame
        """
        df = self.df
        # 初始化 corrected_position 和 corrected_operation 列
        df['corrected_position'] = df['position']
        df['corrected_operation'] = df['operation']

        df['tradable'] = df['tradable'].fillna(4)
        # 当 'tradable' == 4 时，将 'corrected_operation' 设为 0，行情缺失不交易
        df.loc[df['tradable'] == 4, 'corrected_operation'] = 0

        # 当 'tradable' == 2 时，将 'corrected_operation' > 0 的值改为 0，涨停不能买入
        # df.loc[(df['tradable'] == 2) & (df['corrected_operation'] > 0), 'corrected_operation'] = 0
        df.loc[(df['tradable'] == 2) & (df['roll_signal']==0), 'corrected_operation'] = 0
        # 当 'tradable' == 3 时，将 'corrected_operation' < 0 的值改为 0，跌停不能卖出
        # df.loc[(df['tradable'] == 3) & (df['corrected_operation'] < 0), 'corrected_operation'] = 0
        df.loc[(df['tradable'] == 3)& (df['roll_signal']==0), 'corrected_operation'] = 0

        df.loc[(df['trade_price'] == 0)& (df['roll_signal']==0), 'corrected_operation'] = 0
        # df['position_copy'] = df['position'].where(df['corrected_operation'] != 0)

        df['position_copy'] = df['position'].where(
            ((df['tradable'] == 1) & (df['trade_price'] != 0)) | (df['roll_signal'] == 1)
        )
        # 使用前向填充方法填充 NaN 值
        df['corrected_position'] = df['position_copy'].ffill()
        df['corrected_position'] =df['corrected_position'].fillna(0)
        df.drop(columns=['position_copy'], inplace=True)

        df['corrected_operation'] = df['corrected_position'].diff().fillna(0)

        df.at[df.index[0], 'corrected_operation'] = df.at[df.index[0], 'corrected_position']


        df['corrected_position_shifted'] = df['corrected_position'].shift(1).fillna(0)

        # 找出需要插入新行的索引位置
        condition = (df['corrected_position'] * df['corrected_position_shifted']) < 0
        split_indices = df.index[condition].tolist()

        # 创建一个空的 DataFrame 用于存放最终结果
        result_df = pd.DataFrame(columns=df.columns)

        # 遍历每个需要拆分的索引
        for idx in df.index:
            if idx in split_indices:
                original_row = df.loc[idx].copy()

                # 上面一行：corrected_position=0
                new_row_upper = original_row.copy()
                new_row_upper['corrected_position'] = 0

                # 下面一行：corrected_position 保持原值
                new_row_lower = original_row.copy()

                # 将两行加入结果 DataFrame
                result_df = pd.concat([result_df, pd.DataFrame([new_row_upper]), pd.DataFrame([new_row_lower])],
                                      ignore_index=True)
            else:
                # 对于不需要拆分的行，直接加入结果 DataFrame
                result_df = pd.concat([result_df, pd.DataFrame([df.loc[idx]])], ignore_index=True)

        # 删除不再需要的列
        result_df.drop(columns=['corrected_position_shifted'], inplace=True)

        # 最后重新计算 corrected_operation 列
        result_df['corrected_operation'] = result_df['corrected_position'].diff().fillna(0)
        result_df.at[result_df.index[0], 'corrected_operation'] = result_df.at[result_df.index[0], 'corrected_position']
        self.df = result_df
        return self.df

    def fill_number(self):
        """
        填充价格类的缺失值
        """
        df = self.df

        # df['trade_price'] = df['trade_price'].fillna(df['base_price'].shift(1))
        df['close'] = df['close'].fillna(df['settle_prc'])
        df['prev_close'] = df['prev_close'].fillna(df.groupby('contract')['close'].shift(1))

        df['index_close'] = df['index_close'].replace(r'^\s*$', np.nan, regex=True)
        df['index_close'] = df['index_close'].ffill()

        df['index_prev_close'] = df['index_prev_close'].replace(r'^\s*$', np.nan, regex=True)
        df['index_prev_close'] = df['index_prev_close'].fillna(df['index_close'].shift(1))

        # 填充 trade_price, base_price, open, prev_close 中的0值和空值
        cols_to_fill = ['trade_price', 'base_price', 'open', 'prev_close','close','settle_prc','index_base_price','close','index_prev_close','index_close',]
        for col in cols_to_fill:
            df[col] = df[col].replace(0, np.nan)  # 将0替换为NaN
            df[col] = df[col].fillna(method='ffill')  # 用前一个非0、非空值填充

        for col in cols_to_fill:
            df[col] = df[col].replace(0, np.nan)  # 将0替换为NaN
            df[col] = df[col].fillna(method='bfill')  # 用前一个非0、非空值填充
        self.df = df
        return self.df

    def slippage(self):
        """
        添加滑点 = trading_price - base_price
        """
        df = self.df
        df['slippage'] = df.apply(
            lambda x: x['trade_price'] - x['base_price'] if pd.notnull(x['trade_price']) and pd.notnull(
                x['base_price']) else None, axis=1)
        self.df = df
        return self.df

    def operation_type(self):
        """
        标注开仓(1)、平仓(2)、无操作(0)
        输出：df
        """
        df = self.df

        df['position_shifted'] = df['corrected_position'].shift(1)
        df['position_shifted'] = df['position_shifted'].fillna(0)

        df['operation_type'] = np.where(
            df['corrected_position'].abs() - df['position_shifted'].abs() > 0, 1,
            np.where(df['corrected_position'].abs() - df['position_shifted'].abs() < 0, 2, 0)
        )
        self.df = df
        return self.df

    def calculate_commission(self, operation_type, piece_change, trade_price):
        """
        计算手续费，需要在calculate_pnl中使用
        返回一个float
        """

        symbol = self.symbol
        multi = self.multi
        future_type = self.future_type
        operation_type = int(operation_type)
        if future_type == 1 and operation_type != 0:
            commission = abs(piece_change) * trade_price * comm_dict[symbol][operation_type]

        elif future_type == 2 and operation_type != 0:
            commission = abs(piece_change) / multi * comm_dict[symbol][operation_type]
        else:
            commission = 0
        return commission

    def calculate_pnl(self):
        """
        核心段
        FIFO原则计算每一次的交易合约张数、pnl，成本价格，手续费
        输出：df
        """

        # 换一种算stage_value的方法！
        # 这里合并到iterrows里面
        df = self.df
        initial_capital = self.initial_capital
        leverage = self.leverage
        multi = self.multi
        symbol = self.symbol
        future_type = comm_dict[symbol][0]

        df['actual_stage_value'] = 0.0
        df['actual_pieces'] = 0.0
        df['actual_piece_change'] = 0
        df['actual_pnl'] = 0.0
        df['commission'] = 0.0

        df['theo_stage_value'] = 0.0
        df['theo_pieces'] = 0.0
        df['theo_piece_change'] = 0.0
        df['theo_pnl'] = 0.0

        df['index_stage_value'] = 0.0
        df['index_pieces'] = 0.0
        df['index_piece_change'] = 0.0
        df['index_pnl'] = 0.0

        # FIFO stacks for long and short positions
        long_positions = []
        short_positions = []

        theo_long_positions = []
        theo_short_positions = []

        index_long_positions = []
        index_short_positions = []

        # Initialize the first day with initial capital

        df.at[0, 'actual_stage_value'] = initial_capital
        df.at[0, 'theo_stage_value'] = initial_capital
        df.at[0, 'index_stage_value'] = initial_capital
        # Loop through each row to compute values
        for i in range(1, len(df)):

            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            operation_type = row['operation_type']
            operation = row['operation']

            trade_price = row['trade_price']
            base_price = row['base_price']
            index_base_price = row['index_base_price']

            position = row['corrected_position']
            prev_position = prev_row['corrected_position']

            # Carry forward actual_stage_value for operation_type == 0
            if operation_type == 0:

                df.at[i, 'actual_stage_value'] = prev_row['actual_stage_value']
                df.at[i, 'actual_pieces'] = prev_row['actual_pieces']

                df.at[i, 'theo_stage_value'] = prev_row['theo_stage_value']
                df.at[i, 'theo_pieces'] = prev_row['theo_pieces']

                df.at[i, 'index_stage_value'] = prev_row['index_stage_value']
                df.at[i, 'index_pieces'] = prev_row['index_pieces']

            else:
                # Calculate lots and pieces for non-zero operation_types
                lots = np.floor(
                    abs(prev_row['actual_stage_value'] * leverage * position / (multi * trade_price))) * np.sign(
                    position)
                actual_pieces = lots * multi
                theo_pieces = prev_row['theo_stage_value'] * leverage * position / base_price
                index_pieces = prev_row['index_stage_value'] * leverage * position / index_base_price

                df.at[i, 'actual_pieces'] = actual_pieces
                df.at[i, 'actual_piece_change'] = actual_pieces - prev_row['actual_pieces']

                df.at[i, 'theo_pieces'] = theo_pieces
                df.at[i, 'theo_piece_change'] = theo_pieces - prev_row['theo_pieces']

                df.at[i, 'index_pieces'] = index_pieces
                df.at[i, 'index_piece_change'] = index_pieces - prev_row['index_pieces']

                actual_piece_change = actual_pieces - prev_row['actual_pieces']
                theo_piece_change = theo_pieces - prev_row['theo_pieces']
                index_piece_change = index_pieces - prev_row['index_pieces']

                if operation_type == 1:  # Opening a position (long or short)
                    commission = self.calculate_commission(operation_type, actual_piece_change, trade_price)

                    df.at[i, 'commission'] += commission

                    if operation > 0:
                        long_positions.append((trade_price, df.at[i, 'actual_piece_change']))
                        theo_long_positions.append((base_price, df.at[i, 'theo_piece_change']))
                        index_long_positions.append((index_base_price, df.at[i, 'index_piece_change']))

                    elif operation < 0:
                        short_positions.append((trade_price, -df.at[i, 'actual_piece_change']))
                        theo_short_positions.append((base_price, -df.at[i, 'theo_piece_change']))
                        index_short_positions.append((index_base_price, -df.at[i, 'index_piece_change']))

                elif operation_type in [2, 3]:  # Closing positions
                    actual_pieces_shift = prev_row['actual_pieces']
                    theo_pieces_shift = prev_row['theo_pieces']
                    index_pieces_shift = prev_row['index_pieces']

                    actual_piece_change = df.at[i, 'actual_piece_change']
                    theo_piece_change = df.at[i, 'theo_piece_change']
                    index_piece_change = df.at[i, 'index_piece_change']

                    actual_total_cost = 0
                    actual_total_close_pos = 0

                    theo_total_cost = 0
                    theo_total_close_pos = 0

                    index_total_cost = 0
                    index_total_close_pos = 0
                    commission = self.calculate_commission(operation_type, actual_piece_change, trade_price)
                    df.at[i, 'commission'] += commission
                    # Close long positions
                    if prev_position > 0:
                        while actual_pieces_shift > 0 and long_positions:
                            actual_buy_price, actual_buy_pos = long_positions.pop(0)
                            actual_close_pos = min(actual_buy_pos, -actual_piece_change)
                            actual_pnl = (trade_price - actual_buy_price) * actual_close_pos
                            df.at[i, 'actual_pnl'] += actual_pnl
                            actual_total_cost += actual_buy_price * actual_close_pos
                            actual_total_close_pos += actual_close_pos
                            actual_pieces_shift -= actual_close_pos

                            if actual_buy_pos > actual_close_pos:
                                long_positions.insert(0, (actual_buy_price, actual_buy_pos - actual_close_pos))

                        while theo_pieces_shift > 0 and theo_long_positions:
                            print()
                            buy_price, theo_buy_pos = theo_long_positions.pop(0)
                            theo_close_pos = min(theo_buy_pos, -theo_piece_change)
                            theoretical_pnl = (base_price - buy_price) * theo_close_pos
                            df.at[i, 'theo_pnl'] += theoretical_pnl
                            theo_total_cost += buy_price * theo_close_pos
                            theo_total_close_pos += theo_close_pos
                            theo_pieces_shift -= theo_close_pos
                            if theo_buy_pos > theo_close_pos:
                                theo_long_positions.insert(0, (buy_price, theo_buy_pos - theo_close_pos))
                        # Close short positions

                        while index_pieces_shift > 0 and index_long_positions:

                            index_buy_price, index_buy_pos = index_long_positions.pop(0)
                            index_close_pos = min(index_buy_pos, -index_piece_change)
                            index_pnl = (index_base_price - index_buy_price) * index_close_pos
                            df.at[i, 'index_pnl'] += index_pnl
                            index_total_cost += index_buy_price * index_close_pos
                            index_total_close_pos += index_close_pos
                            index_pieces_shift -= index_close_pos
                            if index_buy_pos > index_close_pos:
                                index_long_positions.insert(0, (index_buy_price, index_buy_pos - index_close_pos))

                    elif prev_position < 0:
                        while actual_pieces_shift < 0 and short_positions:
                            actual_sell_price, actual_sell_pos = short_positions.pop(0)
                            actual_close_pos = min(actual_sell_pos, actual_piece_change)
                            actual_pnl = (actual_sell_price - trade_price) * actual_close_pos
                            df.at[i, 'actual_pnl'] += actual_pnl
                            actual_total_cost += actual_sell_price * actual_close_pos
                            actual_total_close_pos += actual_close_pos
                            actual_pieces_shift += actual_close_pos

                            if actual_sell_pos > actual_close_pos:
                                short_positions.insert(0, (actual_sell_price, actual_sell_pos - actual_close_pos))

                        while theo_pieces_shift < 0 and theo_short_positions:

                            sell_price, theo_sell_pos = theo_short_positions.pop(0)
                            theo_close_pos = min(theo_sell_pos, theo_piece_change)
                            theoretical_pnl = (sell_price - base_price) * theo_close_pos

                            df.at[i, 'theo_pnl'] += theoretical_pnl
                            theo_total_cost += sell_price * theo_close_pos
                            theo_total_close_pos += theo_close_pos
                            theo_pieces_shift += theo_close_pos

                            if theo_sell_pos > theo_close_pos:
                                theo_short_positions.insert(0, (sell_price, theo_sell_pos - theo_close_pos))

                        while index_pieces_shift > 0 and index_long_positions:

                            index_sell_price, index_sell_pos = index_short_positions.pop(0)
                            index_close_pos = min(index_sell_pos, index_piece_change)
                            index_pnl = (index_sell_price - index_base_price) * index_close_pos

                            df.at[i, 'index_pnl'] += index_pnl
                            index_total_cost += index_sell_price * index_close_pos
                            index_total_close_pos += index_close_pos
                            index_pieces_shift -= index_close_pos
                            if index_sell_pos > index_close_pos:
                                index_short_positions.insert(0,
                                                             (index_sell_price, index_sell_pos - index_close_pos))

                    if actual_total_close_pos > 0:
                        df.at[i, 'actual_init_price'] = actual_total_cost / actual_total_close_pos

                    if theo_total_close_pos > 0:
                        df.at[i, 'theo_init_price'] = theo_total_cost / theo_total_close_pos

                    if index_total_close_pos > 0:
                        df.at[i, 'index_init_price'] = index_total_cost / index_total_close_pos
                else:
                    continue
                # Update actual_stage_value with pnl
                df.at[i, 'actual_stage_value'] = prev_row['actual_stage_value'] + df.at[i, 'actual_pnl'] - df.at[
                    i, 'commission']
                df.at[i, 'theo_stage_value'] = prev_row['theo_stage_value'] + df.at[i, 'theo_pnl']
                df.at[i, 'index_stage_value'] = prev_row['index_stage_value'] + df.at[i, 'index_pnl']
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

        return df

    def calculate_pnl_new(self):
        """

        """

        # 换一种算stage_value的方法！用剩余资金+每天价值的变动
        # 这里合并到iterrows里面
        df = self.df
        initial_capital = self.initial_capital
        leverage = self.leverage
        multi = self.multi
        symbol = self.symbol
        future_type = comm_dict[symbol][0]
        total_capital = initial_capital * leverage
        df['actual_stage_value'] = 0.0
        df['actual_pieces'] = 0.0
        df['actual_piece_change'] = 0
        df['commission'] = 0.0

        df['theo_stage_value'] = 0.0
        df['theo_pieces'] = 0.0
        df['theo_piece_change'] = 0.0

        df['index_stage_value'] = 0.0
        df['index_pieces'] = 0.0
        df['index_piece_change'] = 0.0

        df.at[0, 'actual_stage_value'] = initial_capital
        df.at[0, 'theo_stage_value'] = initial_capital
        df.at[0, 'index_stage_value'] = initial_capital

        df.at[0, 'actual_free'] = initial_capital
        df.at[0, 'theo_free'] = initial_capital
        df.at[0, 'index_free'] = initial_capital
        # Loop through each row to compute values
        for i in range(1, len(df)):

            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            operation_type = row['operation_type']
            operation = row['operation']

            trade_price = row['trade_price']
            base_price = row['base_price']
            index_base_price = row['index_base_price']
            close = row['close']
            index_close = row['index_close']
            position = row['corrected_position']
            prev_position = prev_row['corrected_position']

            if operation_type == 0:

                df.at[i, 'actual_free'] = actual_free = prev_row['actual_free']
                df.at[i, 'actual_pieces'] = actual_pieces = prev_row['actual_pieces']
                df.at[i, 'actual_stage_value'] = actual_free + trade_price * actual_pieces

                df.at[i, 'theo_free'] = theo_free = prev_row['theo_free']
                df.at[i, 'theo_pieces'] = theo_pieces = prev_row['theo_pieces']
                df.at[i, 'theo_stage_value'] = theo_free + base_price * theo_pieces

                df.at[i, 'index_free'] = index_free = prev_row['index_free']
                df.at[i, 'index_pieces'] = index_pieces = prev_row['index_pieces']
                df.at[i, 'index_stage_value'] = index_free + index_base_price * index_pieces


            else:
                # Calculate lots and pieces for non-zero operation_types
                lots = np.floor(
                    abs(prev_row['actual_stage_value'] * leverage * position / (multi * trade_price))) * np.sign(
                    position)
                actual_pieces = lots * multi
                theo_pieces = prev_row['theo_stage_value'] * leverage * position / base_price
                index_pieces = prev_row['index_stage_value'] * leverage * position / index_base_price

                df.at[i, 'actual_pieces'] = actual_pieces
                df.at[i, 'actual_piece_change'] = actual_pieces - prev_row['actual_pieces']

                df.at[i, 'theo_pieces'] = theo_pieces
                df.at[i, 'theo_piece_change'] = theo_pieces - prev_row['theo_pieces']

                df.at[i, 'index_pieces'] = index_pieces
                df.at[i, 'index_piece_change'] = index_pieces - prev_row['index_pieces']

                actual_piece_change = actual_pieces - prev_row['actual_pieces']
                theo_piece_change = theo_pieces - prev_row['theo_pieces']
                index_piece_change = index_pieces - prev_row['index_pieces']

                if operation_type == 1:  # Opening a position (long or short)

                    commission = self.calculate_commission(operation_type=operation_type, piece_change=actual_piece_change, trade_price=trade_price)

                    df.at[i, 'commission'] += commission

                    actual_free = prev_row['actual_free'] - trade_price * actual_piece_change - commission
                    theo_free = prev_row['theo_free'] - base_price * theo_piece_change
                    index_free = prev_row['index_free'] - index_base_price * index_piece_change

                    df.at[i, 'actual_stage_value'] = actual_free + close * actual_pieces
                    df.at[i, 'theo_stage_value'] = theo_free + close * theo_pieces
                    df.at[i, 'index_stage_value'] = index_free + index_close * index_pieces

                    df.at[i, 'actual_free'] = actual_free
                    df.at[i, 'theo_free'] = theo_free
                    df.at[i, 'index_free'] = index_free

                elif operation_type in [2, 3]:  # Closing positions
                    actual_pieces_shift = prev_row['actual_pieces']
                    theo_pieces_shift = prev_row['theo_pieces']
                    index_pieces_shift = prev_row['index_pieces']

                    actual_piece_change = df.at[i, 'actual_piece_change']
                    theo_piece_change = df.at[i, 'theo_piece_change']
                    index_piece_change = df.at[i, 'index_piece_change']

                    commission = self.calculate_commission(operation_type=operation_type, piece_change=actual_piece_change, trade_price=trade_price)

                    df.at[i, 'commission'] += commission

                    actual_free = prev_row['actual_free'] - trade_price * actual_piece_change - commission
                    theo_free = prev_row['theo_free'] - base_price * theo_piece_change
                    index_free = prev_row['index_free'] - index_base_price * index_piece_change

                    df.at[i, 'actual_stage_value'] = actual_free + close * actual_pieces
                    df.at[i, 'theo_stage_value'] = theo_free + close * theo_pieces
                    df.at[i, 'index_stage_value'] = index_free + index_close * index_pieces

                    df.at[i, 'actual_free'] = actual_free
                    df.at[i, 'theo_free'] = theo_free
                    df.at[i, 'index_free'] = index_free

        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

        return df

    # 包含浮盈浮亏计算每日net_value。思路：计算出每一天的浮盈和浮亏（持仓数（包含正负）*价格变化）
    # 当天开仓：当天的浮动 =（ 收盘价-开仓价）* 张数， operation_type =1，actual_net_value= （close-trade_price）*piece_change + (pieces-piece_change )*(close-prev_close) = pieces*(close-prev_close) +piece_change *(prev_close)-trade_price
    # 当天平仓： 当天的浮动 = （平仓价-前一天收盘价）*张数， operation_type = 2
    # 平今，operation_type = 3， 沿用theoretical_pnl， actual_pnl
    # 当天无操作： （收盘价-前一天收盘价）*张数
    # trade_price = 0 : 填入base_price。

    def calculate_margin_and_free_capital(self):
        """
        计算保证金的占用情况
        输出：df
        """
        df = self.df
        margin_df = self.margin_df
        df['date'] = df['date'].astype(str)
        margin_df['TRADE_DT'] = margin_df['TRADE_DT'].astype(str)
        margin_df['S_INFO_WINDCODE'] = margin_df.apply(lambda row: correct_czc_code(row['S_INFO_WINDCODE'], row['TRADE_DT']), axis=1)
        # 合并数据框，使用日期和合约代码作为键
        merged_df = pd.merge(
            df,
            margin_df[['S_INFO_WINDCODE', 'TRADE_DT', 'MARGINRATIO']],
            left_on=['contract', 'date'],
            right_on=['S_INFO_WINDCODE', 'TRADE_DT'],
            how='left'
        )

        # 删除多余的合并键
        merged_df.drop(columns=['S_INFO_WINDCODE', 'TRADE_DT'], inplace=True)

        merged_df['MARGINRATIO'].fillna(method='ffill', inplace=True)

        # 填充剩余空值：对每个合约，使用第一个非空值进行填充
        def fill_first_pct_chg_limit(row, merged_df, unmatched_contracts):
            if pd.isna(row['MARGINRATIO']):
                matching_rows = merged_df.loc[merged_df['S_INFO_WINDCODE'] == row['contract'], 'MARGINRATIO']
                if not matching_rows.empty:
                    return matching_rows.iloc[0]
                else:
                    unmatched_contracts.append(row['contract'])
                    return pd.NA  # 如果找不到匹配项，返回 NaN
            return row['MARGINRATIO']

        merged_df['MARGINRATIO'].fillna(method='bfill', inplace=True)
        unmatched_contracts = []
        merged_df['MARGINRATIO'] = merged_df.apply(
            lambda row: fill_first_pct_chg_limit(row, margin_df, unmatched_contracts), axis=1)

        if unmatched_contracts:
            print(f"未找到匹配项的 contract: {set(unmatched_contracts)}")

        # mean_value = merged_df['MARGINRATIO'].mean()

        merged_df['MARGINRATIO'].replace([pd.NA, np.inf, -np.inf], 10, inplace=True)
        merged_df['MARGINRATIO'] = merged_df['MARGINRATIO'].astype(float)

        merged_df['MARGINRATIO'] = merged_df['MARGINRATIO'] / 100

        # 将列重命名为 'limit'
        merged_df.rename(columns={'MARGINRATIO': 'margin_rate'}, inplace=True)

        df = merged_df
        # 获取对应品种的保证金费率
        initial_capital = self.initial_capital

        # 计算空闲资金数量
        df['free_capital'] = initial_capital - df['actual_pieces'].abs() * df['settle_prc'] * df['margin_rate']

        # 计算保证金占用比例
        df['occupied_margin_ratio'] = 1 - df['free_capital'] / initial_capital
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        self.df = df
        return self.df

    def calculate_actual_net_value_change(self, row):
        """
        计算每天包含浮盈浮亏的净值变化
        在net_value_change中使用
        """
        if row['operation_type'] == 1:
            return row['actual_piece_change'] * (row['close'] - row['trade_price']) + (
                    row['actual_pieces'] - row['actual_piece_change']) * (row['close'] - row['prev_close'])
        elif row['operation_type'] == 2 or 3:

            return (-row['trade_price'] + row['prev_close']) * row['actual_piece_change'] + (
                    row['actual_pieces_shift'] + row['actual_piece_change']) * (row['close'] - row['prev_close'])

        # elif row['operation_type'] == 3:
        #     return row['actual_pnl'] + (row['pieces_shift'] + row['piece_change']) * (row['close'] - row['prev_close'])
        elif row['operation_type'] == 0:
            return row['pieces'] * (row['close'] - row['prev_close'])
        else:
            return 0  # 若无效的 operation_type 返回 0

    def calculate_theo_net_value_change(self, row):
        """
        计算每天包含浮盈浮亏的净值变化
        在net_value_change中使用
        """
        if row['operation_type'] == 1:
            return row['theo_piece_change'] * (row['close'] - row['base_price']) + (
                    row['theo_pieces'] - row['theo_piece_change']) * (row['close'] - row['prev_close'])

        elif row['operation_type'] == 2 or 3:

            return (-row['base_price'] + row['prev_close']) * row['theo_piece_change'] + (
                    row['theo_pieces_shift'] + row['theo_piece_change']) * (row['close'] - row['prev_close'])

        elif row['operation_type'] == 0:
            return row['theo_pieces'] * (row['close'] - row['prev_close'])

        else:
            return

    def calculate_index_net_value_change(self, row):
        """
        计算每天包含浮盈浮亏的净值变化
        在net_value_change中使用
        """
        if row['operation_type'] == 1:
            return row['index_piece_change'] * (row['index_close'] - row['index_base_price']) + (
                    row['index_pieces'] - row['index_piece_change']) * (row['index_close'] - row['index_prev_close'])

        elif row['operation_type'] == 2 or 3:

            return (-row['index_base_price'] + row['index_prev_close']) * row['index_piece_change'] + (
                    row['index_pieces_shift'] + row['index_piece_change']) * (
                    row['index_close'] - row['index_prev_close'])

        elif row['operation_type'] == 0:
            return row['index_pieces'] * (row['index_close'] - row['index_prev_close'])

        else:
            return

    def net_value_change(self):
        df = self.df
        df['actual_pieces_shift'] = df['actual_pieces'].shift(1).fillna(0)
        df['theo_pieces_shift'] = df['theo_pieces'].shift(1).fillna(0)
        df['index_pieces_shift'] = df['index_pieces'].shift(1).fillna(0)

        df['actual_net_value_change'] = df.apply(self.calculate_actual_net_value_change, axis=1) - df['commission']
        df['theo_net_value_change'] = df.apply(self.calculate_theo_net_value_change, axis=1)
        df['index_net_value_change'] = df.apply(self.calculate_index_net_value_change, axis=1)
        self.df = df
        return self.df

    def net_value(self):
        df = self.df
        initial_capital = self.initial_capital
        df['actual_net_value'] = initial_capital + df['actual_net_value_change'].cumsum()
        df['theo_net_value'] = initial_capital + df['theo_net_value_change'].cumsum()
        df['index_net_value'] = initial_capital + df['index_net_value_change'].cumsum()

        df['actual_net_value2'] = (df['actual_net_value'] - df['commission'].cumsum()) / initial_capital
        df['theo_net_value2'] = df['theo_net_value']/ initial_capital
        df['index_net_value2'] = df['index_net_value']/ initial_capital
        base_index_price = df['index_base_price'][0]
        df['index_base_price2'] = df['index_base_price']/base_index_price

        self.df = df
        return self.df


    def slippage_rounding_revenue(self):
        """
        计算滑点收益和取整收益
        """
        # 添加滑点收益列
        df = self.df
        df['slippage_profit'] = df['slippage'] * df['actual_piece_change'] * df['operation'] * (-1)

        # 添加取整收益列
        df['round_profit'] = - df['theo_net_value_change'] + df['actual_net_value_change'] - df['slippage_profit'] + df[
            'commission']
        self.df = df
        return self.df

    def convenience_profit(self):
        """
        计算便利收益
        """
        df = self.df
        df['convenience_profit'] = -df['theo_net_value_change'] + df['index_net_value_change']
        self.df = df
        return self.df

    def calculate_roll_revenue(self):
        """
        计算展期收益
        """
        df = self.df
        df['roll_profit'] = 0
        i = 0
        while i < len(df) - 1:
            if df.iloc[i]['roll_signal'] == 1 and df.iloc[i + 1]['roll_signal'] == 1:
                A1 = df.iloc[i]['actual_net_value_change'] - df.iloc[i]['theo_net_value_change']
                A2 = df.iloc[i + 1]['actual_net_value_change'] - df.iloc[i + 1]['theo_net_value_change']
                A = A1 + A2
                B = df.iloc[i]['actual_piece_change'] + df.iloc[i + 1]['actual_piece_change']
                C1 = df.iloc[i]['actual_piece_change']
                C2 = df.iloc[i + 1]['actual_piece_change']
                roll_revenue1 = 0
                roll_revenue2 = 0
                # print(A, C1,C2,B,A2)
                if C1 > 0 > C2:
                    if B > 0:
                        # i行的值按B/C1倍数更新
                        df.loc[i, ['commission', 'slippage_profit', 'round_profit']] *= B / C1
                        # i+1行的值设为0
                        df.loc[i + 1, ['commission', 'slippage_profit', 'round_profit']] = 0
                        roll_revenue1 = A1 - B / C1 * A1
                        roll_revenue2 = A2
                    else:
                        # i+1行的值按B/C2倍数更新
                        df.loc[i + 1, ['commission', 'slippage_profit', 'round_profit']] *= B / C2
                        # i行的值设为0
                        df.loc[i, ['commission', 'slippage_profit', 'round_profit']] = 0
                        # roll_revenue = A - B / C2 * A2
                        roll_revenue1 = A1 - B / C1 * A1
                        roll_revenue2 = A2
                elif C1 < 0 < C2:
                    if B > 0:
                        # i+1行的值按B/C2倍数更新
                        df.loc[i + 1, ['commission', 'slippage_profit', 'round_profit']] *= B / C2
                        # i行的值设为0
                        df.loc[i, ['commission', 'slippage_profit', 'round_profit']] = 0
                        # roll_revenue = A - B / C2 * A2
                        roll_revenue1 = A1 - B / C1 * A1
                        roll_revenue2 = A2
                    else:
                        # i行的值按B/C1倍数更新
                        df.loc[i, ['commission', 'slippage_profit', 'round_profit']] *= B / C1
                        # i+1行的值设为0
                        df.loc[i + 1, ['commission', 'slippage_profit', 'round_profit']] = 0
                        # roll_revenue = A - B / C1 * A1
                        roll_revenue1 = A1 - B / C1 * A1
                        roll_revenue2 = A2

                df.loc[i, 'roll_profit'] = roll_revenue1
                df.loc[i + 1, 'roll_profit'] = roll_revenue2

                i += 2
            else:
                i += 1
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        self.df = df
        return self.df

    def drawdown(self):
        df = self.df
        initial_capital = self.initial_capital
        df['current_max'] = df['theo_net_value'].cummax()
        df['current_drawdown'] = (df['theo_net_value'] - df['current_max']) / df['current_max']

        # 创建 max_time 列记录之前的最大值出现的时间
        def find_max_time(row):
            max_rows = df[df['theo_net_value'].cummax() == row['current_max']]
            if not max_rows.empty:
                return max_rows['date'].iloc[0]
            return None

        # df['start_time'] = df.apply(lambda row: df.loc[df['theo_net_value'].cummax()== row['current_max'], 'date'], axis=1)
        df['max_time'] = df.apply(find_max_time, axis=1)
        df['total_pnl'] = df['theo_net_value'] - initial_capital
        self.df = df
        return self.df

    def yields(self):
        """
        计算每天的收益率
        """
        df = self.df
        # df = df[df['tradable'] != 4]
        df['actual_yields'] = df['actual_net_value'].pct_change(fill_method=None)
        df['theo_yields'] = df['theo_net_value'].pct_change(fill_method=None)
        df['index_yields'] = df['index_net_value'].pct_change(fill_method=None)

        df['slippage_yields'] = df['slippage_profit'] / df['theo_net_value']
        df['round_yields'] = df['round_profit'] / df['theo_net_value']
        df['roll_yields'] = df['roll_profit'] / df['theo_net_value']
        df['convenience_yields'] = df['convenience_profit'] / df['theo_net_value']
        self.df = df
        return self.df

    def net_value_table(self):
        """
        输出净值表格
        """
        df = self.df
        selected_columns = [
            'date', 'daynight', 'total_pnl', 'actual_net_value', 'theo_net_value', 'theo_net_value2',
            'actual_net_value2',
            'commission', 'slippage_profit', 'round_profit', 'roll_profit',
            'theo_net_value_change', 'theo_yields', 'actual_yields', 'current_drawdown'
        ]

        net_value_df = df.loc[:, selected_columns]
        return net_value_df

    def find_top_drawdowns(self, top_n=3):
        """
        找到不同最高净值点开始的3个最大回撤区间
        """
        df = self.df
        # 按 max_time 分组，找到每组中最小的 current_drawdown 和对应的日期
        grouped = df.groupby('max_time').apply(lambda x: x.loc[x['current_drawdown'].idxmin()])

        # 对最小的 current_drawdown 进行排序，选择前 top_n 个最小值
        top_drawdowns = grouped.nsmallest(top_n, 'current_drawdown')
        results = []

        for i, row in enumerate(top_drawdowns.itertuples(), start=1):
            # 获取下一个 max_time 的日期作为 end_date
            filtered_df = df[df['date'] >= row.date]['max_time'].drop_duplicates()

            # next_max_time = df[df['date'] >= row.date]['max_time'].drop_duplicates().iloc[1]
            # if not df[df['date'] > row.date]['max_time'].drop_duplicates().empty else None
            if len(filtered_df) > 1:
                next_max_time = filtered_df.iloc[1]
            else:
                next_max_time = None  # 或者你可以选择其他默认值
            start_date_str = str(int(row.max_time))
            bottom_date_str = str(int(row.date))
            end_date_str = str(int(next_max_time)) if next_max_time is not None else None

            # 将结果存储为字典
            results.append({
                'index': ['max', 'second', 'third'][i - 1],
                'ratio': round(row.current_drawdown, 4),
                'start_date': start_date_str,
                'bottom_date': bottom_date_str,
                'end_date': end_date_str
            })

        # 使用 pd.DataFrame 创建结果 DataFrame
        drawdown_df = pd.DataFrame(results)

        return drawdown_df

    def get_daily_df(self):
        df = self.df

        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        daily_df = df.groupby('date').agg(
            daily_pnl_sum=('theo_net_value_change', 'sum'),
            daily_yields=('theo_yields', lambda x: np.prod(1 + x) - 1),
            position=('position', 'last'),
            operation=('operation', 'sum'),
            trade_price=('trade_price', 'mean'),
            base_price=('base_price', 'mean'),
            open_price=('open', 'first'),
            prev_close=('prev_close', 'first'),
            index_base_price=('index_base_price', 'mean'),
            index_open_price=('index_open', 'first'),
            index_prev_close=('index_prev_close', 'first'),
            slippage=('slippage', 'mean'),
            close_price=('close', 'last'),
            settle_prc=('settle_prc', 'last'),
            corrected_position=('corrected_position', 'last'),
            corrected_operation=('corrected_operation', 'sum'),
            actual_stage_value=('actual_stage_value', 'last'),
            actual_pieces=('actual_pieces', 'last'),
            # actual_pnl=('actual_pnl', 'sum'),
            commission=('commission', 'sum'),
            theo_stage_value=('theo_stage_value', 'last'),
            theo_pieces=('theo_pieces', 'last'),
            # theo_pnl=('theo_pnl', 'sum'),
            # actual_init_price=('actual_init_price', 'mean'),
            # theo_init_price=('theo_init_price', 'mean'),
            # index_init_price=('index_init_price', 'mean'),
            free_capital=('free_capital', 'last'),
            margin_ratio=('occupied_margin_ratio', 'last'),
            actual_net_value_change=('actual_net_value_change', 'sum'),
            theo_net_value_change=('theo_net_value_change', 'sum'),
            index_net_value_change=('index_net_value_change', 'sum'),
            actual_net_value=('actual_net_value', 'last'),
            theo_net_value=('theo_net_value', 'last'),
            index_net_value=('index_net_value', 'last'),
            actual_net_value2=('actual_net_value2', 'last'),
            theo_net_value2=('theo_net_value2', 'last'),
            index_net_value2=('index_net_value2', 'last'),
            roll_profit=('roll_profit', 'sum'),
            slippage_profit=('slippage_profit', 'sum'),
            round_profit=('round_profit', 'sum'),
            convenience_profit=('convenience_profit', 'sum'),
            actual_yields=('actual_yields', lambda x: np.prod(1 + x) - 1),
            theo_yields=('theo_yields', lambda x: np.prod(1 + x) - 1),
            index_yields=('index_yields', lambda x: np.prod(1 + x) - 1),
            roll_yields=('roll_yields', lambda x: np.prod(1 + x) - 1),
            slippage_yields=('slippage_yields', lambda x: np.prod(1 + x) - 1),
            round_yields=('round_yields', lambda x: np.prod(1 + x) - 1),
            convenience_yields=('convenience_yields', lambda x: np.prod(1 + x) - 1),
            current_max=('current_max', 'max'),
            current_drawdown=('current_drawdown', 'min'),
            max_time=('max_time', 'last'),  # max_time对应于每组最后一个值
            total_pnl=('total_pnl', 'last')
        ).reset_index()

        daily_df.rename(columns={'daily_yields_product': 'daily_yields'}, inplace=True)
        if 'Unnamed: 0' in daily_df.columns:
            daily_df.drop(columns=['Unnamed: 0'], inplace=True)
        return daily_df

    def calculate_ratios(self, daily_df):

        # 计算年化收益率、年化波动率、夏普比率、索提诺比率
        daily_df['theo_net_value2'] = daily_df['theo_net_value2'].astype(float)

        # 年化收益率
        first_valid_index = daily_df['theo_net_value2'].first_valid_index()
        last_valid_index = daily_df['theo_net_value2'].last_valid_index()
        annual_return = (daily_df['theo_net_value2'].iloc[last_valid_index] / daily_df['theo_net_value2'].iloc[
            first_valid_index]) ** (
                                250 / len(daily_df['date'].unique())) - 1

        # 年化波动率
        annual_volatility = daily_df['theo_yields'].std() * np.sqrt(250)

        # 夏普比率
        sharpe_ratio = annual_return / annual_volatility

        # 索提诺比率
        sortino_ratio = annual_return / (daily_df['theo_yields'][daily_df['theo_yields'] < 0].std() * np.sqrt(250))

        ratio_df = pd.DataFrame({
            'Annual_Return': [annual_return],
            'Annual_Volatility': [annual_volatility],
            'Sharpe_Ratio': [sharpe_ratio],
            'Sortino_Ratio': [sortino_ratio]
        }).round(4)

        return ratio_df, sharpe_ratio

    def win_rate(self, daily_df):

        # 胜率,
        # Convert date column to datetime format
        daily_df['date'] = pd.to_datetime(daily_df['date'], format='%Y%m%d')

        # Calculate daily metrics
        total_days = daily_df['daily_pnl_sum'].ne(0).sum()  # 收益不等于0的天数
        win_days = daily_df['daily_pnl_sum'].gt(0).sum()  # 收益大于0的天数
        daily_win_rate = win_days / total_days if total_days > 0 else 0
        daily_expected_return = daily_df['daily_yields'].mean()

        positive_yields_mean = daily_df.loc[daily_df['daily_yields'] > 0, 'daily_yields'].mean()
        negative_yields_mean = daily_df.loc[daily_df['daily_yields'] < 0, 'daily_yields'].mean()

        # Resample to calculate weekly metrics
        weekly_agg = daily_df.resample('W-MON', on='date').agg(
            weekly_pnl_sum=('daily_pnl_sum', 'sum'),
            weekly_yields_sum=('daily_yields', 'sum')
        ).reset_index()

        total_weeks = weekly_agg['weekly_pnl_sum'].ne(0).sum()  # Count of non-zero PnL weeks
        win_weeks = weekly_agg['weekly_pnl_sum'].gt(0).sum()  # Count of profitable weeks
        weekly_win_rate = win_weeks / total_weeks if total_weeks > 0 else 0
        weekly_expected_return = weekly_agg['weekly_yields_sum'].mean()

        # Create summary DataFrame
        summary = pd.DataFrame({
            'daily_win_rate': [daily_win_rate],
            'daily_expected_return': [daily_expected_return],
            'positive_expected_return': [positive_yields_mean],
            'negative_expected_return': [negative_yields_mean],
            'weekly_win_rate': [weekly_win_rate],
            'weekly_expected_return': [weekly_expected_return]
        }).round(4)

        return summary

    def calculate_trading_frequency(self):
        df = self.df

        filtered_df = df[df['roll_signal'] == 0]

        # 计算总交易数和不同交易日的数目
        total_trades = df['corrected_operation'].abs().sum() / 2
        total_days = df['date'].nunique()

        # 计算交易频率，一天交易多少次
        trades_per_day = total_trades / total_days if total_days > 0 else 0
        # 平均持仓天数（交易日）
        days_hold = 1 / trades_per_day if trades_per_day > 0 else float('inf')

        frequency = pd.DataFrame({
            'average_trades_per_day': [trades_per_day],
            'average_days_hold': [days_hold]
        }).round(4)
        return frequency

    def plot_net_value_and_drawdown_html(self, daily_df, drawdown_df):
        # 使用plotly绘制交互式净值曲线
        fig = go.Figure()

        # 绘制 actual_net_value2 和 theo_net_value2 曲线
        fig.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['actual_net_value2'],
                                 mode='lines', name='Actual Net Value', line=dict(color='#32A1DE')))
        fig.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['theo_net_value2'],
                                 mode='lines', name='Theo Net Value', line=dict(color='#ED8C45')))
        fig.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['index_net_value2'],
                                 mode='lines', name='Index Net Value', line=dict(color='#E868E5')))

        # 绘制 current_drawdown 曲线和填充区域
        fig.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['current_drawdown'],
                                 mode='lines', name='Current Drawdown', line=dict(color='#17F2AB'),
                                 yaxis="y2"))
        fig.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['current_drawdown'],
                                 mode='lines', line=dict(color='rgba(0,0,0,0)'),
                                 fill='tozeroy', fillcolor='rgba(23,242,171,0.15)', yaxis="y2", showlegend=False))

        # 标记 drawdown_df 中的最大回撤区间
        for _, row in drawdown_df.iterrows():
            start_date = pd.to_datetime(row['start_date'], format='%Y%m%d')
            if pd.isna(row['end_date']):
                end_date = daily_df['date'].max()
            else:
                end_date = pd.to_datetime(row['end_date'], format='%Y%m%d')

            fig.add_vline(x=start_date, line=dict(color='gray', dash='dash'))
            fig.add_vline(x=end_date, line=dict(color='gray', dash='dash'))
            fig.add_vrect(x0=start_date,
                          x1=end_date,
                          fillcolor='gray', opacity=0.08, line_width=0)

        # 设置图表布局以同时支持横向和纵向缩放
        fig.update_layout(
            title=dict(text="Net Value Curve", font=dict(size=18)),
            xaxis=dict(
                rangeslider=dict(visible=True),
                rangemode='normal',
                scaleanchor="y",  # 横轴和纵轴按比例缩放
                scaleratio=1,  # 确保缩放比例一致
                range=[daily_df['date'].min(), daily_df['date'].max()],
            ),
            yaxis=dict(
                rangemode='normal',
                scaleanchor="x",  # 纵轴和横轴按比例缩放
                scaleratio=1,  # 确保缩放比例一致
                fixedrange=False,  # 确保y轴可以缩放
            ),
            yaxis2=dict(title='Drawdown', overlaying='y', side='right', range=[-1, 0]),
            dragmode='zoom',
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='rgba(245, 245, 245, 1)',
            legend=dict(
                x=0.85, y=1,
                traceorder='normal',
                font=dict(size=10),
                bgcolor='rgba(0,0,0,0)'
            ),
        )

        fig2 = go.Figure()

        # 绘制 Price_Index 曲线
        fig2.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['index_base_price2'],
                                  mode='lines', name='Price Index', line=dict(color='#1670E6')))

        # 绘制 Theo Net Value 和 Index Net Value 曲线
        fig2.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['theo_net_value2'],
                                  mode='lines', name='Theo Net Value', line=dict(color='#ED8C45')))
        fig2.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['index_net_value2'],
                                  mode='lines', name='Index Net Value', line=dict(color='#E868E5')))

        # 获取买入和卖出点的坐标
        # 获取每个日期的最后一个买入点
        # 先按照日期分组，并提取每个日期的最后一个点
        last_points = daily_df.groupby('date').tail(1)

        # 从最后一个点中筛选买入点（corrected_operation > 0）
        buy_points = last_points[(last_points['corrected_operation'] > 0 )&( last_points['roll_signal'] == 0) ]

        # 从最后一个点中筛选卖出点（corrected_operation < 0）
        sell_points = last_points[(last_points['corrected_operation'] < 0 )&( last_points['roll_signal'] == 0)]
        # 绘制买入散点
        fig2.add_trace(go.Scatter(x=buy_points['date'], y=buy_points['index_base_price2'],
                                  mode='markers', name='Buy', marker=dict(color='#E6530C', size=10, symbol='triangle-up')))
        # 绘制卖出散点
        fig2.add_trace(go.Scatter(x=sell_points['date'], y=sell_points['index_base_price2'],
                                  mode='markers', name='Sell',
                                  marker=dict(color='#04E548', size=10, symbol='triangle-down')))

        # 设置图表布局以支持缩放
        fig2.update_layout(
            title=dict(text="Price Index with Buy/Sell Points", font=dict(size=18)),
            xaxis=dict(
                rangeslider=dict(visible=True),
                rangemode='normal',
                scaleanchor="y",  # 横轴和纵轴按比例缩放
                scaleratio=1,  # 确保缩放比例一致
                range=[daily_df['date'].min(), daily_df['date'].max()],
            ),
            yaxis=dict(
                rangemode='normal',
                scaleanchor="x",  # 纵轴和横轴按比例缩放
                scaleratio=1,  # 确保缩放比例一致
                fixedrange=False,  # 确保y轴可以缩放
            ),
            dragmode='zoom',
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='rgba(245, 245, 245, 1)',
            legend=dict(
                x=0.85, y=1,
                traceorder='normal',
                font=dict(size=10),
                bgcolor='rgba(0,0,0,0)'
            ),
        )

        return fig, fig2



    # 生成 HTML 文件
    # 图表加滚轮，比率数据保留四位小数
    def generate_html(self, daily_df, drawdown_df, ratio_df, frequency, win, ):
        html_title = f"{self.factor_name} Report"
        html_file = os.path.join(self.output_path, f"{self.factor_name}_report.html")
        # 绘制图表
        fig1, fig2 = self.plot_net_value_and_drawdown_html(daily_df, drawdown_df)
        # fig2 = self.plot_price_index_and_scatter(daily_df)
        config = {'scrollZoom': True}

        fig1_html = pio.to_html(fig1, full_html=False, config=config)
        fig2_html = pio.to_html(fig2, full_html=False, config=config)

        # 构造HTML文件的内容
        html_content = f"""
        <html>
        <head>
            <title>Net Value Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                h1 {{ text-align: center; }}
                table {{ width: 80%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .indicator-section {{ margin: 30px 0; }}
                .chinese {{ font-family: "Microsoft YaHei", sans-serif; }}
            </style>
        </head>
        <body>
            <h1>Net Value Report</h1>
            {fig1_html}
            {fig2_html}

            <div class="indicator-section">
                <h2 class="chinese">Performance Indicators </h2>
                <h3 class="chinese">Ratio </h3>
                {ratio_df.to_html(index=False)}
                <h3 class="chinese">Frequency </h3>
                {frequency.to_html(index=False)}
                <h3 class="chinese">Win rate </h3>
                {win.to_html(index=False)}
                <h3 class="chinese">Drawdown </h3>
                {drawdown_df.to_html(index=False)}
            </div>
        </body>
        </html>
        """

        # 保存HTML文件
        with open(html_file, "w", encoding="utf-8") as file:
            file.write(f"<html><head><title>{html_title}</title></head><body>")
            file.write("<h1>" + html_title + "</h1>")
            file.write(html_content)

        print("HTML report generated successfully.")

    def adjust_columns(self):
        self.df.drop(columns=['theo_pieces_shift', 'index_pieces_shift', 'index_pieces_shift'], inplace=True)
        return self.df

    def run_backtest(self):
        '''
        执行完整的回测流程，返回最终的df
        '''
        self.df = self.calculate_operation()

        contract_df = self.determine_trade_contracts()

        self.df = self.process_trades(contract_df)

        self.df = self.query_trade_prices()
        self.df = self.add_snapshot()
        self.df = self.add_settle_prc()

        self.df = self.correct_trade()

        self.df = self.fill_number()
        self.df = self.slippage()
        self.df = self.operation_type()

        self.df = self.calculate_pnl_new()

        self.df = self.calculate_margin_and_free_capital()
        self.df = self.net_value_change()
        self.df = self.net_value()
        self.df = self.slippage_rounding_revenue()
        self.df = self.convenience_profit()
        self.df = self.calculate_roll_revenue()
        self.df = self.yields()
        self.df = self.drawdown()

        net_value_df = self.net_value_table()
        drawdown_df = self.find_top_drawdowns()
        daily_df = self.get_daily_df()
        ratio_df, sharpe = self.calculate_ratios(daily_df)
        win_df = self.win_rate(daily_df)
        frequency_df = self.calculate_trading_frequency()
        self.generate_html(self.df, drawdown_df, ratio_df, frequency_df, win_df)
        self.df = self.adjust_columns()
        return self.df, net_value_df, sharpe


if __name__ == '__main__':
    signal_df = pd.read_csv('fb.csv')
    bt = backtest('FB', 1000000, 1, signal_df, factor_name='barra', output_path='')
    df, net_value_df, sharpe = bt.run_backtest()
    # backtest.df = pd.read_csv(r"Z:\data\future\factor\momentum\reports\AG_barra_momentum_detail.csv")
    # drawdown_df = backtest.find_top_drawdowns(backtest.df)
    # backtest.generate_html()
