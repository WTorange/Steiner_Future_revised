import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpld3
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
from io import BytesIO
import pdfkit
from connect_wind import ConnectDatabase

class get_information(ConnectDatabase):
    def __init__(self):

        # self.future_symbol = future_symbol  #例 铜CU(大写)
        # b
        self.sql = f'''
                       SELECT S_INFO_WINDCODE, S_INFO_CODE, S_INFO_PUNIT, S_INFO_FTMARGINS, S_INFO_MAXPRICEFLUCT
                       FROM CFUTURESCONTPRO
                       '''

        super().__init__(self.sql)
        self.df = super().get_data()

    def run(self):

        return self.df

im = get_information()
temp = im.run()

comm_dict  = {
    # 类型，开仓，平仓，平今手续费
    'I':(1, 0.00012, 0.00006, 0.00012),
    'RB':(1, 0.000045, 0.000045, 0.000045),
    'AG':(1, 0.000051,0.000051,0.000051),
    'BR':(1, 0.000021, 0.000021,  0.000021)
}

margin_dict = {
    # symbol : 保证金费率
    'RB': 0.05,
    'AG': 0.12,
    'BR': 0.12
}

multi_dict = {
    'BR': 5
}
class backtest:
    def __init__(self, symbol: str, initial_capital: int, leverage: float, signal_df, start_date, end_date):
        self.symbol = symbol
        self.df = signal_df
        self.leverage = leverage
        self.margin = margin_dict[symbol]
        self.comm_dict = comm_dict
        self.multi = multi_dict[symbol]
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital= initial_capital
    def calculate_operation(self):
        """
        输入：包含信号的dataframe,start_date:yyyymmdd,end_date: yyyymmdd.
        根据初始和结束日期选择信号df中的片段，计算每天的operation
        输出：计算每天的operation，新的df
        """
        # 将date列转换为datetime类型
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df[(self.df['date'] >= self.start_date) & (self.df['date'] <= self.end_date)]
        # 计算operation列
        self.df['operation'] = self.df['position'].diff().fillna(0).astype(int)

        return self.df

    def determine_trade_contracts(self):
        """
        根据主力合约数据获取每天要交易的合约和换仓
        """
        # 确定每天交易的合约代码
        symbol = self.symbol
        folder_path = r"Z:\temporary\Steiner\data_wash\linux_so\py311\essentialcontract"
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
        wap_dir = r'Z:\temporary\Steiner\data_wash\linux_so\py311\temp'

        # 查找符合symbol的wap结果文件
        wap_file = [file for file in os.listdir(wap_dir) if
                    file.startswith(symbol + '_') and file.endswith('_results.csv')]
        if not wap_file:
            raise FileNotFoundError(f"No matching files found for symbol {symbol} in directory {wap_dir}")

        print(wap_file)

        wap_filepath = os.path.join(wap_dir, wap_file[0])

        # 加载wap结果表格
        wap_df = pd.read_csv(wap_filepath)

        # 将日期格式转换为一致的格式
        wap_df['date'] = pd.to_datetime(wap_df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        wap_df['trading_date'] = pd.to_datetime(wap_df['trading_date'], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        wap_df['time'] = pd.to_datetime(wap_df['start_time']).dt.time

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
            matched_wap_df = wap_df[(wap_df['trading_date'] == trading_date) & (wap_df['contract'] == contract)]

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
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        self.df = df
        return self.df


    def add_snapshoot(self):
        """
        根据symbol添加snapshoot数据库中的信息
        输出：添加了可交易标签、base_price，开盘价，前收和收盘价
        """
        symbol = self.symbol
        df = self.df
        # 设置文件夹路径
        directory_path = r'Z:\temporary\Steiner\data_wash\linux_so\py311\snapshoot_results_oi'

        # 找到符合条件的 CSV 文件
        files = [f for f in os.listdir(directory_path) if f.startswith(symbol + '_') and f.endswith('_results.csv')]
        if not files:
            raise FileNotFoundError(f"No CSV file found for symbol: {symbol}")

        # 读取对应的 snapshoot_df
        snapshoot_df = pd.read_csv(os.path.join(directory_path, files[0]))

        # 过滤掉 query_notrade 为 1 的行
        snapshoot_df = snapshoot_df[snapshoot_df['query_notrade'] != 1]
        snapshoot_df['time'] = pd.to_datetime(snapshoot_df['query_time']).dt.time

        # 定义匹配时间
        day_time = pd.to_datetime('14:30:00').time()
        night_time = pd.to_datetime('22:30:00').time()

        # 创建新列 'base_price'，初始化为空值
        df['base_price'] = None
        df['open'] = None
        df['prev_close'] = None
        df['tradable'] = None

        # 定义一个辅助函数进行匹配
        def match_snapshoot_data(row, snapshoot_df):
            """
            根据日期和交易合约匹配快照文件中的数据
            """
            trading_date = row['date']
            contract = row['contract']
            daynight = row['daynight']

            # 筛选对应的交易数据
            matched_snapshoot_df = snapshoot_df[
                (snapshoot_df['trading_date'] == trading_date) & (snapshoot_df['contract'] == contract)]

            # 根据 daynight 列的值选择合适的 time 进行匹配
            if daynight == 'day':
                matched_row = matched_snapshoot_df[matched_snapshoot_df['time'] == day_time]
            else:
                matched_row = matched_snapshoot_df[matched_snapshoot_df['time'] == night_time]

            if not matched_row.empty:
                return matched_row[['last_prc', 'open', 'prev_close', 'tradable']].iloc[0]
            return pd.Series([None, None, None, None], index=['last_prc', 'open', 'prev_close', 'tradable'])

        # 应用辅助函数匹配 'base_price'，open, pre_close, close
        snapshoot_data = df.apply(match_snapshoot_data, axis=1, snapshoot_df=snapshoot_df)
        # print(snapshoot_data)
        df[['base_price', 'open', 'prev_close', 'tradable']] = snapshoot_data[
            ['last_prc', 'open', 'prev_close', 'tradable']]

        df['slippage'] = df.apply(
            lambda x: x['trade_price'] - x['base_price'] if pd.notnull(x['trade_price']) and pd.notnull(
                x['base_price']) else None, axis=1)

        df['close'] = df.groupby('contract')['prev_close'].shift(-1)

        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        self.df =df
        return self.df

    def add_settle_prc(self):
        """
        根据symbol匹配合适的daybar文件，添加settle列
        输出：df
        """
        # 先处理日期格式，使其统一
        symbol = self.symbol
        df = self.df
        folder_path = r"Z:\data\future\daybar"
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

        # 当 'tradable' == 4 时，将 'corrected_operation' 设为 0，行情缺失不交易
        df.loc[df['tradable'] == 4, 'corrected_operation'] = 0

        # 当 'tradable' == 2 时，将 'corrected_operation' > 0 的值改为 0，涨停不能买入
        df.loc[(df['tradable'] == 2) & (df['corrected_operation'] > 0), 'corrected_operation'] = 0

        # 当 'tradable' == 3 时，将 'corrected_operation' < 0 的值改为 0，跌停不能卖出
        df.loc[(df['tradable'] == 3) & (df['corrected_operation'] < 0), 'corrected_operation'] = 0

        df['position_copy'] = df['position'].where(df['corrected_operation'] != 0)

        # 使用前向填充方法填充 NaN 值
        df['corrected_position'] = df['position_copy'].ffill().fillna(0)
        df.drop(columns=['position_copy'], inplace=True)
        df['corrected_operation'] = df['corrected_position'].diff().fillna(0)

        # 计算 corrected_position 列，保证可交易的部分符合原来的目标position
        df['corrected_position'] = df['corrected_operation'].cumsum()

        df['corrected_position_shifted'] = df['corrected_position'].shift(1).fillna(0)

        # 找出需要插入新行的索引位置
        condition = (df['corrected_position'] * df['corrected_position_shifted']) < 0
        insert_indices = df.index[condition].tolist()

        # 复制符合条件的行，并设置相关列为0
        rows_to_insert = df.loc[insert_indices].copy()
        rows_to_insert['position'] = 0
        rows_to_insert['operation'] = 0
        rows_to_insert['corrected_position'] = 0

        # 在原 DataFrame 中插入新行
        for idx in insert_indices:
            # 通过 DataFrame 分割插入新行
            df_upper = df.loc[:idx - 1]  # 上半部分
            df_lower = df.loc[idx:]  # 下半部分

            # 插入新行
            df = pd.concat([df_upper, pd.DataFrame(rows_to_insert.loc[idx]).T, df_lower]).reset_index(drop=True)

        # 重新计算 corrected_operation 列
        df['corrected_operation'] = df['corrected_position'].diff().fillna(0)
        df.drop(columns=['corrected_position_shifted'])
        self.df =df
        return self.df

    def fill_number(self):
        """
        填充价格类的缺失值
        """
        df = self.df
        df['trade_price'] = df['trade_price'].fillna(df['base_price'] + df['slippage'].shift(1))
        df['close'] = df['close'].fillna(df['settle_prc'])
        df['prev_close'] = df['prev_close'].fillna(df.groupby('contract')['close'].shift(1))
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
        future_type = comm_dict[symbol][0]
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
        # 这里合并到iterrows里面
        df = self.df
        initial_capital = self.initial_capital
        leverage = self.leverage
        multi = self.multi
        symbol = self.symbol
        future_type = comm_dict[symbol][0]
        df['actual_stage_value'] = 0.0
        df['actual_pieces'] = 0
        df['actual_piece_change'] = 0
        df['actual_pnl'] = 0.0
        df['commission'] = 0.0

        df['theo_stage_value'] = 0.0
        df['theo_pieces'] = 0
        df['theo_piece_change'] = 0
        df['theo_pnl'] = 0.0

        # FIFO stacks for long and short positions
        long_positions = []
        short_positions = []

        theo_long_positions = []
        theo_short_positions = []
        # Initialize the first day with initial capital

        df.at[0, 'actual_stage_value'] = initial_capital
        df.at[0, 'theo_stage_value'] = initial_capital

        # Loop through each row to compute values
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            operation_type = row['operation_type']
            operation = row['operation']
            trade_price = row['trade_price']
            base_price = row['base_price']
            position = row['corrected_position']
            prev_position = prev_row['corrected_position']

            # Carry forward actual_stage_value for operation_type == 0
            if operation_type == 0:

                df.at[i, 'actual_stage_value'] = prev_row['actual_stage_value']
                df.at[i, 'actual_pieces'] = prev_row['actual_pieces']
                df.at[i, 'theo_stage_value'] = prev_row['theo_stage_value']
                df.at[i, 'theo_pieces'] = prev_row['theo_pieces']

            else:
                # Calculate lots and pieces for non-zero operation_types
                lots = np.floor(
                    abs(prev_row['actual_stage_value'] * leverage * position / (multi * trade_price))) * np.sign(
                    position)
                actual_pieces = lots * multi
                theo_pieces = prev_row['theo_stage_value'] * leverage * position / base_price

                df.at[i, 'actual_pieces'] = actual_pieces
                df.at[i, 'actual_piece_change'] = actual_pieces - prev_row['actual_pieces']

                df.at[i, 'theo_pieces'] = theo_pieces
                df.at[i, 'theo_piece_change'] = theo_pieces - prev_row['theo_pieces']

                actual_piece_change = actual_pieces - prev_row['actual_pieces']
                theo_piece_change = theo_pieces - prev_row['theo_pieces']

                if operation_type == 1:  # Opening a position (long or short)
                    commission = self.calculate_commission(operation_type, actual_piece_change, trade_price)

                    df.at[i, 'commission'] += commission

                    if operation > 0:
                        long_positions.append((trade_price, df.at[i, 'actual_piece_change']))
                        theo_long_positions.append((base_price, df.at[i, 'theo_piece_change']))

                    elif operation < 0:
                        short_positions.append((trade_price, -df.at[i, 'actual_piece_change']))
                        theo_short_positions.append((base_price, -df.at[i, 'theo_piece_change']))


                elif operation_type in [2, 3]:  # Closing positions
                    actual_pieces_shift = prev_row['actual_pieces']
                    theo_pieces_shift = prev_row['theo_pieces']

                    actual_piece_change = df.at[i, 'actual_piece_change']
                    theo_piece_change = df.at[i, 'theo_piece_change']

                    actual_total_cost = 0
                    actual_total_close_pos = 0

                    theo_total_cost = 0
                    theo_total_close_pos = 0

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

                    if actual_total_close_pos > 0:
                        df.at[i, 'actual_init_price'] = actual_total_cost / actual_total_close_pos

                    if theo_total_close_pos > 0:
                        df.at[i, 'theo_init_price'] = theo_total_cost / theo_total_close_pos
                else:
                    continue
                # Update actual_stage_value with pnl
                df.at[i, 'actual_stage_value'] = prev_row['actual_stage_value'] + df.at[i, 'actual_pnl'] - df.at[
                    i, 'commission']
                df.at[i, 'theo_stage_value'] = prev_row['theo_stage_value'] + df.at[i, 'theo_pnl']
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
        '''
        计算保证金的占用情况
        输出：df
        '''

        # 获取对应品种的保证金费率
        symbol = self.symbol
        initial_capital = self.initial_capital
        df = self.df
        margin_rate = margin_dict.get(symbol, 0)

        # 计算空闲资金数量
        df['free_capital'] = initial_capital - df['actual_pieces'].abs() * df['settle_prc'] * margin_rate

        # 计算保证金占用比例
        df['margin_ratio'] = 1 - df['free_capital'] / initial_capital
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        self.df = df
        return self.df

    def calculate_actual_net_value_change(self,row):
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

        # elif row['operation_type'] == 3: return row['actual_pnl'] + (row['theo_pieces_shift'] + row[
        # 'theo_piece_change']) * (row['close'] - row['prev_close'])

        elif row['operation_type'] == 0:
            return row['theo_pieces'] * (row['close'] - row['prev_close'])

        else:
            return 0  # 若无效的 operation_type 返回 0

    def net_value_change(self):
        df = self.df
        df['actual_pieces_shift'] = df['actual_pieces'].shift(1).fillna(0)
        df['theo_pieces_shift'] = df['theo_pieces'].shift(1).fillna(0)
        df['actual_net_value_change'] = df.apply(self.calculate_actual_net_value_change, axis=1) - df['commission']
        df['theo_net_value_change'] = df.apply(self.calculate_theo_net_value_change, axis=1)
        self.df = df
        return self.df

    def net_value(self):
        df = self.df
        initial_capital = self.initial_capital
        df['actual_net_value'] = initial_capital + df['actual_net_value_change'].cumsum()
        df['theo_net_value'] = initial_capital + df['theo_net_value_change'].cumsum()
        df['actual_net_value2'] = 1 + (
                    df['actual_net_value_change'].cumsum() - df['commission'].cumsum()) / initial_capital
        df['theo_net_value2'] = 1 + df['theo_net_value_change'].cumsum() / initial_capital
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

    def yields(self):
        """
        计算每天的收益率
        """
        df = self.df
        df['theo_yields'] = df['theo_net_value'].pct_change(fill_method=None)
        df['actual_yields'] = df['actual_net_value'].pct_change(fill_method=None)
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

    def net_value_table(self):
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
        df = self.df
        # 按 max_time 分组，找到每组中最小的 current_drawdown 和对应的日期
        grouped = df.groupby('max_time').apply(lambda x: x.loc[x['current_drawdown'].idxmin()])

        # 对最小的 current_drawdown 进行排序，选择前 top_n 个最小值
        top_drawdowns = grouped.nsmallest(top_n, 'current_drawdown')

        results = []

        for i, row in enumerate(top_drawdowns.itertuples(), start=1):
            # 获取下一个 max_time 的日期作为 end_date
            next_max_time = df[df['date'] >= row.date]['max_time'].drop_duplicates().iloc[1]
            # if not df[df['date'] > row.date]['max_time'].drop_duplicates().empty else None

            start_date_str = str(int(row.max_time))
            end_date_str = str(int(next_max_time)) if next_max_time is not None else None

            # 将结果存储为字典
            results.append({
                'index': ['max', 'second', 'third'][i - 1],
                'ratio': row.current_drawdown,
                'start_date': start_date_str,
                'end_date': end_date_str
            })

        # 使用 pd.DataFrame 创建结果 DataFrame
        drawdown_df = pd.DataFrame(results)

        return drawdown_df

    def calculate_ratios(self):
        df = self.df
        # 计算年化收益率、年化波动率、夏普比率、索提诺比率
        df['theo_net_value2'] = df['theo_net_value2'].astype(float)
        df['return'] = df['theo_net_value2'].pct_change(fill_method=None)
        df['return'] = df['return'].fillna(0)

        # 年化收益率
        first_valid_index = df['theo_net_value2'].first_valid_index()
        last_valid_index = df['theo_net_value2'].last_valid_index()
        annual_return = (df['theo_net_value2'].iloc[last_valid_index] / df['theo_net_value2'].iloc[
            first_valid_index]) ** (
                                240 / len(df['date'].unique())) - 1

        # 年化波动率
        annual_volatility = df['return'].std() * np.sqrt(240)

        # 夏普比率
        sharpe_ratio = annual_return / annual_volatility

        # 索提诺比率
        sortino_ratio = annual_return / (df['return'][df['return'] < 0].std() * np.sqrt(240))

        ratio_df = pd.DataFrame({
            'Annual_Return': [annual_return],
            'Annual_Volatility': [annual_volatility],
            'Sharpe_Ratio': [sharpe_ratio],
            'Sortino_Ratio': [sortino_ratio]
        })

        return ratio_df

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
            slippage=('slippage', 'mean'),
            close_price=('close', 'last'),
            settle_prc=('settle_prc', 'last'),
            corrected_position=('corrected_position', 'last'),
            corrected_operation=('corrected_operation', 'sum'),
            actual_stage_value=('actual_stage_value', 'last'),
            actual_pieces=('actual_pieces', 'last'),
            actual_pnl=('actual_pnl', 'sum'),
            commission=('commission', 'sum'),
            theo_stage_value=('theo_stage_value', 'last'),
            theo_pieces=('theo_pieces', 'last'),
            theo_pnl=('theo_pnl', 'sum'),
            actual_init_price=('actual_init_price', 'mean'),
            theo_init_price=('theo_init_price', 'mean'),
            free_capital=('free_capital', 'last'),
            margin_ratio=('margin_ratio', 'last'),
            actual_net_value_change=('actual_net_value_change', 'sum'),
            theo_net_value_change=('theo_net_value_change', 'sum'),
            actual_net_value=('actual_net_value', 'last'),
            theo_net_value=('theo_net_value', 'last'),
            actual_net_value2=('actual_net_value2', 'last'),
            theo_net_value2=('theo_net_value2', 'last'),
            slippage_profit=('slippage_profit', 'sum'),
            round_profit=('round_profit', 'sum'),
            actual_yields=('actual_yields', lambda x: np.prod(1 + x) - 1),
            roll_profit=('roll_profit', 'sum'),
            current_max=('current_max', 'max'),
            current_drawdown=('current_drawdown', 'min'),
            max_time=('max_time', 'last'),  # max_time对应于每组最后一个值
            total_pnl=('total_pnl', 'last')
        ).reset_index()

        daily_df.rename(columns={'daily_yields_product': 'daily_yields'}, inplace=True)
        if 'Unnamed: 0' in daily_df.columns:
            daily_df.drop(columns=['Unnamed: 0'], inplace=True)
        return daily_df

    def win_rate(self,daily_df):

        # 胜率,
        # Convert date column to datetime format
        daily_df['date'] = pd.to_datetime(daily_df['date'], format='%Y%m%d')

        # Calculate daily metrics
        total_days = daily_df['daily_pnl_sum'].ne(0).sum()  # Count of non-zero PnL days
        win_days = daily_df['daily_pnl_sum'].gt(0).sum()  # Count of profitable days
        daily_win_rate = win_days / total_days if total_days > 0 else 0
        daily_expected_return = daily_df['daily_yields'].mean()

        positive_yields_mean = daily_df.loc[daily_df['daily_yields'] > 0, 'daily_yields'].mean()
        negative_yields_mean = daily_df.loc[daily_df['daily_yields'] < 0, 'daily_yields'].mean()

        # Resample to calculate weekly metrics
        weekly_agg = daily_df.resample('W-MON', on='date').agg(
            weekly_pnl_sum=('daily_pnl_sum', 'sum'),
            weekly_yields_mean=('daily_yields', 'mean')
        ).reset_index()

        total_weeks = weekly_agg['weekly_pnl_sum'].ne(0).sum()  # Count of non-zero PnL weeks
        win_weeks = weekly_agg['weekly_pnl_sum'].gt(0).sum()  # Count of profitable weeks
        weekly_win_rate = win_weeks / total_weeks if total_weeks > 0 else 0
        weekly_expected_return = weekly_agg['weekly_yields_mean'].mean()

        # Create summary DataFrame
        summary = pd.DataFrame({
            'daily_win_rate': [daily_win_rate],
            'daily_expected_return': [daily_expected_return],
            'positive_expected_return': [positive_yields_mean],
            'negative_expected_return': [negative_yields_mean],
            'weekly_win_rate': [weekly_win_rate],
            'weekly_expected_return': [weekly_expected_return]
        })

        return summary

    def calculate_trading_frequency(self):
        df = self.df
        # 筛选roll_signal等于0的行
        filtered_df = df[df['roll_signal'] == 0]

        # 筛选operation不等于0的交易
        non_zero_operations = filtered_df[filtered_df['operation'] != 0]

        # 计算总交易数和不同交易日的数目
        total_trades = non_zero_operations.shape[0]
        total_days = filtered_df['date'].nunique()

        # 计算交易频率，一天交易多少次
        trades_per_day = total_trades / total_days if total_days > 0 else 0
        # 平均持仓天数（交易日）
        days_hold = 1 / trades_per_day if trades_per_day > 0 else float('inf')

        frequency = pd.DataFrame({
            'average_trades_per_day': [trades_per_day],
            'average_days_hold': [days_hold]
        })
        return frequency

    def plot_net_value_and_drawdown_html(self, daily_df, drawdown_df):
        # 使用plotly绘制交互式净值曲线
        fig = go.Figure()

        # 绘制 actual_net_value2 和 theo_net_value2 曲线
        fig.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['actual_net_value2'],
                                 mode='lines', name='Actual Net Value', line=dict(color='#32A1DE')))
        fig.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['theo_net_value2'],
                                 mode='lines', name='Theo Net Value', line=dict(color='#ED8C45')))

        # 绘制 current_drawdown 曲线和填充区域
        fig.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['current_drawdown'],
                                 mode='lines', name='Current Drawdown', line=dict(color='#17F2AB'),
                                 yaxis="y2"))
        fig.add_trace(go.Scatter(x=daily_df['date'], y=[0] * len(daily_df['date']),
                                 mode='lines', line=dict(color='rgba(0,0,0,0)'),
                                 fill='tonexty', fillcolor='rgba(23,242,171,0.5)', yaxis="y2"))

        # 标记 drawdown_df 中的最大回撤区间
        for _, row in drawdown_df.iterrows():
            fig.add_vline(x=pd.to_datetime(row['start_date'], format='%Y%m%d'), line=dict(color='gray', dash='dash'))
            fig.add_vline(x=pd.to_datetime(row['end_date'], format='%Y%m%d'), line=dict(color='gray', dash='dash'))
            fig.add_vrect(x0=pd.to_datetime(row['start_date'], format='%Y%m%d'),
                          x1=pd.to_datetime(row['end_date'], format='%Y%m%d'),
                          fillcolor='gray', opacity=0.15, line_width=0)

        # 设置图表布局
        # fig.update_layout(
        #     title='Net Value',
        #     xaxis_title='Date',
        #     yaxis_title='Net Value',
        #     yaxis2=dict(title='Drawdown', overlaying='y', side='right', range=[-1, 0]),
        #     font=dict(family="Arial, sans-serif", size=12),
        # )
        fig.update_layout(
            title=dict(text="Net Value Curve", font=dict(size=18)),  # 修改标题样式
            xaxis_title='Date',
            yaxis_title='Net Value',
            yaxis2=dict(title='Drawdown', overlaying='y', side='right', range=[-1, 0]),
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='rgba(245, 245, 245, 1)',  # 调浅底色
            legend=dict(
                x=0.85, y=1,
                traceorder='normal',
                font=dict(size=10),
                bgcolor='rgba(0,0,0,0)'
            )
        )
        return fig

    # 生成 HTML 文件
    # 图表加滚轮，比率数据保留四位小数
    def generate_html(self, daily_df, drawdown_df, ratio_df, frequency, win, ):
        # 绘制图表
        fig = self.plot_net_value_and_drawdown_html(daily_df, drawdown_df)
        fig_html = pio.to_html(fig, full_html=False)

        # 构造HTML文件的内容
        html_content = f"""
        <html>
        <head>
            <title>Net Value Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                h1 {{ text-align: center; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .indicator-section {{ margin: 30px 0; }}
                .chinese {{ font-family: "Microsoft YaHei", sans-serif; }}
            </style>
        </head>
        <body>
            <h1>Net Value Report</h1>
            {fig_html}

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
        with open("net_value_report.html", "w", encoding="utf-8") as file:
            file.write(html_content)

        print("HTML report generated successfully.")
    def run_backtest(self):
        '''
        执行完整的回测流程，返回最终的df
        '''
        self.calculate_operation()
        contract_df = self.determine_trade_contracts()
        self.df = self.process_trades(contract_df)
        self.df = self.query_trade_prices()
        self.df = self.add_snapshoot()
        self.df = self.add_settle_prc()
        self.df = self.correct_trade()
        self.df = self.fill_number()
        self.df = self.slippage()
        self.df = self.operation_type()
        self.df = self.calculate_pnl()
        self.df = self.calculate_margin_and_free_capital()
        self.df = self.net_value_change()
        self.df = self.net_value()
        self.df = self.slippage_rounding_revenue()
        self.df = self.yields()
        self.df = self.calculate_roll_revenue()
        self.df = self.drawdown()
        net_value_df = self.net_value_table()
        drawdown_df = self.find_top_drawdowns()
        daily_df = self.get_daily_df()
        ratio_df = self.calculate_ratios()
        win_df = self.win_rate(daily_df)
        frequency_df = self.calculate_trading_frequency()
        self.generate_html(self.df, drawdown_df, ratio_df, frequency_df, win_df)



        return self.df, net_value_df


if __name__ == '__main__':
    signal_df = pd.read_csv('001.csv')
    bt = backtest('BR', 1000000, 2, signal_df, '20220101', '20240630')
    df, net_value_df, html_content = bt.run_backtest()


