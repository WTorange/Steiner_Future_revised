# 运行第二种方法的回测，固定持有期N天，
# 为了避免特定日期买入带来的偶然性，N天中每天调仓目标仓位的1/N,与前9天的仓位相加变成这一天的目标仓位。
# 修改daily信号

import os
import pandas as pd
from backtest_new import backtest


def generate_momentum_signal(df,N=70,cs=False):
    if df.iloc[:, 1].isna().all():
        return pd.DataFrame(columns=['date', 'position']), False
    if 'TRADE_DT' in df.columns:
        df = df.rename(columns={'TRADE_DT': 'trading_date'})
    df['trading_date'] = pd.to_datetime(df['trading_date'],format='%Y-%m-%d',errors='coerce')
    df = df[df['trading_date'] >= pd.Timestamp('2014-01-01')]
    if df.iloc[:, 1].isna().all():
        return pd.DataFrame(columns=['date', 'position']), False
    df = df.dropna(subset=[df.columns[1]])  # 去除第三列有缺失值的行
    df['daynight'] = 'day'

    # 初始化 position 为 0
    position = pd.Series(0, index=df.index)

    if cs:
        print('cs')
        # 当cs为True时截面信号的新规则
        position[(df.iloc[:, 1] > 0.3)] = 1
        position[(df.iloc[:, 1] < -0.3)] = -1
        df['position'] = position
    else:
        # 时序规则
        position[(df.iloc[:, 1] > 0)] = 1
        position[(df.iloc[:, 1] < 0)] = -1
        df['position'] = position
        absolute_values = df.iloc[:, 1].abs()
        threshold = absolute_values.quantile(0.1)
        df.loc[absolute_values <= threshold, 'position'] = 0
    # 添加 position 列

    df['position'] = df['position'] / N
    df['position'] = df['position'].rolling(window=N, min_periods=1).sum()

    df.rename(columns={'trading_date': 'date'}, inplace=True)
    df = df[['date', 'position', 'daynight']]

    return df, True # 返回包含日期和仓位的DataFrame

def get_many_signals(symbol, input_folder, initial_capital, leverage, output_folder, N=70,cs=False):
    '''
    通过读取因子值，设定规则，输出信号，传达因子名称，给Backtest_new进行回测
    需要from backtest_new import backtest
    输入：持有期N，因子值df（包含trading_date和因子值列）（包含所有因子值的文件夹），cs参数（是否是截面回测，默认False），资金量，杠杆，输出路径。
    可根据输入因子文件读取因子名，要回测的品种代码
    输出：summary和detail的每天交易净值表格，html报告，夏普率（可删除）

    '''
    os.makedirs(output_folder, exist_ok=True)  # 确保输出文件夹存在

    # 遍历文件夹中的所有CSV文件
    for file_name in sorted(os.listdir(input_folder)):
        if file_name.startswith(('FB_', 'BB_','JR','ME','ER','RO')):
            continue
        if file_name.endswith('.csv') and file_name.startswith(f'{symbol}_'):
            file_path = os.path.join(input_folder, file_name)
            print(file_path)
            # 提取symbol (第一个下划线之前的部分)
            symbol = file_name.split('_')[0]
            factor_name = file_name.split('.')[0]
            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 生成signal_df
            signal_df, should_process = generate_momentum_signal(df,N,cs=cs)
            if not should_process:
                print(f"Skipping {file_name} as the third column is entirely NaN.")
                continue  # 跳过这个文件，继续处理下一个文件
            if file_name.startswith(('NI_')):
                signal_df = signal_df[signal_df['date'] >= pd.Timestamp('2016-01-01')]

            if file_name.startswith(('FU_')):
                signal_df = signal_df[signal_df['date'] >= pd.Timestamp('2018-08-01')]

            first_row = signal_df .iloc[0].copy()
            first_row['position'] = 0  # 将 position 设为 0
            signal_df = pd.concat([pd.DataFrame([first_row]), signal_df], ignore_index=True)
            # 创建回测对象
            bt = backtest(symbol, initial_capital, leverage, signal_df, factor_name=factor_name,output_path=output_folder)

            result_df, net_value_df = bt.run_backtest()

            # 保存结果文件
            detail_output_path = os.path.join(output_folder, f"{factor_name}_{N}dhold_detail.csv")
            summary_output_path = os.path.join(output_folder, f"{factor_name}_{N}dhold_summary.csv")

            result_df.to_csv(detail_output_path, index=False)
            net_value_df.to_csv(summary_output_path, index=False)





if __name__ == '__main__':
    # 读取信号数据
    input_folder_barra = "/nas92/data/future/factor/ILLIQ/cs/all/ILLIQ_cs"

    symbol_list = ['IC', 'IF', 'IH', 'IM']
    N_list = [1, 3, 5, 10]
    initial_capital = 10000000
    leverage = 1
    for symbol in symbol_list:
        for N in N_list:
            output_folder_barra = f"/nas92/data/future/factor/ILLIQ/reports/stock_index_futures/{symbol}/{N}/"
            get_many_signals(symbol, input_folder_barra, initial_capital, leverage, output_folder_barra, N=N, cs=True)

