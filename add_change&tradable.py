import re
import warnings
import os
import pandas as pd
import glob
import os


# 给所有snapshot添加pre_settle, settle, limit, change列
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
    if contract.endswith('.GFE'):
        corrected_contract = contract.replace('.GFE', '.GFEX')
        return corrected_contract

    return contract


def add_limit_column(merged_df, limit_df):
    '''
    给snapshoot文件的数据每天添加涨跌停标签。1-可交易 2-涨停 3-跌停 4-行情缺失
从组内数据库的daybar数据中读取对应的settle和pre_settle价格数据，
Limit（涨跌幅限额）从\\samba-1.quantchina.pro\quanyi4g\temporary\Steiner\data_wash\linux_so\py311\limit.csv中读取，此为wind表格Cfuturespricechangelimit，里面记录了某合约从某天开始的涨跌停限额。
如果该合约涨跌停限额数据缺失，则选择对应品种的第一个数值替代。
输入：snapshot数据文件夹。
输出：新的snapshot文件。
    '''
    # 格式化日期为字符串格式 'yyyymmdd' 以确保匹配
    merged_df['date'] = merged_df['date'].astype(str)
    limit_df['CHANGE_DT'] = limit_df['CHANGE_DT'].astype(str)

    # 合并数据框，使用日期和合约代码作为键
    merged_df = pd.merge(
        merged_df,
        limit_df[['S_INFO_WINDCODE', 'CHANGE_DT', 'PCT_CHG_limit']],
        left_on=['contract', 'date'],
        right_on=['S_INFO_WINDCODE', 'CHANGE_DT'],
        how='left'
    )

    # 删除多余的合并键
    merged_df.drop(columns=['S_INFO_WINDCODE', 'CHANGE_DT'], inplace=True)


    merged_df['PCT_CHG_limit'].fillna(method='ffill', inplace=True)

    # 填充剩余空值：对每个合约，使用第一个非空值进行填充
    def fill_first_pct_chg_limit(row, limit_df,unmatched_contracts):
        if pd.isna(row['PCT_CHG_limit']):
            matching_rows = limit_df.loc[limit_df['S_INFO_WINDCODE'] == row['contract'], 'PCT_CHG_limit']
            if not matching_rows.empty:
                return matching_rows.iloc[0]
            else:
                unmatched_contracts.append(row['contract'])
                return pd.NA  # 如果找不到匹配项，返回 NaN
        return row['PCT_CHG_limit']

    merged_df['PCT_CHG_limit'].fillna(method='bfill', inplace=True)
    unmatched_contracts=[]
    merged_df['PCT_CHG_limit'] = merged_df.apply(lambda row: fill_first_pct_chg_limit(row, limit_df,unmatched_contracts), axis=1)

    if unmatched_contracts:
        print(f"未找到匹配项的 contract: {set(unmatched_contracts)}")
    # 将所有limit列的值除以100
    merged_df['PCT_CHG_limit'] = merged_df['PCT_CHG_limit'] / 100

    # 将列重命名为 'limit'
    merged_df.rename(columns={'PCT_CHG_limit': 'limit'}, inplace=True)

    return merged_df


def add_tradable(df):
    df['tradable'] = 1

    # 根据条件设置 'tradable' 值

    df.loc[df['change'] <= -df['limit']+0.002, 'tradable'] = 3
    df.loc[df['change'] >= df['limit']-0.002, 'tradable'] = 2
    df.loc[df['query_notrade'] == 1, 'tradable'] = 4
    return df


# Directories
snapshot_dir = r'Z:\temporary\Steiner\data_wash\linux_so\py311\snapshot_results_index'
daybar_dir = r'Z:\data\future\daybar'
output_dir = r'Z:\temporary\Steiner\data_wash\linux_so\py311\snapshot_temp'

limit = pd.read_csv('limit.csv')
limit['S_INFO_WINDCODE'] = limit.apply(lambda row: correct_czc_code(row['S_INFO_WINDCODE'], row['CHANGE_DT']),
                                       axis=1)
# List all files in snapshot directory
snapshot_files = [f for f in os.listdir(snapshot_dir) if f.endswith('_results.csv')]

for snapshot_file in snapshot_files:
    # Read the snapshot (A) file
    snapshot_df = pd.read_csv(os.path.join(snapshot_dir, snapshot_file))
    snapshot_df['trading_date'] = pd.to_datetime(snapshot_df['trading_date'].astype(str), format='%Y-%m-%d')
    snapshot_df['trading_date'] = snapshot_df['trading_date'].dt.date

    # Generate the corresponding daybar (B) file name
    base_name = snapshot_file.replace('_results.csv', '_daybar.csv')
    daybar_file = base_name.replace('_results', '_daybar')
    daybar_path = os.path.join(daybar_dir, daybar_file)

    if os.path.exists(daybar_path):
        # Read the daybar (B) file
        daybar_df = pd.read_csv(daybar_path, engine='python', encoding='utf-8')

        # Filter required columns: S_INFO_WINDCODE, S_DQ_SETTLE, trading_date
        daybar_df = daybar_df[['S_INFO_WINDCODE', 'S_DQ_SETTLE', 'S_DQ_PRESETTLE', 'TRADE_DT']]

        # Rename columns to match the snapshot dataframe
        daybar_df.rename(columns={'S_INFO_WINDCODE': 'contract', 'S_DQ_SETTLE': 'settle', 'TRADE_DT': 'trading_date',
                                  'S_DQ_PRESETTLE': 'pre_settle'}, inplace=True)

        daybar_df['trading_date'] = pd.to_datetime(daybar_df['trading_date'].astype(str), format='%Y%m%d')
        daybar_df['trading_date'] = daybar_df['trading_date'].dt.date

        daybar_df['contract'] = daybar_df.apply(lambda row: correct_czc_code(row['contract'], row['trading_date']),
                                                axis=1)

        # Sort daybar dataframe by contract and trading_date
        daybar_df.sort_values(by=['contract', 'trading_date'], inplace=True)

        # daybar_df['pre_settle'] = daybar_df.groupby('contract')['settle'].shift(1)

        # Merge the daybar data into the snapshot dataframe based on contract and trading_date
        merged_df = pd.merge(snapshot_df, daybar_df[['contract', 'trading_date', 'settle', 'pre_settle']],
                             on=['contract', 'trading_date'], how='left')


        # Calculate the change column as last_prc / pre_settle
        merged_df['change'] = merged_df['last_prc'] / merged_df['pre_settle'] - 1

        merged_df = add_limit_column(merged_df, limit)

        merged_df = add_tradable(merged_df)

        # Write the result to the output directory with the same file name
        output_path = os.path.join(output_dir, snapshot_file)
        merged_df.to_csv(output_path, index=False)

        # 添加limit列

        print(f"Processed and saved: {output_path}")
    else:
        print(f"Daybar file not found: {daybar_file}")






