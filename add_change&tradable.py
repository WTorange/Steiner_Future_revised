
import re
import warnings
import os
import pandas as pd
import glob
import os



def correct_czc_code(contract, trading_date):
    if contract.endswith('.CZC'):
        # 提取期货代码中的字母和数字部分
        match = re.match(r'^([A-Z]+)(\d+)\.CZC$', contract)
        if match:
            letters, numbers = match.groups()
            # 提取查询时间的年份
            string_date = trading_date.strftime('%Y')
            year = int(string_date)
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
# Directories
snapshoot_dir = r'Z:\temporary\Steiner\data_wash\linux_so\py311\snapshoot_results_oi'
daybar_dir = r'Z:\data\future\daybar'
output_dir = r'Z:\temporary\Steiner\data_wash\linux_so\py311\snapshoot_temp'

# List all files in snapshoot directory
snapshoot_files = [f for f in os.listdir(snapshoot_dir) if f.endswith('_results.csv')]

for snapshoot_file in snapshoot_files:
    # Read the snapshot (A) file
    snapshoot_df = pd.read_csv(os.path.join(snapshoot_dir, snapshoot_file))
    snapshoot_df['trading_date'] = pd.to_datetime(snapshoot_df['trading_date'].astype(str), format='%Y-%m-%d')
    snapshoot_df['trading_date'] = snapshoot_df['trading_date'].dt.date

    # Generate the corresponding daybar (B) file name
    base_name = snapshoot_file.replace('_results.csv', '_daybar.csv')
    daybar_file = base_name.replace('_results', '_daybar')
    daybar_path = os.path.join(daybar_dir, daybar_file)

    if os.path.exists(daybar_path):
        # Read the daybar (B) file
        daybar_df = pd.read_csv(daybar_path, engine='python', encoding='utf-8')

        # Filter required columns: S_INFO_WINDCODE, S_DQ_SETTLE, trading_date
        daybar_df = daybar_df[['S_INFO_WINDCODE', 'S_DQ_SETTLE', 'TRADE_DT']]

        # Rename columns to match the snapshot dataframe
        daybar_df.rename(columns={'S_INFO_WINDCODE': 'contract', 'S_DQ_SETTLE': 'settle', 'TRADE_DT':'trading_date'}, inplace=True)

        daybar_df['trading_date'] = pd.to_datetime(daybar_df['trading_date'].astype(str), format='%Y%m%d')
        daybar_df['trading_date'] = daybar_df['trading_date'].dt.date

        daybar_df['contract'] = daybar_df.apply(lambda row: correct_czc_code(row['contract'], row['trading_date']),
                                                axis=1)



        # Sort daybar dataframe by contract and trading_date
        daybar_df.sort_values(by=['contract', 'trading_date'], inplace=True)

        # Create a pre_settle column with previous day's settle value
        daybar_df['pre_settle'] = daybar_df.groupby('contract')['settle'].shift(1)

        # Merge the daybar data into the snapshoot dataframe based on contract and trading_date
        merged_df = pd.merge(snapshoot_df, daybar_df[['contract', 'trading_date', 'settle', 'pre_settle']],
                             on=['contract', 'trading_date'], how='left')

        # Calculate the change column as last_prc / pre_settle
        merged_df['change'] = merged_df['last_prc'] / merged_df['pre_settle']-1

        # Write the result to the output directory with the same file name
        output_path = os.path.join(output_dir, snapshoot_file)
        merged_df.to_csv(output_path, index=False)

        print(f"Processed and saved: {output_path}")
    else:
        print(f"Daybar file not found: {daybar_file}")



df = pd.read_csv(r"Z:\temporary\Steiner\data_wash\linux_so\py311\snapshoot_temp\BR_20091201_20240710_results.csv")
limit = 0.05

# 初始化 'tradable' 列为 1
df['tradable'] = 1

# 根据条件设置 'tradable' 值

df.loc[df['change'] <= -limit, 'tradable'] = 3
df.loc[df['change'] >= limit, 'tradable'] = 2
df.loc[df['query_notrade'] == 1, 'tradable'] = 4

df.to_csv('tradable.csv')



