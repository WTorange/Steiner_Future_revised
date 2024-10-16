import os
import glob
import pandas as pd
import numpy as np
import  re




def cs_momentum(type,symbol_list,N):
    input_folder = f'/nas92/data/future/factor/ILLIQ/{type}_daily'
    # input_folder = rf'\\samba-1.quantchina.pro\quanyi4g\data\future\factor\momentum\{type}_daily_momentum'
    all_files = glob.glob(os.path.join(input_folder, "*.csv"))

    matching_files = [
        file for file in all_files
        if
           any(os.path.basename(file).startswith(symbol + "_") for symbol in symbol_list)
            and f'_{N}days' in os.path.basename(file)
    ]
    # matching_files = [
    #     file for file in all_files
    #     if
    #        any(os.path.basename(file).startswith(symbol + "_") for symbol in symbol_list)
    # ]
    output_folder1 = "/nas92/data/future/factor/term_structure/cs/rank_n_weight"
    # output_folder1 = rf'\\samba-1.quantchina.pro\quanyi4g\data\future\factor\momentum\cs_momentum\rank_n_weight'

    output_folder2 = f"/nas92/data/future/factor/ILLIQ/cs/financial_futures/{type}_cs"
    # output_folder2 = rf'\\samba-1.quantchina.pro\quanyi4g\data\future\factor\momentum\cs_momentum\{type}_cs_momentum'
    # 初始化空的DataFrame以存储合并后的数据
    data_frames = []

    # 读取每个CSV文件并合并
    for file in matching_files:

        df = pd.read_csv(file)
        df['trading_date'] = pd.to_datetime(df['trading_date'], format='%Y-%m-%d')
        df.set_index('trading_date', inplace=True)
        # 重命名列名为第一个下划线之前的字母
        new_col_name = df.columns[0].split('_')[0]
        if new_col_name in [col for df in data_frames for col in df.columns]:
            print(f"警告：列名 '{new_col_name}' 在不同文件中重复，文件 {file} 中可能有冲突，正在处理...")
            # 如果列名重复，给列名添加唯一后缀
            new_col_name = f"{new_col_name}_{len(data_frames)}"
            print(new_col_name)
        df.rename(columns={df.columns[0]: new_col_name}, inplace=True)


        data_frames.append(df)
    cs_df = pd.concat(data_frames, axis=1)

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
            weights = valid_ranks.apply(lambda rank: rank/((1 + s_t) / 2) - 1)

            weight_df.loc[date, weights.index] = weights

    # print(weight_df)
    # 将trading_date列重置为第一列
    rank_df.reset_index(inplace=True)
    weight_df.reset_index(inplace=True)
    #
    for symbol in weight_df.columns:
        if symbol != 'trading_date':  # 忽略 'trading_date' 列
            symbol_df = weight_df[['trading_date', symbol]].dropna()  # 删除NaN值的行
            output_file = os.path.join(output_folder2, f"{symbol}_{type}_{N}days_cs.csv")
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
    rank_output_file = os.path.join(output_folder1, f"rank_{type}_factor.csv")
    weight_output_file = os.path.join(output_folder1, f"weight_{type}_factor.csv")

    rank_df.to_csv(rank_output_file, index=False)
    weight_df.to_csv(weight_output_file, index=False)
    print(f"Rank and weight data saved to {output_folder1}")


N_list = [5, 10, 20, 60, 120, 252]
type_list = ['ILLIQ']
symbol_list = ['IC','IF','IH','IM','T','TF','TS']
symbol_list2 = ["A", "AG", "AL", "AO", "AP", "AU", "B", "BB", "BC", "BR", "BU", "C", "CF", 'CJ', 'CS', 'CU', 'CY', 'EB',
               'EC', 'EG', 'ER', 'FB', 'FG', 'FU', 'HC', 'I', 'IC',
               'IF', 'IH', 'IM', 'J', 'JD', 'JM', 'JR', 'L', 'LH', 'LU', 'M', 'MA', 'ME', 'NI', 'NR', 'OI', 'P', 'PB',
               'PF', 'PG', 'PK', 'PM', 'PP', 'PX', 'RB', 'RI', 'RM', 'RO',
               'RR', 'RS', 'RU', 'SA', 'SC', 'SF', 'SH', 'SM', 'SN', 'SP', 'SR', 'SS', 'T', 'TA', 'TC', 'TF', 'TL',
               'TS', 'UR', 'V', 'WH', 'WR', 'WS', 'WT', 'Y', 'ZC', 'ZN']
for type in type_list:
    for N in N_list:
        cs_momentum(type, symbol_list,N)
