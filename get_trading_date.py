import pandas as pd
import glob
import os

'''
当wap数据中没有trading_date只有自然日的数据时，可以通过此程序从快照数据中匹配trading_date数据添加到wap的csv中，方便后续操作。
输入：wap文件夹，snapshot文件夹，输出文件夹
输出：匹配trading_date之后的wap数据
'''

# 路径定义
wap_path = r'Z:\temporary\Steiner\data_wash\linux_so\py311\wap_results_oi'
snapshot_path = r'Z:\temporary\Steiner\data_wash\linux_so\py311\snapshot_results_oi'
temp_path = r'Z:\temporary\Steiner\data_wash\linux_so\py311\temp'

# 获取所有文件列表
wap_files = glob.glob(os.path.join(wap_path, '*.csv'))
snapshot_files = glob.glob(os.path.join(snapshot_path, '*.csv'))

# 读取snapshot文件并存入字典，键为文件名（不含路径和扩展名）
snapshot_dict = {os.path.basename(f).replace('.csv', ''): pd.read_csv(f) for f in snapshot_files}


def process_file(wap_file):

    file_name = os.path.basename(wap_file).replace('.csv', '')
    print(f"Processing {file_name}.csv")

    wap_df = pd.read_csv(wap_file)

    # 检查是否有相应的snapshot文件
    if file_name in snapshot_dict:
        snapshot_df = snapshot_dict[file_name]

        # 合并数据，根据条件匹配trading_date
        merged_df = wap_df.merge(snapshot_df[['query_time', 'trading_date']],
                                 left_on='start_time', right_on='query_time',
                                 how='left')

        # 删除多余的query_time列
        merged_df.drop(columns=['query_time'], inplace=True)

        # 保存到temp目录
        output_file = os.path.join(temp_path, f"{file_name}.csv")
        merged_df.to_csv(output_file, index=False)


# 使用列表解析批量处理并打印进度
[process_file(f) for f in wap_files]