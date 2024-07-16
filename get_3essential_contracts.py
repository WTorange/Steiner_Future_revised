import pandas as pd
import re
import warnings
import os
from connect_wind import ConnectDatabase

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

class get_essential_contract(ConnectDatabase):

    def __init__(self, future_symbol, type: str, start_date: str, end_date: str):
        self.future_symbol = future_symbol  #例 铜CU(大写)
        self.type = type #com商品， ind股指， bond国债
        self.start_date = start_date
        self.end_date = end_date
        if type == 'com':
            self.table = 'CCOMMODITYFUTURESEODPRICES'
        elif type == 'ind':
            self.table = 'CINDEXFUTURESEODPRICES'
        elif type == 'bond':
            self.table = 'CBONDFUTURESEODPRICES'
# 这里不对，应该用S_DQ_OI 持仓量（手）。确实是用日行情数据。遍历交易日期————找到所有的对应品种的合约代码————找到持仓量最大的前三名合约，但是没有到七月份（可以通过合约代码的数字判断，数字大的是远月合约）。
        self.sql = f'''
                    SELECT S_INFO_WINDCODE, TRADE_DT, S_DQ_OI
                    FROM {self.table}
                    WHERE TRADE_DT between {start_date} and {end_date}
                    AND S_INFO_WINDCODE LIKE '{future_symbol}%'
                    '''
        super().__init__(self.sql)
        self.df = super().get_data()

        # self.df.sort_values(by='date', inplace=True)
        # self.df.reset_index(drop=True, inplace=True)

    def process(self):
        # 将 S_DQ_OI 转换为数值型
        self.df['S_DQ_OI'] = pd.to_numeric(self.df['S_DQ_OI'], errors='coerce')
        mask = self.df['S_INFO_WINDCODE'].str.match(f'^({"|".join(symbol_list)})\d{{3,4}}\.\w+$')
        filtered_df = self.df[mask]
        # print(filtered_df)
        # top3_per_day = filtered_df.groupby('TRADE_DT').apply(lambda x: x.nlargest(3, 'S_DQ_OI')).reset_index(drop=True)

        top3_per_day = filtered_df.sort_values(by=['S_DQ_OI', 'S_INFO_WINDCODE'], ascending=[False, False]) \
                                 .groupby('TRADE_DT') \
                                 .apply(lambda x: x.head(3)) \
                                 .reset_index(drop=True)

        output_df = pd.DataFrame(columns=['future_code', 'date', 'main_contract', 'second', 'third'])
        for date, group in top3_per_day.groupby('TRADE_DT'):
            top3_codes = list(group['S_INFO_WINDCODE'].iloc[:3])
            output_df = pd.concat([
                output_df,
                pd.DataFrame({
                    'future_code': [self.future_symbol],
                    'date': [date],
                    'main_contract': [top3_codes[0] if len(top3_codes) > 0 else None],
                    'second': [top3_codes[1] if len(top3_codes) > 1 else None],
                    'third': [top3_codes[2] if len(top3_codes) > 2 else None]
                })
            ], ignore_index=True)

        return output_df

    # def get_main_and_secondary_contracts(self, group):
    #     # 假设已经按持仓量排序
    #     main_contract = group.iloc[0]['S_INFO_WINDCODE']
    #     second_contract = group.iloc[1]['S_INFO_WINDCODE'] if len(group) > 1 else None
    #     third_contract = group.iloc[2]['S_INFO_WINDCODE'] if len(group) > 2 else None
    #     return pd.Series([main_contract, second_contract, third_contract], index=['main_contract', 'second', 'third'])

    def run(self):
        df = self.process()
        df['future'] = self.future_symbol

        return df

if __name__ == '__main__':
    symbol_df = pd.read_csv(r"C:\Users\maki\Desktop\quantchina\Futures-main\data_wash\品种列表.csv")

    # 提取品种代码并去除.后的字符
    symbol_list = symbol_df['S_INFO_WINDCODE'].apply(lambda x: x.split('.')[0]).tolist()
    output_folder = r'C:\Users\maki\Desktop\quantchina\Futures-main\data_wash\essentialcontract'
    os.makedirs(output_folder, exist_ok=True)
    start_date = '20091201'
    end_date = '20240710'
    #
    for type in ['com', 'ind', 'bond']:
        for symbol in symbol_list:
            mc = get_essential_contract(symbol, type, start_date=start_date, end_date=end_date)
            temp = mc.run()
            print(temp)
            if temp is None or temp.empty:
                print(f"No data for {symbol}, skipping.")
                continue

            file_name = f"{symbol}_{start_date}_{end_date}.csv"
            file_path = os.path.join(output_folder, file_name)
            temp.to_csv(file_path, index=False)
            print(f"Saved file: {file_path}")


    # get2 = get_essential_contract('IM', 'ind', '20230103', '20230110')
    # print(get2.run())

    # get3= get_essential_contract('T', 'bond', '20230103', '20230110')
    # print(get3.run())



