import pandas as pd
import re
import warnings

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
            self.table = 'CCOMMODITYFUTURESPOSITIONS'
        elif type == 'ind':
            self.table = 'CINDEXFUTURESPOSITIONS'
        elif type == 'bond':
            self.table = 'CBONDFUTURESPOSITIONS'

        self.sql = f'''
                    SELECT S_INFO_WINDCODE, TRADE_DT, FS_INFO_TYPE, fs_info_positionsnum
                    FROM {self.table}
                    WHERE TRADE_DT between {start_date} and {end_date}
                    AND S_INFO_WINDCODE like '{future_symbol}____.___'
                    '''
        super().__init__(self.sql)
        self.df = super().get_data()
        # self.df.sort_values(by='date', inplace=True)
        # self.df.reset_index(drop=True, inplace=True)

    def process(self):
        temp= self.df.rename(columns={'S_INFO_WINDCODE': 'symbol',
                                          'TRADE_DT': 'date',
                                          'FS_INFO_TYPE': 'type',
                                          'fs_info_positionsnum': 'position'})
        temp['date'] = temp['date'].astype(int)
        temp['type'] = temp['type'].astype(int)

        temp1 = temp.groupby(['symbol', 'date', 'type'])['position'].sum().reset_index()
        mark = (temp1['type'] == 2) | (temp1['type'] == 3)
        temp2 = temp1.loc[mark].groupby(['symbol', 'date'])['position'].sum().reset_index()
        temp2 = temp2.rename(columns={'position': 'open_interest'})
        # temp2.drop('type', axis=1, inplace=True)

        temp3 = temp1.loc[temp1['type'] == 1]
        del temp3['type']

        result = pd.merge(temp2, temp3, on=['symbol', 'date'], how='outer')
        # result.dropna(inplace=True)
        # result['open_interest', 'position'] = result['open_interest', 'position'].fillna(0)
        result = result.sort_values(by='date')
        result.reset_index(drop=True, inplace=True)

        return result

    def get_main_and_secondary_contracts(self, df):
        sorted_df = df.sort_values(by=['open_interest', 'position'], ascending=[False, False])
        max_open_interest = sorted_df.iloc[0]['open_interest']
        max_position = sorted_df.iloc[0]['position']
        main_contracts = sorted_df[(sorted_df['open_interest'] == max_open_interest) & (sorted_df['position'] == max_position)]
        main_contract = main_contracts.iloc[0]['symbol'] if len(main_contracts) > 0 else None

        if len(main_contracts) > 1:
            main_contracts['month'] = main_contracts['symbol'].apply(lambda x: int(re.findall(r'\d+', x)))
            sorted_main_contracts = main_contracts.sort_values(by='month', ascending=False)
            main_contract = sorted_main_contracts.iloc[0]['symbol']

        secondary_contracts = sorted_df[
            (sorted_df['open_interest'] < max_open_interest) | (sorted_df['position'] < max_position)]
        secondary_contract = secondary_contracts.iloc[0]['symbol'] if len(secondary_contracts) > 0 else None

        return pd.Series([main_contract, secondary_contract])

    def run(self):
        df = self.process()
        df['future'] = self.future_symbol
        main_and_sec_contract = df.groupby(['date', 'future']).apply(self.get_main_and_secondary_contracts)
        main_and_sec_contract.columns = ['main_contract', 'secondary_contract']

        main_and_sec_contract.reset_index(inplace=True)
        main_and_sec_contract.sort_values(['future', 'date'])
        # 下面这里有问题，已经注释并且修改

        # main_and_sec_contract['main_switch'] = main_and_sec_contract.groupby('future')['main_contract'].apply(lambda x: x.shift() != x).astype(int)
        # main_and_sec_contract['secondary_switch'] = main_and_sec_contract.groupby('future')['secondary_contract'].apply(lambda x: x.shift() != x).astype(int)
        main_and_sec_contract['main_switch'] = main_and_sec_contract.groupby('future')['main_contract'].transform(
            lambda x: x.shift() != x).astype(int)
        main_and_sec_contract['secondary_switch'] = main_and_sec_contract.groupby('future')[
            'secondary_contract'].transform(lambda x: x.shift() != x).astype(int)

        # main_and_sec_contract['main_switch'][0] = 0
        # main_and_sec_contract['secondary_switch'][0] = 0
        main_and_sec_contract['main_switch'].iloc[0] = 0
        main_and_sec_contract['secondary_switch'].iloc[0] = 0

        return main_and_sec_contract[['future', 'date', 'main_contract', 'secondary_contract', 'main_switch', 'secondary_switch']]


if __name__ == '__main__':
    get1 = get_essential_contract('RU', 'com', '20230103', '20230110')
    print(get1.run())

    # get2 = get_essential_contract('IM', 'ind', '20230103', '20230110')
    # print(get2.run())

    # get3= get_essential_contract('T', 'bond', '20230103', '20230110')
    # print(get3.run())



