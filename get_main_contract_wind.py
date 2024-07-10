import pandas as pd
import os as os

from connect_wind import ConnectDatabase

# 获取历史主力对应月合约
class get_main_contract(ConnectDatabase):

    def __init__(self, future_symbol, start_date: str, end_date: str):
        self.future_symbol = future_symbol  #例 铜CU(大写)
        self.start_date = start_date
        self.end_date = end_date
        self.sql = f'''
                    SELECT S_INFO_WINDCODE as symbol, 
                           FS_MAPPING_WINDCODE as contract, 
                           STARTDATE as start, 
                           ENDDATE as end
                    FROM CFUTURESCONTRACTMAPPING
                    WHERE ((STARTDATE BETWEEN '{self.start_date}' AND '{self.end_date}') 
                    OR (ENDDATE BETWEEN '{self.start_date}' AND '{self.end_date}'))
                    AND S_INFO_WINDCODE LIKE '{future_symbol}.%'
                    AND S_INFO_TYPE = 706006000
                    '''

        super().__init__(self.sql)
        self.df = super().get_data()
        self.df = self.df.rename(columns={'symbol': 'symbol', 'contract': 'contract', 'start': 'start', 'end': 'end'})
        self.df['symbol'] = self.df['symbol'].str.split('.').str[0]
        self.df.sort_values(by='start', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def  run(self):
        return self.df



if __name__ == '__main__':
    # 铜CU， 棉花CF， 碳酸锂LC， 中质含硫原油SC， 中证1000指数IM， 10年期国债期货T，铁矿石I

    symbol_df = pd.read_csv(r"C:\Users\maki\Desktop\quantchina\Futures-main\data_wash\品种列表.csv")

    # 提取品种代码并去除.后的字符
    symbol_list = symbol_df['S_INFO_WINDCODE'].apply(lambda x: x.split('.')[0]).tolist()

    start_date = '20091201'
    end_date = '20240624'
    output_folder = r'C:\Users\maki\Desktop\quantchina\Futures-main\data_wash\maincontract'
    os.makedirs(output_folder, exist_ok=True)

    #
    for symbol in symbol_list:

        mc = get_main_contract(symbol, start_date, end_date)
        temp = mc.run()

        file_name =  f"{symbol}_{start_date}_{end_date}.csv"
        file_path = os.path.join(output_folder, file_name)
        temp.to_csv(file_path, index=False)
        print(f"Saved file: {file_path}")
    # print(result)




    # result['start'] = result['start'].astype(int)
    # result['end'] = result['end'].astype(int)

    # mark1 = result['start'] <= start_date
    # mark2 = result['end'] >= end_date
    # result.loc[mark1, 'start'] = start_date
    # result.loc[mark2, 'end'] = end_date
    #
    # result.sort_values(by=['symbol', 'start'], inplace=True)
    # result.reset_index(drop=True, inplace=True)
    #
    # result.loc[result['symbol'] == 'CF', 'contract'] = (result.loc[result['symbol'] == 'CF', 'contract'].
    #                                                     str.replace(r'(\D+)(\d{3})(\..+)', r'\g<1>2\g<2>\g<3>', regex=True))
    # result['start'] = pd.to_datetime(result['start'], format= '%Y%m%d').dt.strftime('%Y-%m-%d')
    # result['end'] = pd.to_datetime(result['end'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
