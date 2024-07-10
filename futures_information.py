import pandas as pd

from connect_wind import ConnectDatabase

# 获取历史主力对应月合约
class get_information(ConnectDatabase):
    """s
    获取指定期货品种在给定日期范围内的主力合约信息
    """

    def __init__(self):
        """
        初始化方法，定义查询所需的参数和SQL语句
        :param future_symbol: 期货品种代码（例：铜CU）
        :param start_date: 开始日期（格式：YYYYMMDD）
        :param end_date: 结束日期（格式：YYYYMMDD）
        """
        # self.future_symbol = future_symbol  #例 铜CU(大写)
        self.sql = f'''
                    SELECT S_INFO_WINDCODE,S_INFO_CODE, S_INFO_CDMONTHS, S_INFO_THOURS,S_INFO_EXNAME
                    FROM CFUTURESCONTPRO
                    '''

        super().__init__(self.sql)
        self.df = super().get_data()
        # self.df = self.df[['FS_MAPPING_WINDCODE', 'STARTDATE', 'ENDDATE']]
        self.df = self.df.rename(columns={'S_INFO_WINDCODE': 'symbol',
                                          'S_INFO_CODE': 'code',
                                          'S_INFO_CDMONTHS': 'month',
                                          'S_INFO_THOURS': 'hours',
                                          'S_INFO_EXNAME': 'exname'})
        # self.df['symbol'] = self.df['symbol'].str.split('.').str[0]


    def  run(self):
        """
        返回处理后的数据框
        :return: DataFrame 包含主力合约信息
        """
        return self.df



if __name__ == '__main__':
    """
    主程序，获取多个期货品种在指定日期范围内的主力合约信息，并进行处理
    """

    result = pd.DataFrame(columns=['symbol', 'contract', 'start', 'end'])

    mc = get_information()
    temp = mc.run()

    temp.to_csv('future_information.csv')

