# Steiner_Future_revised
rebuild and revise relative code 

## get 3essential_contracts.py
实际上是获取四个合约数据。获取每日持仓量最大的前三名合约和最近期的合约编号。
需要读取 品种列表.csv 文件，或者自己指定品种

## get_main_contract_wind.py
根据品种列表.csv从wind获取对应品种的主力合约数据，供get_snapshoot3.py twap_vwap.py使用。

## quote_wash.py  quote_wash_revised.py
用于去除不同期货品种开盘时间外的数据，过采样规整数据，频率为0.5秒。
期货的开盘时间从future_information.csv获得。需要读取.csv文件。
quote_wash_revised.py调整了部分语句，优化了future_information的读取逻辑。但是没有处理足够多期货品种数据来证明可靠性和兼容性。

## get_snapshoot3.py 
获取快照数据，内置查询数据、调整合约代码使得和数据库查询格式匹配，区分日夜盘，返回快照数据等功能。
需要读取wind主力合约数据的csv表格，和calendar.csv交易所开市日期表格。
调用quote_wash.py清洗数据

## quote_to_database.py
quote_wash.py的修改版，用来把清洗后的数据按天落到组内数据库。
需要读取get 3essential_contracts.py生成的csv文件，四种合约类型分别存放在各自对应的文件夹。

## twap_vwap.py
获取合约的twap_vwap价格，需要读取wind主力合约数据的csv表格或类似格式的表格。可自己指定时间和区间长度。
需要读取wind主力合约数据的csv表格，和calendar.csv交易所开市日期表格。

## snapshoot&wap.py
整合了get_snapshoot3和twap_vwap的共性部分，可以同时输出快照数据和wap价格数据，提高效率。
需要读取wind主力合约数据的csv表格，和calendar.csv交易所开市日期表格。
