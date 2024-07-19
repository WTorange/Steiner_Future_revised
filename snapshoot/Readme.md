# snapshoots.py
主要程序，用来获取snapshoots的数据。在主程序下可以通过修改或添加时间范围获得合适区间的快照数据。   

## get_main_contract_wind.py
用来获取主力合约，已经根据期货表格将主力合约的csv文件放在maincontract文件夹，如果需要不同日期的只需要在主程序下修改日期即可    

## future_information.py
用来获取期货的信息，包括合约代码，合约名称，上市日期，交易时间等信息。已经获取完成。如果发现有错误，可以在主程序下修改日期，重新获取并修改。   

## generate_time.py
一个可以根据future_information表格生成交易时间区间的函数。   

## calendar.csv
交易日历，用来判断是否为交易日，已经获取完成。   

## 品种列表.csv
需要的期货品种表格    

## crs_data_api
选择需要的python版本对应的文件放在文件夹中，或者在snapshoot2以及ApiClient程序中修改引用的位置。   

### check_data.py
用来测试数据，复制代码进去，方便操纵。可以疏略。    

# 注意！
1. 目前RU数据可以顺利运行，但是在测试其他品种（比如AG）时有的会遇到报错导致循环无法顺利运行，需要在ApiClient中下载数据通过具体数据进行调试。目前AG的报错还没有完全排查出来。
2. 经验上来说，如果出现问题有较大可能是来源于时间日期格式。
3. 郑商所的合约代码需要补全，目前还没有进行补全，可以先跳过郑商所。我这几天抽空完善一下。
# zlt_quotewash
# zlt_quotewash
# zlt_quotewash
