# coding:utf-8
import pandas as pd

from 链家二手房价预测.connect_sql import read_mongodb_data

# 获取数据
room_data = read_mongodb_data()
'''
['_id', 'House_door_model', 'architecture_acreage', 'within_acreage',
       'Building_head', 'decorate_situation', 'heating_method', 'period_int',
       'floor', 'Family_structure', 'building_types', 'building_age',
       'building_structure', 'Ladder_household_proportion',
       'Equipped_elevator', 'house_id', 'listing_time', 'house_age',
       'Trading_ownership', 'usage_house', 'houseing_belog', 'house_site',
       'bargain_time', 'bargain_price', 'unit_price', 'listing_price',
       'bargain_day_num', 'adjust_price', 'take_look', 'attention',
       'view_count', ]
'''
columns = room_data.columns
print(columns)
data = room_data.iloc[:,1:-3].copy()
# 开始对数据进行处理。

print(data.columns)
data.to_csv('./data/data.csv')





