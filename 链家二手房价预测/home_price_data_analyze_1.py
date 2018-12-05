# coding:utf-8
import pandas as pd
import numpy as np

data = pd.read_csv('./data/data.csv',sep=',',header=0,index_col=0)
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

# 清除空格数据

idx,columns = data.shape
room_info = pd.DataFrame(data=np.zeros((idx,columns)),columns=list(data.columns))
print(room_info.shape,data.shape)
for i in range(idx):
    for j in range(columns):

        room_info.iloc[i,j] = str(data.iloc[i,j]).strip()

room_info.to_csv('./data/room_data.csv')










