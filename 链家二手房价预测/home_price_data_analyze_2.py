# coding:utf-8
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

data = pd.read_csv('./data/room_data.csv',sep=',',header=0,index_col=0)
# data = data1.copy()
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



# 插入某列 数据
# data.insert(0,'shi',[i for i in range(len(data.iloc[:,2]))])
# print(data.head())

# 1、对 House_door_model ：房户结构进行分解  2室1厅1厨1卫

def house_model(info):
    n = len(info)
    house_door_model = np.zeros((n,4))
    data = info.copy()
    for i,item in enumerate(data):
        if re.findall('\d',item).__len__():
            house_door_model[i,0],house_door_model[i,1],house_door_model[i,2], \
            house_door_model[i, 3]= [int(i) for i in re.findall('\d',item)]
        else:
            # 没有数据的全部给它一个室一卫
            house_door_model[i, 0], house_door_model[i, 1], house_door_model[i, 2], \
            house_door_model[i, 3] = [1,0,0,0]


    return house_door_model

# 把数据插入data。
info = data['House_door_model'].copy()
house_door_model = house_model(info)
house_list = ['house_室','house_厅','house_厨','house_卫']
for i,item in enumerate(house_list):
    data.insert(i,item,house_door_model[:,i])

# 删除掉house_door_model列
data.drop('House_door_model', axis=1, inplace=True)
# plt.scatter(range(len(house_door_model[:,2])),house_door_model[:,3])
# plt.show()



# 2、对 architecture_acreage 房子面积进行提取

architecture_acreage = data['architecture_acreage'].copy()

data.insert(4,'architecture_acreage_',[float(i) for i in [item.split('㎡')[0] for item in architecture_acreage]])

data.drop('architecture_acreage',axis=1,inplace=True)



# 3、对 within_acreage ：内部面积

within_acreage = data['within_acreage'].copy()
index_list = [i for i,item in enumerate(list(data.columns)) if item=='within_acreage'][0]

def rinse_within_acreage(info):
    df = info.copy()
    within_acreage_list = np.zeros((len(df),1))
    for i,item in enumerate(df):
        if str(item) == '暂无数据':
            within_acreage_list[i] = data['architecture_acreage_'][i]
        else:
            within_acreage_list[i] = float(item.split('㎡')[0])
    return within_acreage_list

data.insert(index_list,'within_acreage_',rinse_within_acreage(within_acreage)[:,0])

data.drop('within_acreage',axis=1,inplace=True)

#  清除空字段
data.drop('heating_method',axis=1,inplace=True)



# 字段 floor 楼层的高低 清洗

index_floor = [i for i,item in enumerate(list(data.columns)) if item=='floor'][0]

room_floor = data['floor'].copy()

def rinalyze_floor(df):
    df = df.copy()
    floor_list = np.zeros((len(df)))
    f_l = []
    for i,item in enumerate(df):
        floor_info = re.findall('(.*?)\(共(\d)层',item)
        if len(floor_info):
            # print(floor_info)
            f_l.append(floor_info[0][0])
            floor_list[i] = int(floor_info[0][1])
        else:
            f_l.append('底楼层')
            floor_list[i] = 1



    return [f_l,floor_list]

floor_ = rinalyze_floor(room_floor)

fl = ['floor_楼','floor_层']
for i,item in enumerate(fl):
    data.insert(i+index_floor,item,floor_[i])

data.drop('floor',axis=1,inplace=True)



# 处理 building_age 建筑时间

building_age = data['building_age'].copy()
build_index = [i for i,item in enumerate(list(data.columns)) if item=='building_age'][0]

def rinalyze_building_age(df):
    df = df.copy()
    build = np.zeros((len(df)),dtype='int')
    for i,item in enumerate(df):
        if str(item) == '未知':
            build[i] = 2000
        else:
            build[i] = int(item)
    return build

data.insert(build_index,'building_age_',rinalyze_building_age(building_age))

data.drop('building_age',axis=1,inplace=True)



# 删除链接编号
data.drop('house_id',axis=1,inplace=True)
data.drop('house_site',axis=1,inplace=True)
data.drop('listing_time',axis=1,inplace=True)
data.drop('bargain_time',axis=1,inplace=True)



# 清洗view_count

view_count = data['view_count'].copy()

def rinalyze_view_count(df,str_s):
    df = df.copy()
    view_data = []
    for item in df:
        if str(item)==str_s:
            pass
            # print('暂无数据')
        else:
            view_data.append(int(item))
    mean = np.floor(np.mean(np.array(view_data)))
    count = []
    for di in df:
        if str(di) == str_s:
            count.append(mean)
        else:
            count.append(int(di))

    return count

view_count_index = [i for i,item in enumerate(list(data.columns)) if item=='view_count'][0]

data.insert(view_count_index,'view_count_',rinalyze_view_count(view_count,str_s='暂无数据'))

data.drop('view_count',axis=1,inplace=True)

# print(data.iloc[0:5,-1])
# bargain_day_num

view_count_index = [i for i,item in enumerate(list(data.columns)) if item=='bargain_day_num'][0]

data.insert(view_count_index,'bargain_day_num_',rinalyze_view_count(view_count,str_s='暂无数据'))

data.drop('bargain_day_num',axis=1,inplace=True)


print(data.shape)


data1 = data.copy()
data1.to_csv('./data/room_data_2.csv',encoding='gbk')
