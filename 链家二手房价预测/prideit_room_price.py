# coding:utf-8
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
'''
model = pd.read_table('./data/模板字符.txt',sep='，',header=None,index_col=None,encoding='gbk',engine='python')
print(model)
str_model = list(model.values[0])
print(str_model)
'''

data = pd.read_csv('./data/room_data_2.csv',sep=',',header=0,index_col=0,encoding='gbk')

data.drop('Ladder_household_proportion',axis=1,inplace=True)

# 归一化数据 ：architecture_acreage_  within_acreage_  building_age_   unit_price
#   listing_price   bargain_day_num_   view_count_
# y_train    bargain_price


#  数据归一化处理，使用岭回归进行预测。

rinalyze_list = ['architecture_acreage_','within_acreage_','building_age_','unit_price',
                 'listing_price','bargain_day_num_','attention','view_count_']

y_train_ = data.pop('bargain_price')

x = [1,2,3,4,5,6,0]

def min_max_scaler(df):
    df = np.array(df)
    min_max = preprocessing.MaxAbsScaler()
    return min_max.fit_transform(df.reshape((-1,1))).reshape((-1))

for item in rinalyze_list:
    data[item] = min_max_scaler(data[item].copy())

y_train_data = min_max_scaler(y_train_.copy())

linsting_price = data['listing_price'].copy()[1500:]

# onehot编码处理
room_data = pd.get_dummies(data)
# room_data.to_csv('./data/train.csv',encoding='gbk')
# 分割训练数据和测试数据。
x_train = room_data.iloc[:1500,:]
y_train = y_train_data[:1500]
x_test = room_data.iloc[1500:,:]
y_test = y_train_data[1500:]


# plt.scatter(range(len(y_train)),y_train)

# 使用xgboost来选择模型超参


# 岭回归

plt.figure(figsize=(12,6))
alphas = np.logspace(-3, 2, 50)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, x_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))


plt.plot(alphas, test_scores)
plt.title("Alpha vs CV Error")


# 随机森林。

max_features = [.1, .3, .5, .7, .9, .99]
test_scores = []
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
    test_score = np.sqrt(-cross_val_score(clf, x_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.figure(figsize=(12,6))
plt.plot(max_features, test_scores)
plt.title("Max Features vs CV Error")




# 在选择超参数的使用随机森林表现更加优秀。大概max = 0.5左右

rf_model = RandomForestRegressor(n_estimators=200, max_features=0.5)
rf_model.fit(x_train,y_train)
y_pred = rf_model.predict(x_test)


# score = np.sum(np.abs(np.array(y_test)-np.array(y_pred)))/len(x_test)
# print(score)
from sklearn.metrics import mean_absolute_error
score = mean_absolute_error(np.array(y_test),np.array(y_pred))
print(score)

plt.figure(figsize=(12,6))
plt.plot(y_test[:100],'o--',label='成交价格')
plt.plot(y_pred[:100],'.-',label='预测价格')
plt.plot(range(len(linsting_price[:100])),linsting_price[:100],'*',label='挂单价格')
plt.ylabel('价格: x $10^3$万元')
plt.title('链家网二手房价预测')
plt.legend(loc='upper right')
plt.text(36,0.6,'平均误差：$%s x 10^3$万元'%score)
plt.show()

