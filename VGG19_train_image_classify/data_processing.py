# coding:utf-8
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
import cv2


# 灰度化
def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# 读取图片数据
def image_read(path):
    # 获取文件夹下面的所有文件或文件夹路径
    folder_path = os.walk(path)

    path_list = []
    for i,j,k in folder_path:
        path_list.append(i)

    del path_list[0]

    n = len(path_list)
    m = len(os.listdir(path_list[0]))
    wh = 72
    x_train = np.zeros((n*m,wh,wh,3))
    y_train = []

    #遍历文件夹
    for i,item in enumerate(path_list):
        image_path_list = os.listdir(item)
        # print(i,len(image_path_list))
        # 遍历图片文件
        for j,image_path in enumerate(image_path_list):
            x_train[j+100*i,:,:,:] =imresize(cv2.imread(item+"/"+image_path),[wh,wh])
            y_train.append(i)
    return x_train,y_train,path_list

# 打乱数据集（洗牌）
def shuffle(X, Y):
    Y = np.array(Y)
    n = len(Y)
    index = random.sample(range(n), n)
    x_, y_ = X[index, :, :, :], Y[index]
    return x_, y_

# 把标签转换为onehot编码
def onehot(label):
    return np.array(pd.get_dummies(label))

# 最终处理结果
def get_DS():
    # 获取训练集和测试集数据，
    x_, y_, label = image_read("./face_data/")

    # 打乱数据集
    x_1, y_1 = shuffle(x_,y_)

    # 转换onehot编码
    y_2 = onehot(y_1)

    return x_1,y_2,y_1,label


# # 用于测试代码
# if __name__=="__main__":
#     x_,y_,pl = image_read("./face_data/")
#     print(x_.shape,y_.__len__(),pl)

    # 检测数据是否正确
    # plt.plot(y_)
    # plt.show()

    # label = one_hot(y_)
    # print(label[90:105])
