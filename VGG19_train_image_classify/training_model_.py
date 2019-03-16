# coding:utf-8
import tensorflow as tf
import os
from data_processing import get_DS
import numpy as np
from structure_vgg19_classifier import VGG19_image_classifier
from load_vgg19_model import preprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def minus_mean(x_,mean_pixel):
    # 输入层img预处理：减均值

    n,w,h,d = x_.shape

    x = np.zeros((n,w,h,d))

    for i in range(n):
        x[i,:,:,:] = np.array([preprocess(x_[i,:,:,:], mean_pixel)])

    return x


# 获取训练集与测试集以 8:2 分割
x_,y_,y_true,label = get_DS()

label_number = len(label)

x_train,y_train = x_[:960,:,:,:],y_[:960,:]

x_test,y_test = x_[960:,:,:,:],y_[960:,:]

print(x_train.shape,y_test.shape)




n,w,h,d = x_train.shape

# 开始训练模型


# 开始准备训练cnn
X = tf.placeholder(tf.float32,[None,w,h,d],name = 'X')
# 这个12属于人脸类别，一共有几个id
Y = tf.placeholder(tf.float32, [None,label_number],name = 'Y')

# 实例化模型
vgg19_model = VGG19_image_classifier(X,Y,label_number)

loss,optimizing,accuracy,out = vgg19_model["loss"],vgg19_model["optimizing"],vgg19_model["accuracy"],vgg19_model["out"]
mean_pixel = vgg19_model["mean_pixel"]

# 启动训练模型
bsize = 960/60

# 模型保存
saver = tf.train.Saver()

if __name__=="__main__":
    with tf.Session() as sess:
        # 实例所有参数
        sess.run(tf.global_variables_initializer())
        for epoch in range(10):
            for i in range(15):
                x_bsize,y_bsize = x_train[i*60:i*60+60,:,:,:],y_train[i*60:i*60+60,:]

                # 输入层img预处理：减均值
                # x_bsize = minus_mean(x_bsize,mean_pixel)
                # print(x_bsize.shape)

                sess.run(optimizing,feed_dict={X:x_bsize,Y:y_bsize})

            if (epoch+1)%1==0:

                # x_test = minus_mean(x_test,mean_pixel)

                # print(x_test.shape)

                los = sess.run(loss,feed_dict={X:x_test,Y:y_test})
                acc = sess.run(accuracy,feed_dict={X:x_test,Y:y_test})

                print("epoch:%s loss:%s accuracy:%s"%(epoch,los,acc))

        score= sess.run(accuracy,feed_dict={X:x_test,Y:y_test})

        print(score)

        # # 保存cnn模型
        # saver.save(sess,"./save_data_and_model/cnn_model")

        # y_pred = sess.run(out,feed_dict={X:x_test})
        #
        # # 这个是类别，测试集预测出来的类别。
        # y_pred = np.argmax(y_pred,axis=1)
        #
        # print("最后的精确度为：%s"%score)
        #
        # # 最后类别比对
        # print(y_pred)
        # print(y_true[960:])