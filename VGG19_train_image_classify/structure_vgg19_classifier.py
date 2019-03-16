# coding:utf-8
import tensorflow as tf
import os
from load_vgg19_model import net

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def VGG19_image_classifier(X,Y,nn_classes):

    vgg19_path = "./vgg19_model/imagenet-vgg-verydeep-19.mat"
    net_list,mean_pixel,all_layers = net(vgg19_path,X)

    vgg19_pool5 = net_list[-1]["pool5"]

    vgg19_pool5_shape = vgg19_pool5.get_shape().as_list()

    vgg19_pool5_number = vgg19_pool5_shape[1]*vgg19_pool5_shape[2]*vgg19_pool5_shape[3]

    weights = {
        'wd1': tf.Variable(tf.random_normal([vgg19_pool5_number, 4096])),
        'wd2': tf.Variable(tf.random_normal([4096, 4096])),
        'out': tf.Variable(tf.random_normal([4096, nn_classes]))
    }

    biases = {
        'bd1': tf.Variable(tf.zeros([4096])),
        'bd2': tf.Variable(tf.zeros([4096])),
        'out': tf.Variable(tf.zeros([nn_classes]))
    }

    # 全连接一层
    _densel = tf.reshape(vgg19_pool5, [-1, vgg19_pool5_number])

    fc6 = tf.add(tf.matmul(_densel,weights["wd1"]),biases["bd1"])
    relu6 = tf.nn.relu(fc6)

    # 全连接二层

    fc7 = tf.add(tf.matmul(relu6,weights["wd2"]),biases["bd2"])
    relu7 = tf.nn.relu(fc7)

    # 输出层
    fc8 = tf.add(tf.matmul(relu7,weights["out"]),biases["out"])

    # out = tf.nn.softmax(fc8)
    out = fc8

    # 损失函数 loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=out))  # 计算交叉熵

    # 优化目标 optimizing
    optimizing = tf.train.AdamOptimizer(0.0001).minimize(loss)  # 使用adam优化器来以0.0001的学习率来进行微调

    # 精确度 accuracy
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(out, 1))  # 判断预测标签和实际标签是否匹配
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 想要保存的模型参数，方便加载找到。
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("out", out)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("optimizing", optimizing)

    return {
        "loss": loss,
        "optimizing": optimizing,
        "accuracy": accuracy,
        "out": out,
        "mean_pixel":mean_pixel
    }

