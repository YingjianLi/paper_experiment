from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
from pylab import *
import random
from sklearn.metrics import confusion_matrix

import tensorflow as tf

batch_size = 30
iter_num = 1000


def deep_cnn(images):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
     Returns:
       A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
     dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    x_image = tf.reshape(images, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, 7])
    b_fc2 = bias_variable([7])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main():
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])  # 图片占位符

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 7])  # 标签y_#####标签变7

    # Build the graph for the deep net
    y_conv, keep_prob = deep_cnn(x)  # 运算得到的结果y_conv

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))  # OK

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # OK

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # OK

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # OK

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # OK
        count = 1
        for i in range(iter_num):  # OK
            ##batch = mnist.train.next_batch(20)####???
            # 留下1000张测试
            if count > (4000 - batch_size):  # 保证最后一个batch不会取到数据集外,保证只去前4000中的图片
                count = 1
            # if count>(5640-batch_size):#保证最后一个batch不会取到数据集外
            #    count = 1
            batch = get_batch(count, batch_size, "F:/face_data/ck+123_faces/")
            count += batch_size
            # print(batch)

            if i % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.75})

        test_set = get_batch(4000, 1640, "F:/face_data/ck+123_faces/")  # 从4001开始取，取到5640（包括）,

        ''''
         #考虑混淆矩阵
        # y = sess.run(y_conv,feed_dict={x: test_set[0], y_: test_set[1], keep_prob: 1.0})

        # print(y_conv)
        real_y = []
        pred_y = []
        for n in range(4000,5640):
            label_vector = label_list_all[random_arr[n]]
            for xx in range(7):
                if label_vector[xx]==1:
                    real_y.append(xx)
        for n in range(1640):
            label_vector = y[n]
            for xx in range(7):
                if label_vector[xx]==1:
                    pred_y.append(xx)

        print('/n')
        #print(confusion_matrix(y_true=real_y,y_pred=pred_y))
        print(real_y)
        '''

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: test_set[0], y_: test_set[1], keep_prob: 1.0}))


# 读出所有的label
def get_label():
    fin = open("F:/face_data/ck+123/ck+labels_maxtrix_with_NE1.txt", 'r')
    counter = 0
    label_list_all_temp = []
    for line in fin:
        counter += 1
        x = line.split(',')[1:8]
        x[6] = x[6].split(']')[0]
        xx = np.array(x, 'f')
        label_list_all_temp.append(xx)
    label_list_all = np.array(label_list_all_temp)
    # print(counter)
    # print(type(label_list_all))
    fin.close()
    return label_list_all


# 读出所有的label
label_list_all = get_label()
random_arr = []
for i in range(0, 5640):
    random_arr.append(i)  # 注意从0开始
random.shuffle(random_arr)


# print(min(random_arr))

def get_batch(start_num, batch_size, file_path):
    img_list = []
    label_list = []
    for x in range(start_num, start_num + batch_size):  # 从srat开始去batch张图片，根据random中的随机数随机取
        img = Image.open(file_path + str(random_arr[x] + 1) + ".png")  # random_arr[x]对应的图片编号是random_arr[x]+1
        im = np.array(img.resize([28, 28]), 'f')  # 图转数组
        #去均值操作
        mean = im.sum()/(28*28)
        im = im-mean
        im1 = reshape(im, [784])  # 转变形状
        img_list.append(im1)
        label_list.append(label_list_all[random_arr[x]])  # random_arr和label_list_all都是从0开始

    # label_list = label_list_all[start_num+1:start_num+batch_size+1]#去对应图片的label
    list = np.array(img_list), np.array(label_list)
    return list
    # print(len(list[0]),len(list[1]))
    # print (list[0])
    # print (list[1])


'''
#挨个取batch，随机性不好
#读出所有的label
label_list_all = get_label()

#获得batch
def get_batch(start_num,batch_size,file_path):
    img_list = []
    for x in range(start_num,start_num+batch_size):#从start_num开始取batch张图片
        img = Image.open(file_path+str(x)+".png")
        im = np.array(img.resize([28, 28]), 'f')  # 图转数组
        im1 = reshape(im, [784])  # 转变形状
        img_list.append(im1)
    label_list = label_list_all[start_num+1:start_num+batch_size+1]#去对应图片的label
    list = np.array(img_list),np.array(label_list)
    return list
    #print(len(list[0]),len(list[1]))
'''
main()
