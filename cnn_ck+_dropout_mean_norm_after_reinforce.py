#十折交叉验证法
#2017.7.5
#实验结果未保存

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import datetime
from PIL import Image
from pylab import *
import random
from sklearn.metrics import confusion_matrix

import tensorflow as tf

#重要参数,运行结束时写入文件
batch_size = 100
iter_num = 20000
dropout = 1
learn_rate = 10e-3
conv_layers = [3,5,1,'same']#stride = 1, pading=same
pooling_layers=[3,3,2,'same']#stride = 2
full_connected_layers = [2,1024,7]
image_size = 64

def deep_cnn(images):
    """deepnn builds the graph for a deep net for classifying face expression.
    Args:
      x: an input tensor with the dimensions (N_examples, 64*64), where 64*64 is the
      number of pixels in a face image.
     Returns:
       A tuple (y, keep_prob). y is a tensor of shape (N_examples,7), with values
      equal to the seven emotions. keep_prob is a scalar placeholder for the probability of
     dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    x_image = tf.reshape(images, [-1, 64, 64, 1])

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

    # third convolutional layer -- maps 64 feature maps to 128.
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    #Third pooling
    h_pool3 = max_pool_2x2(h_conv3)


    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([8 * 8 * 128, 1024])
    b_fc1 = bias_variable([1024])


    h_conv3_flat = tf.reshape(h_pool3, [-1, 8 * 8 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)


    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


    # 增加这一全连接层时准确率降低到40%
    # W_fc2 = weight_variable([1024, 1024])
    # b_fc2 = bias_variable([1024])

    # h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    # h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)



    W_fc3 = weight_variable([1024, 7])
    b_fc3 = bias_variable([7])

    y_conv = tf.matmul(h_fc1_drop, W_fc3) + b_fc3



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

# 读出所有的label
def get_label():
    fin = open("F:/face_data/ck+123/ck+labels_maxtrix_with_NE.txt", 'r')
    counter = 0
    label_list_all_temp = []
    for line in fin:
        counter += 1
        x = line.split(',')[1:8]
        x[6] = x[6].split(']')[0]
        xx = np.array(x, 'f')
        label_list_all_temp.append(xx)
    label_list_all = np.array(label_list_all_temp)
    fin.close()
    return label_list_all


# 读出所有的label
label_list_all = get_label()
random_arr = []
for i in range(0, 11280):# 使用了11280张
    random_arr.append(i)  # 注意从0开始
random.shuffle(random_arr)
acc_arr = []# 记录每一折的准确率
training_time = [] # 记录训练时间
testing_time = [] # 记录测试时间

# print(min(random_arr))
image_list = []

def get_batch(start_num, batch_size, file_path):
    img_list = []
    label_list = []
    for x in range(start_num, start_num + batch_size):  # 从srat开始去batch张图片，根据random中的随机数随机取
        img = Image.open(file_path + str(image_list[x]+ 1) + ".png")  # random_arr[x]对应的图片编号是random_arr[x]+1
        im = np.array(img.resize([64, 64]), 'f')  # 图转数组

        # 去均值操作
        mean = im.sum()/(64*64)
        im = im-mean
        im1 = reshape(im, [64*64])  # 转变形状

        img_list.append(im1)
        label_list.append(label_list_all[image_list[x]])  # random_arr和label_list_all都是从0开始

    # label_list = label_list_all[start_num+1:start_num+batch_size+1]#去对应图片的label
    list = np.array(img_list), np.array(label_list)
    return list
    # print(len(list[0]),len(list[1]))
    # print (list[0])
    # print (list[1])

def main():
    # Create the model
    x = tf.placeholder(tf.float32, [None, 64 * 64])  # 图片占位符

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 7])  # 标签y_#####标签变7

    # Build the graph for the deep net

    y_conv, keep_prob = deep_cnn(x)  # 运算得到的结果y_conv

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))  # OK

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # OK

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  # OK

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # OK

    with tf.Session() as sess:

        #print(random_arr)
        for fold in range(10):
            sess.run(tf.global_variables_initializer())  # OK
            #temp_count = 0  # zhuyi 从0开始，代表的random_arr 0 的位置
            ii = 0
            del image_list[:]#否则一直追加
            #print(len(image_list))
            while(ii<11280):
                if ii == fold * 1128:
                    #temp_count += 1128
                    ii += 1128
                else:

                    image_list.append(random_arr[ii])  # 这一折的训练集
                    #temp_count += 1
                    ii += 1
            for iii in range(fold * 1128, fold * 1128 + 1128):
                image_list.append(random_arr[iii])#这一折的测试集加在尾部

            count = 1
            train_begin = datetime.datetime.now()
            for i in range(iter_num):  # OK
                # 留下1000张测试
                if count > (10152 - batch_size):  # 保证最后一个batch不会取到数据集外,保证只去前10152中的图片
                    count = 1
                # if count>(5640-batch_size):#保证最后一个batch不会取到数据集外
                #    count = 1
                batch = get_batch(count, batch_size, "F:/face_data/ck+123_faces/")
                count += batch_size
                # print(batch)

                if i % 10 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: dropout})
            train_end = datetime.datetime.now()
            training_time.append(train_end-train_begin)#记录训练时间
            print('--------------------------------')
            print('\033[1;35m training time: \033[0m!', train_end - train_begin)


            test_set = get_batch(10152, 1128, "F:/face_data/ck+123_faces/")  # 从7521开始取，取到11280（包括）,3760

            test_begin = datetime.datetime.now()
            temp_acc = accuracy.eval(feed_dict={x: test_set[0], y_: test_set[1], keep_prob: 1.0})
            acc_arr.append(temp_acc)
            test_end = datetime.datetime.now()
            testing_time.append(test_end-test_begin)

            print('\033[1;35m testing time: \033[0m!',test_end-test_begin)
            print('\033[1;35m training acc: \033[0m!', temp_acc)
            print('--------------------------------')
            #print('test accuracy %g' % accuracy.eval(feed_dict={x: test_set[0], y_: test_set[1], keep_prob: 1.0}))
        saver = tf.train.Saver()
        saver.save(sess, "F:/face_data/10020000")
    print('\033[1;35m training acc: \033[0m!',acc_arr)
    print('\033[1;35m average acc: \033[0m!',sum(acc_arr)/len(acc_arr))
    print('\033[1;35m total training time: \033[0m!', sum(training_time))
    print('\033[1;35m total testing time: \033[0m! ' ,sum(testing_time))
    print('\033[1;35m average forward time per image: \033[0m! ',sum(testing_time)/11280)

    # 写入文件
    result_file = open('F:/face_data/test_result/result.txt', 'a')
    result_file.write('date:' + str(datetime.datetime.now()) + '\n')
    result_file.write('image_size: ' + str(image_size) + '\n')
    result_file.write('batch_size: ' + str(batch_size) + '\n')
    result_file.write('iter_num: ' + str(iter_num) + '\n')
    result_file.write('dropout: ' + str(dropout) + '\n')
    result_file.write('learn_rate: ' + str(learn_rate) + '\n')
    result_file.write('conv_layers: ' + str(conv_layers) + '\n')
    result_file.write('pooling_layers: ' + str(pooling_layers) + '\n')
    result_file.write('full_connected_layers: ' + str(full_connected_layers) + '\n')
    result_file.write('training acc: '+str(acc_arr)+'\n')
    result_file.write('average acc: '+str(sum(acc_arr)/len(acc_arr))+'\n')
    result_file.write('total training time: '+str(sum(training_time))+'\n')
    result_file.write('total testing time: '+str(sum(testing_time))+'\n')
    result_file.write('average forward time per image: '+str(sum(testing_time)/11280)+'\n')
    result_file.write('\n')
    result_file.write('\n')

    result_file.close()
    #备份
    result_file = open('F:/face_data/test_result/result.txt', 'r')
    result_file_backup = open('G:/result_backup.txt', 'w')
    for line in result_file:
        result_file_backup.write(line)
    result_file_backup.close()


main()
