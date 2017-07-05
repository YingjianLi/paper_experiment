#程序没有做置乱，直接选取前1000张图片中作为测试集。结果0.3左右。


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
from pylab import *
import tensorflow as tf



batch_size = 30

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
    x_image = tf.reshape(images, [-1,28, 28,1])

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

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
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
    # Import data
    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #print (mnist)

    # Create the model
    x = tf.placeholder(tf.float32, [None,784])#图片占位符

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None,7])#标签y_#####标签变7

    # Build the graph for the deep net
    y_conv, keep_prob = deep_cnn(x)#运算得到的结果y_conv

    cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))#OK

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#OK

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))#OK

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#OK

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())#OK
        count = 1
        for i in range(150):#OK
            ##batch = mnist.train.next_batch(20)####???
            batch = get_batch(count,batch_size,"F:/face_data/ck+123_faces/")
            count += batch_size
            #print(batch)

            #print('第一位1： ',len(batch[0][0]))
            #print('第二位： ',batch[1])

            if i % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))

            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.55})

        test_set = get_batch(1,1000,"F:/face_data/ck+123_faces/")
        print('test accuracy %g' % accuracy.eval(feed_dict={
                x: test_set[0], y_: test_set[1], keep_prob: 1.0}))

#读出所有的label
def get_label():
    fin = open("F:/face_data/ck+123/ck+labels_maxtrix_with_NE.txt",'r')
    counter = 0
    label_list_all_temp = []
    for line in fin:
        counter += 1
        x = line.split(',')[1:8]
        x[6] = x[6].split(']')[0]
        xx = np.array(x,'f')
        label_list_all_temp.append(xx)
    label_list_all = np.array(label_list_all_temp)
    #print(counter)
    # print(type(label_list_all))
    fin.close()
    return label_list_all
#读出所有的label
label_list_all = get_label()

#获得batch
def get_batch(start_num,batch_size,file_path):
    img_list = []
    label_list = []
    for x in range(start_num,start_num+batch_size):#从start_num开始取batch张图片
        img = Image.open(file_path+str(x)+".png")
        im = np.array(img.resize([28, 28]), 'f')  # 图转数组
        im1 = reshape(im, [784])  # 转变形状
        img_list.append(im1)
    label_list = label_list_all[start_num+1:start_num+batch_size+1]#去对应图片的label
    list = np.array(img_list),np.array(label_list)
    return list
    #print(len(list[0]),len(list[1]))
main()
