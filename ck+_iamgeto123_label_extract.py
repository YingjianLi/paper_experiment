#!/usr/bin/env python
#coding=utf-8

#要改成根据标签目查找图片
import os
from PIL import Image
import cv2
import glob

def process(image_dir,label_dir):
    output_path = 'F:/face_data/ck+123/'
    fout = open('F:/face_data/ck+123/ck+labels_num.txt','w+')
    try:
        os.mkdir(output_path)
    except:
        pass
    #count = 0
    num_of_image = 0#所有图片的数量
    for index in range(1000):
        #num_of_seqence = 0#序列的数量
        if index<10:
            dir = label_dir+'/S00'+str(index)
            idir = image_dir+'/S00'+str(index)
        elif index>=10 and index<100: #######fuck! don't mistake 'and' with &
            dir = label_dir + '/S0' + str(index)
            idir = image_dir + '/S0' + str(index)
        else:
            dir = label_dir + '/S' + str(index)
            idir = image_dir + '/S' + str(index)
        if os.path.exists(dir):
            for i in range(15):
                if i<10:

                    label_file_path =  dir+'/00'+str(i)#临时变量存目录，判断是否存在是否为空
                    image_file_path = idir+'/00'+str(i)

                    if os.path.exists(label_file_path) and os.listdir(label_file_path):#存在不空则进入图像的位置找图像
                        print(image_file_path)
                        for label_txt in glob.glob(label_file_path+'/*.txt'):
                            fin = open(label_txt)
                            label_num = fin.readline();
                            #fout.write(image_file_path + " ")
                            for image_path in glob.glob(image_file_path+'/*.png'):

                                num_of_image += 1
                                fout.write(str(num_of_image)+":"+label_num[3]+"\n")
                                #image = Image.open(image_path)
                                #image.save(output_path+str(num_of_image)+".png","png", quality=100)
                            fin.close()

                else:

                    label_file_path =  dir+'/0'+str(i)#临时变量存目录，判断是否存在是否为空
                    image_file_path = idir+'/0'+str(i)
                    if os.path.exists(label_file_path) and os.listdir(label_file_path):#存在不空则进入图像的位置找图像
                        print(image_file_path)
                        for label_txt in glob.glob(label_file_path+'/*.txt'):

                            fin = open(label_txt)
                            label_num = fin.readline();
                            #fout.write(image_file_path+" ")
                            for image_path in glob.glob(image_file_path+'/*.png'):
                                num_of_image += 1
                                fout.write(str(num_of_image)+":"+label_num[3]+'\n')
                                #image = Image.open(image_path)
                                #image.save(output_path+str(num_of_image)+".png","png", quality=100)
                            fin.close()

    fout.close()


    #print("the num of sequence is:  ", count)

if __name__ == '__main__':
    process("F:/face_data/ck+","F:/face_data/ck+label")

'''
#可输出图像数量情况
import os
from PIL import Image
import cv2
import glob

def process(image_dir,label_dir):
    output_path = 'F:/face_data/'+'ck+123_faces'
    try:
        os.mkdir(output_path)
    except:
        pass
    count = 0
    num_of_image = 0#所有图片的数量
    for index in range(1000):
        num_of_seqence = 0#序列的数量
        if index<10:
            dir = image_dir+'/S00'+str(index)
        elif index>=10 and index<100: #######fuck! don't mistake 'and' with &
            dir = image_dir + '/S0' + str(index)
        else:
            dir = image_dir + '/S' + str(index)
        if os.path.exists(dir):
            for i in range(15):
                if i<10:
                    if os.path.exists(dir+'/00'+str(i)):
                        file_path = dir +'/00'+str(i)
                        num_of_seqence +=1
                        for image_path in glob.glob(file_path+'/*.png'):
                            num_of_image += 1
                            image = Image.open(image_path)

                            image.save("F:/face_data/ck+123/"+str(num_of_image)+".png","png", quality=100)


                else:
                    if os.path.exists(dir+'/0'+str(i)):
                        file_path = dir +'/0'+str(i)
                        num_of_seqence +=1
                        for image_path in glob.glob(file_path+'/*.png'):
                            image = Image.open(image_path)
                            num_of_image += 1
                            image.save("F:/face_data/ck+123/"+str(num_of_image)+".png","png", quality=100)


            print(dir + " " + str(num_of_image))
            #count += num_of_seqence

    #print("the num of sequence is:  ", count)
if __name__ == '__main__':
    process("F:/face_data/ck+","F:/face_data/ck+label")
'''




