#!/usr/bin/env python
#coding=utf-8
#"F:/face_data/ck+123" 中的图像截出人脸保存到"F:/face_data/ck+123_faces"中

import os
from PIL import Image
import cv2
#import grob
#import dlib

def detect_object(input_path, output_path,count):
    image = cv2.imread(input_path)
    '''
    检测图片，获取人脸在图片中的坐标
    '''
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#将当前桢图像转换成灰度图像
    #color = (0,100,200) #设置人脸框的颜色
    classfier=cv2.CascadeClassifier("C:/Python35/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml")
    #人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
    faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.3, minNeighbors = 4, minSize = (32, 32))

    im = Image.open(input_path).convert('L')

    if len(faceRects) > 0: #大于0则检测到人脸
        #draw = ImageDraw.Draw(im)
        for faceRect in faceRects:

            #单独框出每一张人脸
            x, y, w, h = faceRect
            #画出矩形框
            #cv2.rectangle(grey, (x, y), (x + w, y + h), color, 2)

            #确定面部位置
            face = (x, y,x + w,y + h)
            #截取放在a中
            img = im.crop(face).resize((224,224))

            file_name = os.path.join(output_path,str(count)+".png")
            #保存
            img.save(file_name,"png", quality=100)


    else:
        print ("Error: cannot detect faces on %s" % input_path)
    #return 0

def process(file_path):
    output_path = 'F:/face_data/'+'/'+'ck+123_faces'
    try:
        os.mkdir(output_path)
    except:
        pass
    count = 1
    for index in range(5643):
        input_path = file_path+'/'+str(index+1)+'.png'
        print (input_path)
        faces = detect_object(input_path, output_path,count)
        count += 1


if __name__ == '__main__':
    process("F:/face_data/ck+123")



