# -*- coding: utf-8 -*-

from PIL import Image
import glob, os


# 图片批处理
def label_extract():
    count = 1
    emo_id = {"HA":1,"SU":2,"FE":3,"AN":4,"DI":5,"SA":6,"NE":7}
    f = open("F:/face_data/jaffe123/jaffe_label.txt", "w+")
    for files in glob.glob('F:/face_data/jaffe/*.tiff'):
        filepath, filename = os.path.split(files)
        filterame, exts = os.path.splitext(filename)
        emo = filename.split('.')[1][:2]#记录情绪如'HA'两个字母。
        list = [0,0,0,0,0,0,0,0]
        list[emo_id[emo]] = 1#注意从0开始,但是0位存图片序号
        list[0] = count
        f.write(str(list)+"\r\n")
        #print (emo)
        # 输出路径
        opfile = r'F:/face_data/jaffe123/'
        #label_file = r'F:/face_data/jaffe123/jaffe_label.txt'

        # 判断opfile是否存在，不存在则创建

        if (os.path.isdir(opfile) == False):
            os.mkdir(opfile)
        im = Image.open(files)
        #w, h = im.size
        # im_ss = im.resize((400,400))
        # im_ss = im.convert('P')
        #im_ss = im.resize((int(w * 0.12), int(h * 0.12)))
        #im_ss.save(opfile + str(count) + '.tiff')
        im.save(opfile + str(count) + '.tiff')
        count += 1
    f.close()

if __name__ == '__main__':
    label_extract()

    print("处理完成！")