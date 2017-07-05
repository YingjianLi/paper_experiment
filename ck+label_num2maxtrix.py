# -*- coding:utf-8 -*-


fin = open("F:/face_data/ck+123/ck+labels_num.txt")
fout = open("F:/face_data/ck+123/ck+labels_maxtrix_with_NE.txt","w+")

for line in fin:
    emo_mat = [0, 0, 0, 0, 0, 0, 0,0]  # 第一维是编号,最后是中性，都是零
    print(line)
    num, emo_num = line.split(":")
    emo_mat[0] = int(num)
    if int(emo_num) == 1:
        emo_mat[4] = 1
    elif int(emo_num) == 3:
        emo_mat[5] = 1
    elif int(emo_num) == 4:
        emo_mat[3] = 1
    elif int(emo_num) == 5:
        emo_mat[1] = 1
    elif int(emo_num) == 6:
        emo_mat[6] = 1
    else:
        emo_mat[2] = 1
    fout.write(str(emo_mat)+'\n')
fout.close()
fin.close()





