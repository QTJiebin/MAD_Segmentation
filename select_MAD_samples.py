# -*-coding: utf-8 -*-
# @Time    : 2020/3/29 上午11:34
# @Author  : Jiebin Yan from JXUF
# @Email   : qt1222yan@163.com

import scipy.io as io
import numpy as np
import operator
import os
import cv2
from collections import Counter
from color_map import VOCColormap
from PIL import Image
import torch
import torch.nn.functional as F
import shutil
from collections import Counter
from PIL import Image, ImageDraw, ImageFont


COLOR_MAP = VOCColormap().get_color_map()
IMAGES_EXTENSTIONS = ['jpg', 'png', 'jpeg']

NameList = ['Background','Aeroplane','Bicycle','Bird','Boat','Bottle',
            'Bus','Car','Cat','Chair','Cow','Diningtable','Dog','Horse',
            'Motorbike','Person','Pottedplant','Sheep','Sofa','Train','TvMonitor']

rootpath = '/home/windisk1/jiebin/gMADSemanticSegmentation/'
rootpath_1 = '/home/windisk1/jiebin/PredictionData/'
savepath = '/home/windisk3/gMADSegmentation/MIOU/Classification/'

pair_path = '/PairMat/Discrepancy/PairMat/'
# result = 'MPA/'  # MPA, FWMIoU

def handle_bg(miou_score, pair_n):
    class_p = pair_n.split('_')
    class_npy1 = [[]for i in range(20)]
    class_npy2 = [[]for i in range(20)]
    index = 0

    for i in miou_score:

        s1_map = np.load(rootpath_1+class_p[0]+'/'+i[0].split('_')[0] + '/' + i[0].replace(' ',''))
        s2_map = np.load(rootpath_1+class_p[1]+'/'+i[0].split('_')[0] + '/' + i[0].replace(' ',''))

        for j in range(1, 21):

            if (j in s1_map):
                class_npy1[j-1].append(i)
            else:
                pass
            if (j in s2_map):
                class_npy2[j-1].append(i)
            else:
                pass

    return class_npy1, class_npy2

def makedir(pair_n):
    if os.path.exists(savepath+pair_n):
        pass
    else:
        os.mkdir(savepath+pair_n)
    if os.path.exists(savepath+pair_n+'/'+pair_n.split('_')[0]):
        pass
    else:
        os.mkdir(savepath + pair_n + '/' + pair_n.split('_')[0])
    if os.path.exists(savepath+pair_n+'/'+pair_n.split('_')[1]):
        pass
    else:
        os.mkdir(savepath + pair_n + '/' + pair_n.split('_')[1])


def select_image(class_npy, pair_n, num):

    class_p = pair_n.split('_')
    # threshold choice
    threPer25 = [0.034, 0.013, 0.021, 0.031, 0.006,
                 0.167, 0.018, 0.129, 0.021, 0.069,
                 0.079, 0.076, 0.082, 0.082, 0.027,
                 0.012, 0.039, 0.084, 0.132, 0.028]
    threPer75 = [0.164, 0.087, 0.148, 0.158, 0.148,
                 0.45,  0.262, 0.381, 0.129, 0.292,
                 0.286, 0.302, 0.284, 0.264, 0.249,
                 0.109, 0.297, 0.262, 0.356, 0.199]
    threMin = threPer25
    threMax = threPer75
    makedir(pair_n)

    for i in range(len(class_npy)):
        class_npy[i] = np.array(class_npy[i])
        class_npy[i] = class_npy[i][np.argsort(class_npy[i][:, 1])] # 1-MIOU, 2-MPA，3-FWMIOU

    for n in range(0, 20):
        os.mkdir(savepath + pair_n + '/' + class_p[num] + '/' + str(n))

    select_name = []
    for i in range(len(class_npy)):

        show_number = 0

        for j in range(len(class_npy[i])):

            # rootpath_1
            s1_map = np.load(rootpath_1 + class_p[0] + '/' + class_npy[i][j][0].split('_')[0] + '/' + class_npy[i][j][0].replace(' ',''))
            s2_map = np.load(rootpath_1 + class_p[1] + '/' + class_npy[i][j][0].split('_')[0] + '/' + class_npy[i][j][0].replace(' ',''))
            if num==0:
               s_map = s1_map # defender
            else:
               s_map = s2_map # defender

            if bool(np.sum(s_map == (i+1)) <= (s_map.size * threMax[i])) and bool(np.sum(s_map == (i+1)) >= (s_map.size * threMin[i])):

                s1_map = np.squeeze(s1_map)
                s2_map = np.squeeze(s2_map)

                s1_map = Image.fromarray(s1_map.astype(np.uint8))
                s1_map.putpalette(COLOR_MAP)
                s1_map = s1_map.convert('RGB')

                s2_map = Image.fromarray(s2_map.astype(np.uint8))
                s2_map.putpalette(COLOR_MAP)
                s2_map = s2_map.convert('RGB')

                try:

                    name = class_npy[i][j][0].replace(' ','')[:-4]
                    print(name)
                    # original path: rootpath+'../PASCAL_Craw/'
                    shutil.copy('/home/windisk2/PASCAL_Craw/'+class_npy[i][j][0].split('_')[0]+'/'+class_npy[i][j][0].replace(' ','')[:-4]+'.jpg',
                                savepath+pair_n+'/'+class_p[num]+'/'+str(i)+'/')
                    s1_map.save(savepath+pair_n+'/'+class_p[num]+'/'+str(i)+'/'+str(show_number)+'_'+class_p[0]+'_'+class_npy[i][j][0].split('.')[0]+ '_' + str(i) + '.jpg')
                    s2_map.save(savepath+pair_n+'/'+class_p[num]+'/'+str(i)+'/'+str(show_number)+'_'+ class_p[1]+'_'+class_npy[i][j][0].split('.')[0]+ '_' + str(i) +'.jpg')
                    show_number = show_number+1

                except:
                    print('pass')
                if show_number>=20:
                    break


def draw_label(s1_map):

    TopN = 4
    ct = Counter(s1_map.reshape(-1))
    ct_TopN = ct.most_common(TopN)  # ct_TopN: like [(0, 169721), (15, 92423)]

    s1_map = Image.fromarray(s1_map.astype(np.uint8))
    s1_map.putpalette(COLOR_MAP)
    s1_map = s1_map.convert('RGB')
    # s1_map.show()

    # blank box region
    width = s1_map.size[0]
    height = s1_map.size[1]
    for ww in range(width - 150, width - 5):
        for hh in range(height - 100, height - 5):
            s1_map.putpixel((ww, hh), (255, 255, 255))
    # Draw and text
    draw = ImageDraw.Draw(s1_map)
    cnt = 0
    for top in ct_TopN:
        fillColor = tuple(COLOR_MAP[top[0]])
        l = width - 140;
        t = height - 90 + cnt * 20;
        w = 20;
        h = 15
        draw.polygon([(l, t), (l + w - 1, t), (l + w - 1, t + h - 1), (l, t + h - 1)], outline=fillColor)
        for ww in range(l, l + w - 1):
            for hh in range(t, t + h - 1):
                s1_map.putpixel((ww, hh), fillColor)
        pos = (l + w + 10, t)
        text = NameList[top[0]]
        setFont = ImageFont.truetype('times.ttf', 14)  # request 'times.ttf' file
        draw.text(pos, text, font=setFont, fill=fillColor)
        cnt += 1
        if cnt >= TopN:
            break
    return s1_map


def main():

    pair_name = [
                 'BlitzNet_FCN',
                 'BlitzNet_DiCENet',
                 'BlitzNet_EMANet',
                 'BlitzNet_ESPNetv2',
                 'BlitzNet_DeepLab',
                 'BlitzNet_RefineNet',
                 'BlitzNet_LowLightRefineNet']
    # pair_name = ['BlitzNet_BlitzNetPascalVoc712']
    # pair_name = ['BlitzNet_BlitzNetSeg']

    for pair_n in pair_name:
        print(pair_n)
        miou_score = io.loadmat(rootpath+pair_path+pair_n+'.mat')
        class_npy1, class_npy2 = handle_bg(miou_score['score'], pair_n)  # it costs much time

        np.save('./Discrepancy/' + 'NoBG_' + pair_n+'_'+pair_n.split('_')[0]+'.npy', class_npy1)
        np.save('./Discrepancy/' + 'NoBG_' + pair_n+'_'+pair_n.split('_')[1]+'.npy', class_npy2)
        # class_npy1 = np.load('./Discrepancy/' + 'NoBG_' + pair_n+'_'+pair_n.split('_')[0]+'.npy', allow_pickle=True)
        # class_npy2 = np.load('./Discrepancy/' + 'NoBG_' + pair_n+'_'+pair_n.split('_')[1]+'.npy', allow_pickle=True)

        select_image(class_npy1, pair_n, 0)
        select_image(class_npy2, pair_n, 1)


if __name__ == '__main__':
    main()
