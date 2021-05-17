# -*- coding: utf-8 -*-
# @Time    : 2019/10/24 15:53
# @Author  : Jiebin Yan from JXUF
# @Email   : qt1222yan@163.com

import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import threadpool
import scipy.io as io
import time


class IOUMetric(object):
    """
    Class to calculate mean-iou using fast_hist method
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou)

        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()

        return acc, acc_cls, iou, miou, fwavacc


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)  # 21*21的矩阵,行代表ground truth类别,列代表preds的类别,值代表

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)  # 跳过0值求mean,shape:[21]
        return MIoU

    def Class_IOU(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)  # ground truth中所有正确(值在[0, classe_num])的像素label的mask

        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)  # 21 * 21(for pascal)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        tmp = self._generate_matrix(gt_image, pre_image)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) +
                np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU


class CAL_s():
    def __init__(self,image_list,moudle,thread=10):
        super(CAL_s, self).__init__()
        self.image_list = image_list
        self.md = moudle
        self.threads = thread
        self.score = []
    def run(self):
        pool = threadpool.ThreadPool(self.threads)
        requests = threadpool.makeRequests(self.cal_r, self.image_list)
        [pool.putRequest(i) for i in requests]
        pool.wait()
        io.savemat('./'+self.md[0]+ '_'+self.md[1] + '.mat',{'score':np.array(self.score)})

    def cal_r(self,image):
        try:
            MADCalc1 = Evaluator(21)
            # MADCalc2 = IOUMetric(21)

            # gtssss = np.load('/media/h441/Elements/SaveImages/' + self.md[0] + '/' + image.split('_')[0] + '/' + image)
            gtssss = np.load('/home/windisk1/jiebin/PredictionData/' + self.md[0] + '/' + image.split('_')[0] + '/' + image)
            # print('/home/windisk1/jiebin/PredictionData/' + self.md[0] + '/' + image.split('_')[0] + '/' + image)
            gtssss = gtssss.astype(np.int)
            gtssss = np.squeeze(gtssss)

            # predictionssss = np.load('/media/h441/Elements/SaveImages/' + self.md[1] + '/' + image.split('_')[0] + '/' + image)
            predictionssss = np.load('/home/windisk1/jiebin/PredictionData/' + self.md[1] + '/' + image.split('_')[0] + '/' + image)
            predictionssss = predictionssss.astype(np.int)
            predictionssss = np.squeeze(predictionssss)

            MADCalc1.add_batch(gtssss, predictionssss)  # target:[batch_size, 512, 512]

            s1 = MADCalc1.Mean_Intersection_over_Union()
            s2 = MADCalc1.Pixel_Accuracy_Class()
            s3 = MADCalc1.Frequency_Weighted_Intersection_over_Union()

            self.score.append(np.array([image, s1, s2, s3]))
        except Exception as e:
            pass

if __name__ == '__main__':


    root_path ='/home/windisk1/jiebin/PredictionData/'
    
    class_par = [['BlitzNet', 'FCN'],
                 ['BlitzNet', 'DiCENet'],
                 ['BlitzNet', 'EMANet'],
                 ['BlitzNet', 'ESPNetv2'],
                 ['BlitzNet', 'DeepLab'],
                 ['BlitzNet', 'RefineNet']
                 ['BlitzNet', 'LowLightRefineNet']]

    # class_par = [['BlitzNet', 'BlitzNet_Seg']]

    class_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable','dog', 'horse', 'motor', 'person', 'plant', 'sheep', 'sofa', 'train', 'tvmonitor']

    image_list = []
    for class_p in class_par:
        print(class_p)
        for i in class_list:
            image_list.extend(os.listdir(root_path + class_p[0] + '/' + i))
        # print(class_p+str(len(image_list)))
        cal = CAL_s(image_list, class_p, thread=10)
        cal.run()
        image_list = []
    print("ok")
