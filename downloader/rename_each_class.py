# -*-coding: utf-8 -*-
import cv2
from skimage import io
import os


def cvt_data_from_downloads(downloads, cvt_data, cls_name):
    origin_path = downloads + '/' + cls_name
    save_path = cvt_data + '/' + cls_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    i = 0
    for filename in os.listdir(origin_path):
        try:
            origin = io.imread(origin_path + '/' + filename)
            origin_im = cv2.imread(origin_path + '/' + filename)
            resize_im = cv2.resize(origin_im, (512, 512), interpolation=cv2.INTER_CUBIC)
            save_im_path = save_path + f'/{cls_name}_' + str(i) + '.jpg'
            cv2.imwrite(save_im_path, resize_im)
            print(i, filename)
            i += 1
        except:
            print(origin_path + '/' + filename, "wrong!")


if __name__ == '__main__':
    downloads = './downloads'
    cvt_data = './cvt_data'
    for cls_name in os.listdir(downloads):
        cvt_data_from_downloads(downloads, cvt_data, cls_name)
