# -*- encoding:utf-8 -*-
import os
import sqlite3, threadpool
import random
from urllib import request


class PengDownloader:
    def __init__(self, urls, folder, threads=10):
        self.urls = urls
        self.folder = folder
        self.threads = threads

    def run(self):
        pool = threadpool.ThreadPool(self.threads)
        requests = threadpool.makeRequests(self.downloader, self.urls)
        [pool.putRequest(i) for i in requests]
        pool.wait()

    def downloader(self, url):
        pre = url.split('/')[-1]
        name = pre if pre.split(".")[-1] in ["jpg", "jpeg", "png", "bmp"] else pre + ".jpg"
        print(self.folder + name)
        self.auto_down(url, self.folder + name)

    def auto_down(self, url, filename):
        try:
            filename = self.folder + ''.join(random.sample(
                ['z', 'y', 'x', 'w', 'v', 'u', 't', 's', 'r', 'q', 'p', 'o', 'n', 'm', 'l', 'k', 'j', 'i', 'h', 'g',
                 'f', 'e', 'd', 'c', 'b', 'a'], 10)) + '.jpg'
            request.urlretrieve(url, filename)
        except Exception as e:
            print('Network Error, redoing download :' + url)
            self.auto_down(url, filename)


if __name__ == "__main__":
    conn = sqlite3.connect("./database/link.db")
    keys = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    for k in keys:
        urls = []
        cursor = conn.execute("SELECT LINK FROM LINK WHERE CLASS = '{}'".format(k))
        for row in cursor:
            urls.append(row[0])
        save_path = "./downloads/{}/".format(k)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pd = PengDownloader(urls, save_path, threads=200)
        pd.run()
