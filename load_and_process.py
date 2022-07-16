# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import cv2
import glob
import os
def proprecess_input(x,v2=True):
    x=x.astype('float32')
    x=x/255.0
    if v2:
        x=x-0.5
        x=x*2.0
    return x
def dense_to_one_hot(label, num_class):
    num_label = label.shape[0]
    print(num_label)
    index_offset = np.arange(num_label) * num_class
    label_one_hot = np.zeros((num_label,num_class))
    print(label_one_hot)
    label_one_hot.flat[index_offset + label.ravel()] = 1
    return label_one_hot

def imread_img(filename, flags=cv2.IMREAD_COLOR):
    return cv2.imread(filename, flags)

def data_reader(dataset_path):
    # 图片数据
    data_list=[]
    # 图片结果
    label_list=[]
    for cls_path in glob.glob(os.path.join(dataset_path,"*")):
        print("文件名字",cls_path)
        # 因为可能有其他格式图片
        for file_name in glob.glob(os.path.join(cls_path,"*")):
            img=cv2.imread(file_name,0)
            img=img.reshape((300,300,1))
            # cv2.imshow(file_name,img)
            ref=img
            data_list.append(ref)
            label_list.append(int(cls_path.split('\\')[-1]))
            # print("234",file_name)
    data_np=np.array(data_list)
    label_np=np.array(label_list)
    return data_np,label_np