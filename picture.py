# -*- coding:utf-8 -*-
import cv2
import numpy as np

def chose_licence_plate(contours, Min_Area=10000,Max_Area=45000):
    # 根据侧拍的物理特征对所得矩形进行过滤
    # 输入：contours每个轮廓列表的特征是一个三维数组 N*1*2
    # 输出：返回经过过滤后的轮廓集合
    temp_contours = []
    for contour in contours:
        # 对符合面积要求的巨型装进list
        contoursize=cv2.contourArea(contour)
        if contoursize > Min_Area and contoursize < Max_Area:
            # print("矩形框大小",contoursize)
            temp_contours.append(contour)
    car_plate = []
    for item in temp_contours:
        # cv2.boundingRect用一个最小的矩形，把找到的形状包起来
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        # 440mm×140mm
        # print("宽高比：",weight/height)
        if (weight > (height * 0.45)) and (weight < (height * 1.5)):
            car_plate.append(item)
    # 返回车牌列表
    return car_plate

def draw(list,image,x0,y0):
    newx=0
    newy=0
    weight=0
    height=0
    for item in list:
        # cv2.boundingRect用一个最小的矩形，把找到的形状包起来
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        newx=x+x0
        newy=y+y0
        # print("新%sx,新%sy"%(newx,newy))
    return newx,newy,weight,height
        # cv2.imshow("newimage",image)
def binaryMask(frame, x0, y0, width, height):
    result = False
    framecopy=frame.copy()
    # cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 0, 255),3)  # 画出截取的手势框图
    roi = framecopy[y0:y0 + height, x0:x0 + width]  # 获取手势框图
    # print(roi.shape)
    cv2.imshow("roi", roi)  # 显示手势框图 传bgr转rgb
    res = skinMask2(roi)  # 进行肤色检测
    # res1 = skinMask2(framecopy)  # 进行肤色检测
    image2, contours, hierarchy = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    list = chose_licence_plate(contours,10000,45000)
    # print("长度",len(list))
    if(len(list)!=0):
        result=True
    newx,newy,newweight,newheight=draw(list,frame,x0,y0)
    # cv2.drawContours(image1, contours, -1, (0, 255, 0), 5)
    # cv2.imshow("shou",image1)
    # cv2.imshow("res", res)  # 显示肤色检测后的图像
    return res,result,newx,newy,newweight,newheight


##########方法一###################
##########BGR空间的手势识别#########
def skinMask1(roi):
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # 转换到RGB空间
    (R, G, B) = cv2.split(rgb)  # 获取图像每个像素点的RGB的值，即将一个二维矩阵拆成三个二维矩阵
    skin = np.zeros(R.shape, dtype=np.uint8)  # 掩膜
    (x, y) = R.shape  # 获取图像的像素点的坐标范围
    for i in range(0, x):
        for j in range(0, y):
            # 判断条件，不在肤色范围内则将掩膜设为黑色，即255
            if (abs(R[i][j] - G[i][j]) > 15) and (R[i][j] > G[i][j]) and (R[i][j] > B[i][j]):
                if (R[i][j] > 95) and (G[i][j] > 40) and (B[i][j] > 20) \
                        and (max(R[i][j], G[i][j], B[i][j]) - min(R[i][j], G[i][j], B[i][j]) > 15):
                    skin[i][j] = 255
                elif (R[i][j] > 220) and (G[i][j] > 210) and (B[i][j] > 170):
                    skin[i][j] = 255
    res = cv2.bitwise_and(roi, roi, mask=skin)  # 图像与运算

    return res


# 方法二#
# YCrCb之Cr分量 + OTSU二值化
def skinMask2(roi):
    ycrcb=cv2.cvtColor(roi,cv2.COLOR_BGR2YCrCb)
    (y,cr,cb)=cv2.split(ycrcb)
    cr1=cv2.GaussianBlur(cr,(3,3),0) #对cr通道进行高斯滤波
    _,skin1=cv2.threshold(cr1,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    # 去除一些小的白点
    # kernel = np.ones((10,5), np.uint8)
    # img_edge1 = cv2.morphologyEx(skin1, cv2.MORPH_CLOSE, kernel)
    # # img_edge1 = cv2.morphologyEx(img_edge1, cv2.MORPH_CLOSE, kernel)
    # # imshow_img("img_edge3", img_edge1)
    # skin1 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    return skin1