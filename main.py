# -*- coding:utf-8 -*-
import win32api
import time
import win32con
# print([1,2]==[])
# time.sleep(2)
# win32api.keybd_event(87,0,0,0)  #ctrl键位码是17
# win32api.keybd_event(65,0,0,0)  #v键位码是86
# time.sleep(5)
# win32api.keybd_event(87,0,win32con.KEYEVENTF_KEYUP,0) #释放按键
# # win32api.keybd_event(65,0,win32con.KEYEVENTF_KEYUP,0)

import cv2
import picture as pic
import os
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from load_and_process import proprecess_input
import numpy as np
font = cv2.FONT_HERSHEY_SIMPLEX  # 设置字体
size = 0.5  # 设置大小

width, height = 300, 300  # 设置拍摄窗口大小
x0, y0 = 100, 100  # 设置选取位置
ans_list=['w','wd','d','ds','s','sa','a','aw',' ','0']
map_ans=[]
emotion_model_path = 'models/_mini_XCEPTION.048-0.98.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)


cap = cv2.VideoCapture(0)  # 开摄像头
dataset_path="./dataset_test/"
count=10
# 栈
stack = []
class_number=0
data_counter=0
def pop_check():
    while stack:
        s_pop = stack.pop()
        for s_pop_str in s_pop:
            if s_pop_str == 'w':
                win32api.keybd_event(87, 0, win32con.KEYEVENTF_KEYUP, 0)  # w
            elif s_pop_str == 'd':
                win32api.keybd_event(68, 0, win32con.KEYEVENTF_KEYUP, 0)  # d
            elif s_pop_str == 's':
                win32api.keybd_event(83, 0, win32con.KEYEVENTF_KEYUP, 0)  # s
            elif s_pop_str == 'a':
                win32api.keybd_event(65, 0, win32con.KEYEVENTF_KEYUP, 0)  # a
            elif s_pop_str == ' ':
                pass
            elif s_pop_str == '0':
                pass
            # win32api.keybd_event(65, 0, win32con.KEYEVENTF_KEYUP, 0)
def check(s):
    if s == 'w':
        win32api.keybd_event(87, 0, 0, 0)  # w
    elif s=='d':
        win32api.keybd_event(68, 0, 0, 0)  # d
    elif s=='s':
        win32api.keybd_event(83, 0, 0, 0)  # s
    elif s=='a':
        win32api.keybd_event(65, 0, 0, 0)  # a
    elif s==' ':
        pass
    elif s=='0':
        while stack:
            s_pop=stack.pop()
            for s_pop_str in s_pop:
                if s_pop_str == 'w':
                    win32api.keybd_event(87, 0, win32con.KEYEVENTF_KEYUP, 0)  # w
                elif s_pop_str == 'd':
                    win32api.keybd_event(68, 0, win32con.KEYEVENTF_KEYUP, 0)  # d
                elif s_pop_str == 's':
                    win32api.keybd_event(83, 0, win32con.KEYEVENTF_KEYUP, 0)  # s
                elif s_pop_str == 'a':
                    win32api.keybd_event(65, 0, win32con.KEYEVENTF_KEYUP, 0)  # a
                elif s_pop_str == ' ':
                    pass
                elif s_pop_str == '0':
                    pass
                # win32api.keybd_event(65, 0, win32con.KEYEVENTF_KEYUP, 0)


if __name__ == "__main__":
    if not (os.path.exists(dataset_path)):
        os.mkdir(dataset_path)
    for i in range(count):
        class_path=dataset_path+str(i)
        if not (os.path.exists(class_path)):
            os.mkdir(class_path)
    while (1):
        ret, frame = cap.read()  # 读取摄像头的内容
        frame = cv2.flip(frame, 1)
        # roi,result,newx,newy,newweight,newheight = pic.binaryMask(frame, 0, 0, frame.shape[1], frame.shape[0])
        # roi =
        roi,result,newx,newy,newweight,newheight = pic.binaryMask(frame, x0, y0, width, height)  # 取手势所在框图并进行处理
        key = cv2.waitKey(1) & 0xFF  # 按键判断并进行一定的调整
        # 按'a''d''w''s'分别将选框左移，右移，上移，下移
        # 按'q'键退出录像
        if key == ord('s')and y0<frame.shape[0]-width:
            y0 += 5
            print(y0)
        elif key == ord('w')and y0>0:
            y0 -= 5
            print(y0)
        elif key == ord('d')and x0<frame.shape[1]-height:
            x0 += 5
            print(x0)
        elif key == ord('a')and x0>0:
            x0 -= 5
            print(x0)
        elif key == ord('c'):
            file_name = dataset_path + str(class_number) + "/" + "%03d.png" % data_counter
            # 上线999
            cv2.imwrite(file_name,roi)
            print("保存 图片 %d" % data_counter)
            data_counter = data_counter + 1
        elif key == ord('n'):
            class_number = class_number + 1
            data_counter = 0
            print("保存 class %d" % class_number)
            print("采集完毕")
        elif key == ord('q'):
            break
        elif result==True:
            # 给模型传入
            # print(roi.shape)
            roi=cv2.resize(roi,(300,300))
            # print("mohenghui",roi.shape)
            roi=roi.reshape((300,300,1))
            roi = proprecess_input(roi)
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            out = emotion_classifier.predict(roi)[0]
            roi=roi.reshape((300,300,1))
            cv2.imshow("crop",roi)
            # print("result:%d %s" % (out, ans_list[out[0]]))
            # print("result:%d", out)
            # print(type(out))
            # cv2.rectangle(frame, (newx, newy), (newx + newweight, newy + newheight), (0, 255, 0), 3)
            forward=ans_list[out.argmax()]
            cv2.putText(frame,forward, (newx, newy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            if  not stack:
                stack.append(forward)
                for s_forward in forward:
                    check(s_forward)
            else:
                stack_str=stack[-1]
                if stack_str!=forward:
                    pop_check()
                    stack.append(forward)
                    for s_forward in forward:
                        check(s_forward)
            cv2.imshow("crop", roi)
            print(stack)
        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0), 3)
        # 推出图像
        cv2.imshow('frame', frame)  # 播放摄像头的内容
    cap.release()
    cv2.destroyAllWindows()  # 关闭所有窗口
