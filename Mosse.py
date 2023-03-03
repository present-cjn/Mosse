# -*- coding: utf-8 -*-
"""
Time:     2023/3/2 10:03
Author:   cjn
Version:  1.0.0
File:     Mosse.py
Describe: 
"""
import numpy as np
import cv2
import os


def xywh2xyxy(x):
    # Convert 1x4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = [x[0],
         x[1],
         x[0] + x[3],
         x[1] + x[2]]

    return y


def yxwh2xyxy(x):
    # Convert 1x4 boxes from [y, x, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = [x[1],
         x[0],
         x[1] + x[3],
         x[0] + x[2]]

    return y


def window_func_2d(height, width):
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)
    win = mask_col * mask_row

    return win


# pre-processing the image...
def pre_process(img):
    # get the size of the img...
    height, width = img.shape
    img = np.log(img + 1)
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    # use the hanning window...
    window = window_func_2d(height, width)
    img = img * window

    return img


def xywh2xyxy(x):
    # Convert 1x4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # 这里的x方向是纵向，y是横向
    y = [x[0] - int(x[3] / 2),
         x[1] - int(x[2] / 2),
         x[0] + int(x[3] / 2),
         x[1] + int(x[2] / 2)]

    return y


def xyxy_scale(x, scale: float):
    # ToDo 实现xyxy的缩放功能
    pass


def get_gauss_response(h, w, sigma):
    """
    获取(hxw)高斯响应图，以中心为峰值、方差为sigma
    :param h, w:获取长宽
    :param sigma:方差
    :return: response
    """
    # 这里的xy和numpy的相反，横向x，纵向y
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    cx = w/2
    cy = h/2
    dist = (np.square(xx - cx) + np.square(yy - cy)) / (2 * sigma)
    # get the response map
    response = np.exp(-dist)
    # normalize
    response = (response - response.min()) / (response.max() - response.min())

    return response


class Mosse:
    def __init__(self, path, lr=0.125, sigma=100, rotate_flag=False):
        self.path = path
        self.lr = lr
        self.sigma = sigma
        self.rotate_flag = rotate_flag
        self.frame_list = self._get_frame_list()

    def start_tracking(self):
        first_frame = cv2.imread(self.frame_list[0], 0)
        first_frame = first_frame.astype(np.float32)
        # 手动标记第一帧的目标
        box = list(cv2.selectROI('demo', first_frame/255.0, False, False))
        box = yxwh2xyxy(box)

        # 获取第一帧的目标图f和其对应的高斯响应图g
        f = first_frame[box[0]:box[2], box[1]:box[3]]
        g = self.get_gauss_response(f.shape[0], f.shape[1])

        # 获取第一帧框选图片的滤波器所需的分子分母
        A, B = self.get_init_ab(f, g)

        # 循环对每一帧进行检测
        for frame_path in self.frame_list[1:]:
            frame = cv2.imread(frame_path, 0)
            frame = frame.astype(np.float32)
            H = A / B
            # Todo 暂时还未完成

    def get_ab(self, f, g):
        F = np.fft.fft2(pre_process(f))
        G = np.fft.fft2(g)

        A = G * np.conjugate(F)
        B = F * np.conjugate(F)

        return A, B

    def get_init_ab(self, f, g):
        A, B = self.get_ab(f, g)

        # Todo 论文中针对第一张图做了一些数据增强，这里的需要设置
        #  1、开关rotate_flag
        #  2、增强的数量，如下面的128
        #  由于不增强的效果就已经还可以了，所以暂时先没写
        if self.rotate_flag:
            for _ in range(128):
                n_f = f  # 这里加入数据变换函数
                new_A, new_B = self.get_AB(n_f, g)
                A = A + new_A
                B = B + new_B

        return A, B

    def _get_frame_list(self):
        frame_list = []
        for frame in os.listdir(self.path):
            if os.path.splitext(frame)[1] == '.jpg':
                frame_list.append(os.path.join(self.path, frame))
        frame_list.sort()
        return frame_list
