# -*- coding: utf-8 -*-
"""
Time:     2023/3/6 8:50
Author:   cjn
Version:  1.0.0
File:     tracking.py
Describe: 
"""
from Mosse import Mosse
import argparse

parser = argparse.ArgumentParser(description='命令行参数')
parser.add_argument('--path', '-p', default='./data/david', type=str, help='视频的地址')

args = parser.parse_args()

if __name__ == "__main__":
    path = args.path
    tracker = Mosse(path=path)
    tracker.start_tracking()
