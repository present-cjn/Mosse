# -*- coding: utf-8 -*-
"""
Time:     2023/3/6 8:50
Author:   cjn
Version:  1.0.0
File:     tracking.py
Describe: 
"""
from Mosse import Mosse

if __name__ == "__main__":
    path = './data/david'
    tracker = Mosse(path=path)
    tracker.start_tracking()
