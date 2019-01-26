#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : process.py
@author: zijun
@contact : stefan_sun_cn@hotmail.com
@date  : 2019/1/14 23:33
@version: 1.0
@desc  : 
"""
import torch
preprocess_data = torch.load('/home/sunzijun/data/data.pt')

data = {
        'settings': preprocess_data['settings'],
        'dict': {
            'src': preprocess_data['dict']['src'],
            'tgt': preprocess_data['dict']['tgt']},
        'train': {
            'src': [],
            'tgt': []},
        'valid': {
            'src': [],
            'tgt': []}}

# 存到文件里
torch.save(data, '/home/sunzijun/dict.pt')
print('[Info] Finish.')