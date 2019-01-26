#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : reformat_dictionary.py
@author: zijun
@contact : stefan_sun_cn@hotmail.com
@date  : 2018/12/31 15:58
@version: 1.0
@desc  :  重新format字典
"""
import os
import json

def reformat_dict():
    path = '/home/sunzijun/data/grammer_correction'
    # 获取word id对应字典
    with open(os.path.join(path, 'dictionary.json'), 'rb') as f:
        dictionary = json.load(f)

    dictionary['word2idx']['<s>'] = 7858
    dictionary['word2idx']['</s>'] = 7859
    dictionary['word2idx']['<unk>'] = 7860
    dictionary['word2idx']['<blank>'] = 0

    dictionary['idx2word'].append('<s>')
    dictionary['idx2word'].append('</s>')
    dictionary['idx2word'].append('<unk>')
    dictionary['idx2word'][0] = '<blank>'

    print(dictionary['idx2word'][7858])
    print(dictionary['idx2word'][7859])
    print(dictionary['idx2word'][7860])
    print(dictionary['idx2word'][0])

    with open(os.path.join(path, 'format_dictionary.json'), "w", encoding='utf8') as f:
        json.dump(dictionary, f)


if __name__ == '__main__':
    reformat_dict()

