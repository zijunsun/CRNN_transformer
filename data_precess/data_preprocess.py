#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : data_preprocess.py
@author: zijun
@contact : stefan_sun_cn@hotmail.com
@date  : 2018/12/31 14:39
@version: 1.0
@desc  : 将文件中的文字转为数字
"""
import json
import os

from tqdm import tqdm


def convert_word_to_id():
    path = '/home/sunzijun/data/grammer_correction'

    # 获取word id对应字典
    with open(os.path.join(path, 'format_dictionary.json'), 'rb') as f:
        dictionary = json.load(f)
        word2idx = dictionary['word2idx']
        idx2word = dictionary['idx2word']

    # 处理数据
    modes = ['train', 'valid']
    data = {}
    for mode in modes:
        print("********* process ", mode, " data *********")
        # 读取数据转化为id格式
        src = []
        with open(os.path.join(path, mode + '_src'), 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(tqdm(lines, desc='processing src:')):
                single_data = [word2idx[c] for c in line.strip()]
                single_data.insert(0, 7858)
                if len(single_data) > 64:
                    # 如果长度比64长
                    single_data = single_data[:63]
                single_data.append(7859)
                src.append(single_data)
        tgt = []
        with open(os.path.join(path, mode + '_tgt'), 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(tqdm(lines, desc='processing tgt:')):
                single_data = [word2idx[c] for c in line.strip()]
                single_data.insert(0, 7858)
                if len(single_data)> 64:
                    # 如果长度比64长
                    single_data = single_data[:63]
                single_data.append(7859)
                tgt.append(single_data)

        data[mode] = {
            'src': src,
            'tgt': tgt
        }
    with open(os.path.join(path, 'train.json'), "w", encoding='utf8') as f:
        json.dump(data, f)


def data_compare():
    path = '/home/sunzijun/data/grammer_correction'
    with open(os.path.join(path, 'valid_src'), 'r') as f:
        train_src = f.readlines()
    with open(os.path.join(path, 'valid_tgt'), 'r') as f:
        train_tgt = f.readlines()

    count = 0
    for i, line in enumerate(tqdm(train_src)):
        if train_src[i] != train_tgt[i]:
            count += 1
            if count % 1000 == 0:
                print(i)
                print(train_src[i])
                print(train_tgt[i])
                print()
    print(count)
    print(len(train_src))
    print(count / len(train_src))


if __name__ == '__main__':
    convert_word_to_id()
    # data_compare()
