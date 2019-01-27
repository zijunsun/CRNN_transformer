#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : convert_para.py
@author: zijun
@contact : stefan_sun_cn@hotmail.com
@date  : 2019/1/27 17:58
@version: 1.0
@desc  : 
"""
import yaml


def convert_parameters(opt):
    with open('config/config.yaml', 'r') as f:
        paras = yaml.load(f.read())['train_paras']
    opt.data = paras['data']
    opt.log = paras['log']
    opt.epoch = paras['epoch']
    opt.batch_size = paras['batch_size']
    opt.d_word_vec = paras['d_word_vec']
    opt.d_model = paras['d_model']
    opt.d_inner_hid = paras['d_inner_hid']
    opt.d_k = paras['d_k']
    opt.d_v = paras['d_v']
    opt.n_head = paras['n_head']
    opt.n_layers = paras['n_layers']
    opt.n_warmup_steps = paras['n_warmup_steps']
    opt.dropout = paras['dropout']
    opt.save_model = paras['save_model']
    opt.save_mode = paras['save_mode']
    opt.embs_share_weight = paras['embs_share_weight']
    opt.proj_share_weight = paras['proj_share_weight']
    opt.no_cuda = paras['no_cuda']
    opt.label_smoothing = paras['label_smoothing']

    return opt
