#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : grammer_correction.py
@author: zijun
@contact : stefan_sun_cn@hotmail.com
@date  : 2018/12/31 14:33
@version: 1.0
@desc  : 
"""

'''
训练transformer-CRNN模型
'''

import argparse
import itertools
import math
import random
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import BatchSampler, SequentialSampler
from torchvision import transforms
from tqdm import tqdm

import transformer.Constants as Constants
from config.convert_para import convert_parameters
from crnn_utils.data_augmentation import CRNNAugmentor, AugmentationConfig
from crnn_utils.draw_image import inference
from dataset import TranslationDataset, paired_collate_fn
from transformer.Models import CRNN_Transformer
from transformer.Optim import ScheduledOptim


def generate_augment():
    """生成augment"""
    augmentation_config = AugmentationConfig()
    # augmentation_config.rotate_image_params = {'max_angel': 0}  # no rotation!
    augmentation_config.add_table_lines_params = {'add_line_prob': 0.7, 'max_line_thickness': 4}
    augmentation_config.rotate_image_params = {'max_angel': 3}
    augmentation_config.affine_transform_shear_params = {'a': 3}
    augmentation_config.affine_transform_change_aspect_ratio_params = {'ratio': 0.9}
    augmentation_config.brighten_image_params = {'min_alpha': 0.6}
    augmentation_config.darken_image_params = {'min_alpha': 0.6}
    augmentation_config.add_color_filter_params = {'min_alpha': 0.6}
    augmentation_config.add_random_noise_params = {'min_masks': 70, 'max_masks': 90}
    augmentation_config.add_color_font_effect_params = {'beta': 0.6, 'max_num_lines': 90}
    augmentation_config.add_erode_edge_effect_params = {'kernel_size': (3, 3), 'max_sigmaX': 5}
    augmentation_config.add_resize_blur_effect_params = {'resize_ratio_range': (0.9, 1)}
    augmentation_config.add_gaussian_blur_effect_params = {'kernel_size': (3, 3), 'max_sigmaX': 5}
    augmentation_config.add_horizontal_motion_blur_effect_params = {'min_kernel_size': 5, 'max_kernel_size': 8}
    augmentation_config.add_vertical_motion_blur_effect_params = {'min_kernel_size': 5, 'max_kernel_size': 8}
    augmentation_config.add_random_circles_params = {'min_alpha': 0.6, 'max_num_circles': 25}
    augmentation_config.add_random_lines_params = {'min_alpha': 0.6, 'max_num_lines': 25}

    # augmentation (make the image look blurry)
    data_augmentor = CRNNAugmentor(augmentation_config)
    return data_augmentor


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    """cross entropy 损失函数计算 """
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def train_epoch(model, training_data, optimizer, device, idx2word, smoothing):
    """训练一个epoch"""
    data_augmentor = generate_augment()
    loader = transforms.Compose([transforms.ToTensor()])
    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]
        # 根据tgt_seq生成图片
        text = []
        for i in range(tgt_seq.size(0)):
            text_seq = tgt_seq[i]
            text.append("".join([idx2word[idx] for idx in text_seq.tolist()[1:-1]]))

        image = inference(text[0], '/data/nfsdata/data/sunzijun/CV/more_fonts', data_augmentor)
        image_all = loader(image).unsqueeze(0)

        for i in range(1, src_seq.size(0)):
            image = inference(text[i], '/data/nfsdata/data/sunzijun/CV/more_fonts', data_augmentor)
            image = loader(image).unsqueeze(0)
            image_all = torch.cat((image_all, image), 0)

        image_all = image_all.to(device)  # iamge tensor
        image_padding = torch.ones(src_seq.size(0), 76).type(torch.long).to(device)  # image padding

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos, image_all, image_padding)

        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device, idx2word):
    ''' Epoch operation in evaluation phase '''
    data_augmentor = generate_augment()
    loader = transforms.Compose([transforms.ToTensor()])
    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):
            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]
            # 根据tgt_seq生成图片
            text = []
            for i in range(tgt_seq.size(0)):
                text_seq = tgt_seq[i]
                text.append("".join([idx2word[idx] for idx in text_seq.tolist()[1:-1]]))

            image = inference(text[0], '/data/nfsdata/data/sunzijun/CV/more_fonts', data_augmentor)
            image_all = loader(image).unsqueeze(0)

            for i in range(1, src_seq.size(0)):
                image = inference(text[i], '/data/nfsdata/data/sunzijun/CV/more_fonts', data_augmentor)
                image = loader(image).unsqueeze(0)
                image_all = torch.cat((image_all, image), 0)

            image_all = image_all.to(device)  # iamge tensor
            image_padding = torch.ones(src_seq.size(0), 76).type(torch.long).to(device)  # image padding

            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos, image_all, image_padding)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, device, opt, idx2word):
    """训练过程"""

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + 'train.log'
        log_valid_file = opt.log + 'valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device, idx2word, smoothing=opt.label_smoothing)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu,
            elapse=(time.time() - start) / 60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, idx2word)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu,
            elapse=(time.time() - start) / 60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.log + opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.log + opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu))


def prepare_dataloaders(data, opt):
    # 将数据进行处理
    print("整合数据...")
    src_pre = data['train']['src']
    tgt_pre = data['train']['tgt']
    all_data = list(zip(src_pre, tgt_pre))

    print("sample 数据...")
    sampler = BatchSampler(SequentialSampler(all_data), batch_size=opt.batch_size, drop_last=False)
    index = [s for s in sampler]
    random.shuffle(index)
    index = list(itertools.chain.from_iterable(index))

    print("重新赋值...")
    src = []
    tgt = []
    for i in index:
        src.append(src_pre[i])
        tgt.append(tgt_pre[i])
    data['train']['src'] = src
    data['train']['tgt'] = tgt
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['train']['src'],
            tgt_insts=data['train']['tgt']),
        num_workers=0,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=False)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt']),
        num_workers=0,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default='/data/nfsdata/data/sunzijun/transformer/burry4/data.pt')

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)

    # parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=3)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default='/home/sunzijun/data/')
    parser.add_argument('-save_model', default='trained')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='all')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    # ========= Loading Config=========#
    config = True
    opt = parser.parse_args()
    if config:
        opt = convert_parameters(opt)
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # ========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    # 构造id2word字典
    idx2word = {idx: word for word, idx in data['dict']['tgt'].items()}

    # ========= Preparing Model =========#
    print("************ prepare model ****************")
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'
    with open(opt.log + 'parameters', 'w') as f:
        f.write(str(opt))
    print(opt)

    device_ids = [1, 3]
    device = torch.device('cuda', device_ids[0])

    transformer = CRNN_Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout)

    transformer = transformer.cuda(device_ids[0])  # 模型载入cuda device 0
    transformer = torch.nn.DataParallel(transformer, device_ids=device_ids)  # dataParallel重新包装

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    train(transformer, training_data, validation_data, optimizer, device, opt, idx2word)


if __name__ == '__main__':
    main()
