''' Handling the data io '''
import argparse

import torch
import transformer.Constants as Constants


# keep_case: 保持原来的大小写


def read_instances_from_file(inst_file, max_sent_len, keep_case):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    trimmed_sent_count = 0
    # 把 f 中的句子分成词，控制长度后加入 word_insts 中
    # word_insts: a list of words
    with open(inst_file, encoding='utf8') as f:
        # 对 f 中每个句子处理
        for sent in f:
            # 如果不要求 keep_case 就对句子进行小写处理
            # if not keep_case:
            #     sent = sent.lower()
            # 分词
            # words = sent.split()
            words = list(sent.strip())
            # 如果词数大于上限， trimmed_sent_count += 1, 因为会截取词到上限
            if len(words) > max_sent_len:
                trimmed_sent_count += 1
            # 获得句子中的词语 截取到上限
            word_inst = words[:max_sent_len]

            # 如果 word_inst 非空，头尾加上 <s> </s>
            if word_inst:
                word_insts += [[Constants.BOS_WORD] +
                               word_inst + [Constants.EOS_WORD]]
            else:
                word_insts += [None]

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    # 如果有截断现象，打印输出截断的句子数
    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))
    # output: a list of words
    return word_insts


def build_vocab_idx(word_insts, min_word_count):
    ''' Trim vocab by number of occurence '''
    # 将 word_insts 中的所有词放到 set 中，并打印词袋大小
    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    # 先加入 constants
    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    # 首先将所有词的 count 设为 0
    word_count = {w: 0 for w in full_vocab}

    # 遍历所有词，更新 count
    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    # 用于记录未加入词典的词语数量
    ignored_word_count = 0
    # 将 word_insts 中的词加入字典，但是只加入 count 大于 min_word_count 的词
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    # 输出截断的具体情况
    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx


def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    # return [s for s in word_insts]
    return [[word2idx.get(w, Constants.UNK) for w in s] if s else [Constants.PAD] for s in word_insts]


def main():
    ''' Main function '''

    # 设置命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', default='/home/sunzijun/data/CRNN_trans/train_src')
    parser.add_argument('-train_tgt', default='/home/sunzijun/data/CRNN_trans/train_tgt')
    parser.add_argument('-valid_src', default='/home/sunzijun/data/CRNN_trans/valid_src')
    parser.add_argument('-valid_tgt', default='/home/sunzijun/data/CRNN_trans/valid_src')
    parser.add_argument('-save_data', default='/home/sunzijun/data/CRNN_trans/data.pt')
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=64)
    parser.add_argument('-min_word_count', type=int, default=0)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len + 2  # include the <s> and </s>

    DEBUG = True
    if DEBUG:
        opt.min_word_count = 0
        opt.share_vocab = True
    # return torch.load(opt.save_data)

    # Training set
    # 获取 source 的 list of words
    train_src_word_insts = read_instances_from_file(
        opt.train_src, opt.max_word_seq_len, opt.keep_case)
    # 获取 target 的 list of words
    # [['a','boy'],['toy','superman']]
    train_tgt_word_insts = read_instances_from_file(
        opt.train_tgt, opt.max_word_seq_len, opt.keep_case)
    # 检查 source 和 target 句子数量是否一致，如果不一致 报 warning，并将两组句子截到相同数目
    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts),
                             len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    # - Remove empty instances
    # 去除 source 和 target 中任意空的元组，返回 a list of tuples
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    # 将上述 pipeline 在 valid set 上重复一遍
    # Validation set
    valid_src_word_insts = read_instances_from_file(
        opt.valid_src, opt.max_word_seq_len, opt.keep_case)
    valid_tgt_word_insts = read_instances_from_file(
        opt.valid_tgt, opt.max_word_seq_len, opt.keep_case)

    if len(valid_src_word_insts) != len(valid_tgt_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts),
                             len(valid_tgt_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    # - Remove empty instances
    valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t]))

    # Build vocabulary
    # 如果命令行参数中设定了 vocab
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        # 确保 dict 在 predefined_data 中
        # predefined_data 存的到底是什么格式？？？？
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        # 获取 word2idx both source & target
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']

    # 命令行参数没有设定 vocab
    else:
        # 如果命令行选择了 share_vocab ， source 和 target 共享一个 vocab，本质上就是用 source 和 target 的词表一起生成 vocab
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
            # src_word2idx = tgt_word2idx = word2idx
        # 不 share vocab，各自生成 vocab
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(
                train_src_word_insts, opt.min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(
                train_tgt_word_insts, opt.min_word_count)

    # data = {
    #     'settings': opt,
    #     'dict': {
    #         'src': src_word2idx,
    #         'tgt': tgt_word2idx}}

    # # 存到文件里
    # print('[Info] Dumping the processed data to pickle file', opt.save_data)
    # torch.save(data, opt.save_data)
    # print('[Info] Finish.')

    # return

    # data = torch.load(opt.save_data)
    # src_word2idx = data['dict']['src']
    # tgt_word2idx = data['dict']['tgt']

    # word to index
    # 保留 insts 的格式，但是将词变成 index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(
        train_src_word_insts, word2idx)
    valid_src_insts = convert_instance_to_idx_seq(
        valid_src_word_insts, word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(
        train_tgt_word_insts, word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(
        valid_tgt_word_insts, word2idx)

    # data['train'] = {'src': train_src_insts,
    #                  'tgt': train_tgt_insts}
    # data['valid'] = {'src': valid_src_insts,
    #                  'tgt': valid_tgt_insts}

    # 把 data 编成 dict 形式

    data = {
        'settings': opt,
        'dict': {
            'src': word2idx,
            'tgt': word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}

    # 存到文件里
    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

    return data


if __name__ == '__main__':
    data = main()
