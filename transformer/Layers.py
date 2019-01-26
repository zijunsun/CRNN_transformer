''' Define the Layers '''
import torch.nn as nn
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward, ConvRelu

__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        # self attention层
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask
        # feed forward 层
        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        # self attention层
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask
        # encoder-decoder attention层
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask
        # feed forward层
        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn


class CNNLayer(nn.Module):
    """CNN 模型"""

    def __init__(self):
        super(CNNLayer, self).__init__()
        cnn = nn.Sequential()
        cnn.add_module('Conv_Relu 0',
                       ConvRelu(n_in=3, n_out=64, kernal_size=3, stride_size=1, padding_size=1))               # 卷积
        cnn.add_module('pooling 0', nn.MaxPool2d(2, 2))                                                        # 池化
        cnn.add_module('Conv_Relu 1',
                       ConvRelu(n_in=64, n_out=128, kernal_size=3, stride_size=1, padding_size=1))             # 卷积
        cnn.add_module('pooling 1', nn.MaxPool2d(2, 2))                                                        # 池化
        cnn.add_module('Conv_Relu 2',
                       ConvRelu(n_in=128, n_out=256, kernal_size=3, stride_size=1, padding_size=1, bn=True))   # 卷积
        cnn.add_module('Conv_Relu 3',
                       ConvRelu(n_in=256, n_out=256, kernal_size=3, stride_size=1, padding_size=1))            # 卷积
        cnn.add_module('pooling 2'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))                            # 池化
        cnn.add_module('Conv_Relu 4',
                       ConvRelu(n_in=256, n_out=512, kernal_size=3, stride_size=1, padding_size=1, bn=True))   # 卷积
        cnn.add_module('Conv_Relu 5',
                       ConvRelu(n_in=512, n_out=512, kernal_size=3, stride_size=1, padding_size=1))            # 卷积
        cnn.add_module('pooling 3'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))                            # 池化
        cnn.add_module('Conv_Relu 6',
                       ConvRelu(n_in=512, n_out=512, kernal_size=2, stride_size=1, padding_size=0, bn=True))   # 卷积
        self.cnn = cnn

    def forward(self, x):
        return self.cnn(x)


