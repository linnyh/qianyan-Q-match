# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import math
import numpy as np
from paddlenlp.data import Pad
import paddle.nn.functional as F

import paddlenlp as ppnlp

'''
    version: 0.0.1
    将bert的输出直接通过lstm编码，后与cls位输出表示concat，效果不好，模型欠拟合
    version: 0.0.2
    将bert对于两个句子的输出分别通过bilstm编码后经过word级别注意力生成上下文，后与cls输出表示concat
    version: 0.1.0
    改为双塔模型
'''


# 创建mask
def generate_sent_masks(batch_size, max_seq_length, source_lengths):
    """ Generate sentence masks for encoder hidden states.
        returns enc_masks (Tensor): Tensor of sentence masks of shape (b, max_seq_length),where max_seq_length = max source length """
    enc_masks = paddle.zeros(batch_size, max_seq_length, dtype=paddle.float64)
    for e_id, src_len in enumerate(source_lengths):
        enc_masks[e_id, :src_len] = 1
    return enc_masks


def find_split_idx(data, flag):
    if flag:
        for i in range(len(data)):
            if data[i] == 1:
                return i
    else:
        for i in range(len(data) - 1, -1, -1):
            if data[i] == 1:
                return i


# python 用的不熟，只能自己造轮子了，有更好的方法请重写它
def split_bert_output(encoder_output, token_type_ids):
    split_ids = []  # 拆分点
    sentence_a = []
    sentence_b = []
    for lt in token_type_ids:
        split_ids.append([find_split_idx(lt, True), find_split_idx(lt, False)])
    i = 0
    max_len_a = 0
    max_len_b = 0
    max_len = 0
    for sp in range(len(encoder_output)):
        s_a = encoder_output[sp][1:split_ids[i][0], :]
        s_b = encoder_output[sp][split_ids[i][0]:split_ids[i][1] if split_ids[i][1] < len(encoder_output[sp]) else -1,
              :]
        max_len_a = max(max_len_a, s_a.shape[0])
        max_len_b = max(max_len_b, s_b.shape[0])
        max_len = max(max_len, max_len_a, max_len_b)
        sentence_a.append(s_a)
        sentence_b.append(s_b)
        i = i + 1
    pad = encoder_output.shape[2] * [0.0]
    for sp in range(len(encoder_output)):
        if len(sentence_a[sp]) < max_len:
            sentence_a[sp] = paddle.concat(
                x=[sentence_a[sp], paddle.to_tensor([pad] * (max_len - len(sentence_a[sp])))], axis=0)
            # sentence_a[sp] = F.pad(sentence_a[sp], [0, 0, 0, max_len - len(sentence_a[sp]), 0, 0])
        if len(sentence_b[sp]) < max_len:
            sentence_b[sp] = paddle.concat(
                x=[sentence_b[sp], paddle.to_tensor([pad] * (max_len - len(sentence_b[sp])))], axis=0)
            # sentence_b[sp] = F.pad(sentence_b[sp], [0, 0, 0, max_len - len(sentence_b[sp]), 0, 0])
    sentence_a = paddle.to_tensor(sentence_a)
    sentence_b = paddle.to_tensor(sentence_b)
    return sentence_a, sentence_b


class QuestionMatching(nn.Layer):
    def __init__(self, pretrained_model, dropout=None, rdrop_coef=0.0):
        super().__init__()
        self.ptm = pretrained_model  # 预训练模型
        self.hidden_size = self.ptm.config['hidden_size']
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.5)
        self.lstm = MatchLSTM(self.ptm.config["hidden_size"], self.ptm.config["hidden_size"], 1)
        self.word_attention = WordAttention(input_size=self.hidden_size, hidden_size=256, rnn_layers=1)
        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Linear(self.hidden_size * 2, 512)  # 线性分类层
        self.classifier2 = nn.Linear(512, 2)
        self.rdrop_coef = rdrop_coef
        self.rdrop_loss = ppnlp.losses.RDropLoss()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                do_evaluate=False):

        encoder_output, cls_embedding1 = self.ptm(input_ids, token_type_ids, position_ids,
                                                  attention_mask)  # [batch_size = 32,sentence_pair_length,hidden_state = 768] [batch_size = 32,768]

        # 拆分句子嵌入
        sentence_a, sentence_b = split_bert_output(encoder_output, token_type_ids)

        att = self.word_attention(sentence_a, sentence_b, self.ptm.config['hidden_size'])
        print("attention:")
        print(att.shape)

        # 将cls与上下文连接
        classifier_input = paddle.concat(x=[cls_embedding1, att], axis=1)

        print("classifier_input:")
        print(classifier_input.shape)

        final = self.dropout(classifier_input)  # [32,768]
        '''
            分类层改进
        '''
        logits1 = self.classifier(final)  # [batch_size, 2] 二分类
        logits1 = self.classifier2(logits1)

        # For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448
        # Original implementation please refer to this code: https://github.com/dropreg/R-Drop
        if self.rdrop_coef > 0 and not do_evaluate:
            _, cls_embedding2 = self.ptm(input_ids, token_type_ids, position_ids,
                                         attention_mask)
            cls_embedding2 = self.dropout(cls_embedding2)
            logits2 = self.classifier(cls_embedding2)
            kl_loss = self.rdrop_loss(logits1, logits2)
        else:
            kl_loss = 0.0

        return F.softmax(logits1, axis=1), kl_loss


class MatchLSTM(nn.Layer):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, time_major=False,
                            direction="forward")  # 单层双向lstm

    def forward(self, x):
        x = self.lstm(x)
        return x


class WordAttention(nn.Layer):  # 通过将MatchLSTM 的输出做一个词级别attention
    def __init__(self, input_size, hidden_size, rnn_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.u_T = nn.Linear(input_size, hidden_size)
        self.w_a = nn.Linear(input_size, hidden_size)
        self.w_b = nn.Linear(input_size, hidden_size)
        self.layer_norm = nn.LayerNorm(input_size)
        self.lstm = MatchLSTM(input_size, hidden_size, rnn_layers)  # 编码用的lstm
        self.scores = None

    def attention(self, q, k, v, d_k, mask=None):
        scores = paddle.matmul(q, paddle.transpose(k, (0, 2, 1))) / math.sqrt(d_k)
        scores = paddle.tanh(scores)
        self.scores = scores
        # return paddle.matmul(scores, v)
        return scores

    def forward(self, a, b, d_k, mask=None):
        bs = a.shape[0]
        max_len = max(a.shape[1], b.shape[1])
        a = F.pad(a, [0, 0, 0, max_len - a.shape[1], 0, 0])
        b = F.pad(b, [0, 0, 0, max_len - b.shape[1], 0, 0])

        q = F.softmax(self.w_a(a), axis=2)
        k = F.softmax(self.w_b(b), axis=2)

        scores = self.attention(q, k, None, 768)  # (batch, len, len)
        c = paddle.sum(paddle.matmul(scores, b), axis=1)  # (batch, hidden_state)

        c = self.layer_norm(c)

        return c
