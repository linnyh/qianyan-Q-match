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
import paddle.nn.functional as F

import paddlenlp as ppnlp


class QuestionMatching(nn.Layer):
    def __init__(self, pretrained_model, dropout=None, rdrop_coef=0.0):
        super().__init__()
        self.ptm = pretrained_model  # 预训练模型
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.lstm = MatchLSTM(self.ptm.config["hidden_size"], self.ptm.config["hidden_size"], 1)
        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Linear(self.ptm.config["hidden_size"] * 2, 2)  # 线性分类层
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

        '''
            将 encoder_output （各个token的嵌入表示） 作为输入
            在bert后上添加自己的东西
        '''
        output, final_states = self.lstm(encoder_output)
        # print(final_states[0])
        final_state = paddle.squeeze(x=final_states[0], axis=0)
        final = paddle.concat(x=[cls_embedding1, final_state], axis=1)
        print(final.shape)

        final = self.dropout(final)  # [32,768]
        '''
            分类层改进
        '''
        logits1 = self.classifier(final)  # [batch_size, 2] 二分类

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

        return logits1, kl_loss


class MatchLSTM(nn.Layer):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, time_major=False)

    def forward(self, x):
        x = self.lstm(x)
        return x


class CrossAttention(nn.Layer):
    def __init__(self):
        self.rnn = nn.LSTM()

    def forward(self, *inputs, **kwargs):
        pass
