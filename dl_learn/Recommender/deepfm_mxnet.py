# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2020/03/22
"""
Usage Of 'deepfm_mxnet.py' : 
"""

from cf_base_mxnet import CTRDataset
from fm_mxnet import train_ch13

import d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os
import sys

npx.set_np()

sys.path.append('F:/Codes/ailearn/longgb/DL_Learn/Recommender')


class DeepFM(nn.Block):

    def __init__(self, field_dims, num_factors, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()

        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)

        input_dim = self.embed_output_dim = len(field_dims) * num_factors
        self.mlp = nn.Sequential()
        for dim in mlp_dims:
            self.mlp.add(nn.Dense(dim, 'relu', True, in_units=input_dim))
            self.mlp.add(nn.Dropout(rate=drop_rate))
            input_dim = dim
        self.mlp.add(nn.Dense(in_units=input_dim, units=1))  # linear

    def forward(self, x):
        embed_x = self.embedding(x)  # (2048, 34, 10)
        square_of_sum = np.sum(embed_x, axis=1) ** 2  # (2048, 10)
        sum_of_square = np.sum(embed_x ** 2, axis=1)  # (2048, 10)
        inputs = np.reshape(
            embed_x, (-1, self.embed_output_dim))  # (2048, 340)
        # (2048, 1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True) \
            + self.mlp(inputs)
        x = npx.sigmoid(x)
        return x


if __name__ == "__main__":
    batch_size = 2048

    data_dir = d2l.download_extract('ctr')
    train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
    test_data = CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
    field_dims = train_data.field_dims

    num_workers = 0 if sys.platform.startswith('win') else 4
    train_iter = gluon.data.DataLoader(train_data, shuffle=True,
                                       last_batch='rollover',
                                       batch_size=batch_size,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_data, shuffle=False,
                                      last_batch='rollover',
                                      batch_size=batch_size,
                                      num_workers=num_workers)

    ctx = d2l.try_all_gpus()
    net = DeepFM(field_dims, num_factors=10, mlp_dims=[30, 20, 10])
    net.initialize(init.Xavier(), ctx=ctx)

    lr, num_epochs, optimizer = 0.01, 30, 'adam'
    trainer = gluon.Trainer(net.collect_params(), optimizer,
                            {'learning_rate': lr})
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, ctx)
