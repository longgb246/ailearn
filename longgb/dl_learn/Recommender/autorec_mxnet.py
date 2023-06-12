# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2020/03/21
"""
Usage Of 'autorec_mxnet.py' : AutoRec: Rating Prediction with Autoencoders

"""

from cf_base_mxnet import read_data_ml100k, split_data_ml100k, load_data_ml100k
from mf_mxnet import train_recsys_rating

import d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import sys

npx.set_np()

sys.path.append('F:/Codes/ailearn/longgb/DL_Learn/Recommender')


class AutoRec(nn.Block):

    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        # 初始化的dense不用指定input维度，但是一次运算之后，网络就固定了
        # The activation of encoder is set to sigmoid by default
        self.encoder = nn.Dense(num_hidden, activation='sigmoid',
                                use_bias=True)
        # no activation is applied for decoder
        self.decoder = nn.Dense(num_users, use_bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training():  # mask the gradient during training.
            # The gradients of unobserved inputs are masked out to ensure that only observed ratings contribute to the model learning process.
            return pred * np.sign(input)
        else:
            return pred


def evaluator(network, inter_matrix, test_data, ctx):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, ctx, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Calculate the test RMSE.
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)


if __name__ == "__main__":
    ctx = d2l.try_all_gpus()

    # Load the MovieLens 100K dataset
    df, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(df, num_users, num_items)
    _, _, _, train_inter_mat = load_data_ml100k(train_data, num_users,
                                                num_items)
    _, _, _, test_inter_mat = load_data_ml100k(test_data, num_users,
                                               num_items)

    num_workers = 0 if sys.platform.startswith("win") else 4
    train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True,
                                       last_batch="rollover", batch_size=256,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(np.array(train_inter_mat), shuffle=False,
                                      last_batch="keep", batch_size=1024,
                                      num_workers=num_workers)

    # Model initialization, training, and evaluation
    net = AutoRec(500, num_users)
    net.initialize(ctx=ctx, force_reinit=True, init=mx.init.Normal(0.01))
    lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
    loss = gluon.loss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), optimizer,
                            {"learning_rate": lr, 'wd': wd})
    train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        ctx, evaluator, inter_mat=test_inter_mat)
