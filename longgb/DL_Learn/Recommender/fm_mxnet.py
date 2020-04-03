# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2020/03/22
"""
Usage Of 'fm_mxnet.py' : Factorization machines (FM)
"""

from cf_base_mxnet import CTRDataset

import d2l
from mxnet import autograd, init, gluon, np, npx
from mxnet.gluon import nn
import os
import sys

npx.set_np()

sys.path.append('F:/Codes/ailearn/longgb/DL_Learn/Recommender')


class FM(nn.Block):

    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)  # 这个为什么
        self.linear_layer = nn.Dense(1, use_bias=True)

    def forward(self, x):
        square_of_sum = np.sum(self.embedding(x), axis=1) ** 2  # (2048, 20)
        sum_of_square = np.sum(self.embedding(x) ** 2, axis=1)  # (2048, 20)
        # (2048, 1)
        # x = self.linear_layer(self.fc(x)).sum(axis=1) \
        #     + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        x = npx.sigmoid(x)
        return x


def train_batch_ch13(net, features, labels, loss, trainer, ctx_list,
                     split_f=d2l.split_batch):
    X_shards, y_shards = split_f(features, labels, ctx_list)
    with autograd.record():
        pred_shards = [net(X_shard) for X_shard in X_shards]
        ls = [loss(pred_shard, y_shard) for pred_shard, y_shard
              in zip(pred_shards, y_shards)]
    for l in ls:
        l.backward()
    trainer.step(labels.shape[0])
    train_loss_sum = sum([float(l.sum()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(pred_shard, y_shard)
                        for pred_shard, y_shard in zip(pred_shards, y_shards))
    return train_loss_sum, train_acc_sum


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               ctx_list=d2l.try_all_gpus(), split_f=d2l.split_batch):
    num_batches, timer = len(train_iter), d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # Store training_loss, training_accuracy, num_examples, num_features
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, ctx_list, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0:
                animator.add(epoch + i / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter, split_f)
        animator.add(epoch + 1, (None, None, test_acc))
    print('loss %.3f, train acc %.3f, test acc %.3f' % (
        metric[0] / metric[2], metric[1] / metric[3], test_acc))
    print('%.1f examples/sec on %s' % (
        metric[2] * num_epochs / timer.sum(), ctx_list))


if __name__ == "__main__":
    batch_size = 2048

    data_dir = d2l.download_extract('ctr')
    train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
    test_data = CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)

    num_workers = 0 if sys.platform.startswith('win') else 4
    train_iter = gluon.data.DataLoader(
        train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
        num_workers=num_workers)
    test_iter = gluon.data.DataLoader(
        test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
        num_workers=num_workers)

    ctx = d2l.try_all_gpus()
    net = FM(train_data.field_dims, num_factors=20)
    net.initialize(init.Xavier(), ctx=ctx)

    lr, num_epochs, optimizer = 0.02, 30, 'adam'
    trainer = gluon.Trainer(net.collect_params(), optimizer,
                            {'learning_rate': lr})
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, ctx)
