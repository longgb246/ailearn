# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2020/03/22
"""
Usage Of 'caser_mxnet.py' : 
"""

from cf_base_mxnet import read_data_ml100k, split_data_ml100k, load_data_ml100k
from mf_mxnet import train_recsys_rating
from nmf_mxnet import BPRLoss, train_ranking, evaluate_ranking

import d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import random
import sys

npx.set_np()

sys.path.append('F:/Codes/ailearn/longgb/DL_Learn/Recommender')


class Caser(nn.Block):
    def __init__(self, num_factors, num_users, num_items, L=5, d=16,
                 d_prime=4, drop_ratio=0.05, **kwargs):
        super(Caser, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.d_prime, self.d = d_prime, d
        # Vertical convolution layer
        self.conv_v = nn.Conv2D(d_prime, (L, 1), in_channels=1)
        # Horizontal convolution layer
        h = [i + 1 for i in range(L)]
        self.conv_h, self.max_pool = nn.Sequential(), nn.Sequential()
        for i in h:
            self.conv_h.add(nn.Conv2D(d, (i, num_factors), in_channels=1))
            self.max_pool.add(nn.MaxPool1D(L - i + 1))
        # Fully-connected layer
        self.fc1_dim_v, self.fc1_dim_h = d_prime * num_factors, d * len(h)
        self.fc = nn.Dense(in_units=d_prime * num_factors + d * L,
                           activation='relu', units=num_factors)
        # 这里为什么还要embedding
        self.Q_prime = nn.Embedding(num_items, num_factors * 2)
        self.b = nn.Embedding(num_items, 1)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, user_id, seq, item_id):
        item_embs = np.expand_dims(self.Q(seq), 1)
        user_emb = self.P(user_id)  # (4096, 10)
        out, out_h, out_v, out_hs = None, None, None, []
        # 横向卷积
        if self.d_prime:
            out_v = self.conv_v(item_embs)
            out_v = out_v.reshape(
                out_v.shape[0], self.fc1_dim_v)  # (4096, 4*10)
        # 纵向卷积 - 时间
        if self.d:
            for conv, maxp in zip(self.conv_h, self.max_pool):  # 滑动
                conv_out = np.squeeze(npx.relu(conv(item_embs)), axis=3)
                t = maxp(conv_out)
                pool_out = np.squeeze(t, axis=2)
                out_hs.append(pool_out)
            out_h = np.concatenate(out_hs, axis=1)  # (4096, 16*3)
        out = np.concatenate([out_v, out_h], axis=1)  # (4096, 4*10+16*3)
        z = self.fc(self.dropout(out))  # (4096, 10)

        # 和user_emb
        x = np.concatenate([z, user_emb], axis=1)  # (4096, 20)

        # 和item_emb计算
        q_prime_i = np.squeeze(self.Q_prime(item_id))  # (4096, 20)
        b = np.squeeze(self.b(item_id))
        res = (x * q_prime_i).sum(1) + b  # (4096,)
        return res


class SeqDataset(gluon.data.Dataset):
    def __init__(self, user_ids, item_ids, L, num_users, num_items,
                 candidates):
        user_ids, item_ids = np.array(user_ids), np.array(item_ids)
        sort_idx = np.array(sorted(range(len(user_ids)),
                                   key=lambda k: user_ids[k]))
        u_ids, i_ids = user_ids[sort_idx], item_ids[sort_idx]

        temp, u_ids, self.cand = {}, u_ids.asnumpy(), candidates
        self.all_items = set([i for i in range(num_items)])
        # 按照u_id-item编号,一个u的所有item完了，才下一个u的item编号
        [temp.setdefault(u_ids[i], []).append(i) for i, _ in enumerate(u_ids)]
        temp = sorted(temp.items(), key=lambda x: x[0])
        u_ids = np.array([i[0] for i in temp])
        idx = np.array([i[1][0] for i in temp])

        # ns是什么意思
        self.ns = ns = int(sum([c - L if c >= L + 1 else 1 for c
                                in np.array([len(i[1]) for i in temp])]))
        self.seq_items = np.zeros((ns, L))
        self.seq_users = np.zeros(ns, dtype='int32')
        self.seq_tgt = np.zeros((ns, 1))  # target序列
        self.test_seq = np.zeros((num_users, L))

        test_users, _uid = np.empty(num_users), None
        for i, (uid, i_seq) in enumerate(self._seq(u_ids, i_ids, idx, L + 1)):
            # 最后一个序列为test集
            if uid != _uid:
                # 这里也感觉不对
                self.test_seq[uid][:] = i_seq[-L:]
                test_users[uid], _uid = uid, uid
            # 这里总感觉不对
            self.seq_tgt[i][:] = i_seq[-1:]
            self.seq_items[i][:], self.seq_users[i] = i_seq[:L], uid

    def _win(self, tensor, window_size, step_size=1):
        if len(tensor) - window_size >= 0:
            # 从一个u的item尾部往头部滑动
            for i in range(len(tensor), 0, - step_size):
                # 滑出长度为window_size的序列
                if i - window_size >= 0:
                    yield tensor[i - window_size:i]
                else:
                    break
        else:
            yield tensor

    def _seq(self, u_ids, i_ids, idx, max_len):
        for i in range(len(idx)):
            stop_idx = None if i >= len(idx) - 1 else int(idx[i + 1])
            for s in self._win(i_ids[int(idx[i]):stop_idx], max_len):
                # u,子序列
                yield (int(u_ids[i]), s)

    def __len__(self):
        return self.ns

    def __getitem__(self, idx):
        neg = list(self.all_items - set(self.cand[int(self.seq_users[idx])]))
        i = random.randint(0, len(neg) - 1)
        return (self.seq_users[idx], self.seq_items[idx], self.seq_tgt[idx],
                neg[i])


if __name__ == "__main__":
    TARGET_NUM, L, batch_size = 1, 3, 4096

    df, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(df, num_users, num_items,
                                              'seq-aware')
    users_train, items_train, ratings_train, candidates = load_data_ml100k(
        train_data, num_users, num_items, feedback="implicit")
    users_test, items_test, ratings_test, test_iter = load_data_ml100k(
        test_data, num_users, num_items, feedback="implicit")

    train_seq_data = SeqDataset(users_train, items_train, L, num_users,
                                num_items, candidates)
    num_workers = 0 if sys.platform.startswith("win") else 4
    train_iter = gluon.data.DataLoader(train_seq_data, batch_size, True,
                                       last_batch="rollover",
                                       num_workers=num_workers)
    test_seq_iter = train_seq_data.test_seq

    ctx = d2l.try_all_gpus()
    net = Caser(10, num_users, num_items, L)
    net.initialize(ctx=ctx, force_reinit=True, init=mx.init.Normal(0.01))
    lr, num_epochs, wd, optimizer = 0.04, 8, 1e-5, 'adam'
    loss = BPRLoss()
    trainer = gluon.Trainer(net.collect_params(), optimizer,
                            {"learning_rate": lr, 'wd': wd})

    train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter,
                  num_users, num_items, num_epochs, ctx, evaluate_ranking,
                  candidates, eval_step=1)
