# -*- coding:utf-8 -*-
# @Author  : 'longguangbin'
# @Contact : lgb453476610@163.com
# @Date    : 2020/03/29
"""
Usage Of 'similar_word.py' : 
"""

from mxnet import nd
from mxnet.contrib import text


def knn(W, x, k):
    """ knn算法 """
    # 添加的1e-9是为了数值稳定性
    cos = nd.dot(W, x.reshape((-1,))) / (
        (nd.sum(W * W, axis=1) + 1e-9).sqrt() * nd.sum(x * x).sqrt())
    topk = nd.topk(cos, k=k, ret_typ='indices').asnumpy().astype('int32')
    return topk, [cos[i].asscalar() for i in topk]


def get_similar_tokens(query_token, k, embed):
    """ 获取近义词 """
    topk, cos = knn(embed.idx_to_vec,
                    embed.get_vecs_by_tokens([query_token]), k+1)
    for i, c in zip(topk[1:], cos[1:]):  # 除去输入词
        print('cosine sim=%.3f: %s' % (c, (embed.idx_to_token[i])))


def get_analogy(token_a, token_b, token_c, embed):
    """ 获取类比词 """
    vecs = embed.get_vecs_by_tokens([token_a, token_b, token_c])
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[topk[0]]


# 预训练的词向量
glove_6b50d = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.50d.txt')

# 求近义词
get_similar_tokens('chip', 3, glove_6b50d)

# 获取类比词
get_analogy('man', 'woman', 'son', glove_6b50d)
