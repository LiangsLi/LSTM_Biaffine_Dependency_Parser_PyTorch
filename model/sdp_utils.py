# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 16:53:25 2019

@author: mypc
"""
import torch
import numpy as np


def make_unlabeltarget(arcs, sentlens, use_cuda=False):
    max_len = sentlens[0]
    # print(max_len)
    batch_size = len(arcs)
    # print(batch_size)
    graphs = torch.zeros(batch_size, max_len, max_len)
    sent_idx = 0
    for sent in arcs:
        word_idx = 1
        for word in sent:
            for arc in word:
                # print(sent_idx, word_idx, arc)
                head_idx = arc[0]
                graphs[sent_idx, word_idx, head_idx] = 1
            word_idx += 1
        sent_idx += 1
    if use_cuda:
        graphs = graphs.float().cuda()
    else:
        graphs = graphs.float()
    return graphs


def make_labeltarget(arcs, sentlens, use_cuda=False):
    max_len = sentlens[0]
    # print(max_len)
    batch_size = len(arcs)
    # print(batch_size)
    graphs = torch.zeros(batch_size, max_len, max_len)
    sent_idx = 0
    for sent in arcs:
        word_idx = 1
        for word in sent:
            for arc in word:
                # print(sent_idx, word_idx, arc)
                head_idx = arc[0]
                rel_idx = arc[1]
                graphs[sent_idx, word_idx, head_idx] = rel_idx
            word_idx += 1
        sent_idx += 1
    if use_cuda:
        graphs = graphs.long().cuda()
    else:
        graphs = graphs.long()
    return graphs


def make_discriminator_target(sent_num, task_id, use_cuda=False):
    labels = torch.zeros(sent_num)
    labels.fill_(task_id)
    if use_cuda:
        labels = labels.long().cuda()
    else:
        labels = labels.long()
    return labels


def sdp_decoder(semgraph_probs, sentlens):
    '''
    semhead_probs type:ndarray, shape:(n,m,m)
    '''
    semhead_probs = semgraph_probs.sum(axis=-1)
    semhead_preds = np.where(semhead_probs >= 0.5, 1, 0)
    masked_semhead_preds = np.zeros(semhead_preds.shape, dtype=np.int32)
    for i, (sem_preds, length) in enumerate(zip(semhead_preds, sentlens)):
        masked_semhead_preds[i, :length, :length] = sem_preds[:length, :length]
    n_counts = {'no_root': 0, 'multi_root': 0, 'no_head': 0, 'self_circle': 0}
    for i, length in enumerate(sentlens):
        for j in range(length):
            if masked_semhead_preds[i, j, j] == 1:
                n_counts['self_circle'] += 1
                masked_semhead_preds[i, j, j] = 0
        n_root = np.sum(masked_semhead_preds[i, :, 0])
        if n_root == 0:
            n_counts['no_root'] += 1
            new_root = np.argmax(semhead_probs[i, 1:, 0]) + 1
            masked_semhead_preds[i, new_root, 0] = 1
        elif n_root > 1:
            n_counts['multi_root'] += 1
            kept_root = np.argmax(semhead_probs[i, 1:, 0]) + 1
            masked_semhead_preds[i, :, 0] = 0
            masked_semhead_preds[i, kept_root, 0] = 1
        n_heads = masked_semhead_preds[i, :length, :length].sum(axis=-1)
        n_heads[0] = 1
        for j, n_head in enumerate(n_heads):
            if n_head == 0:
                n_counts['no_head'] += 1
                semhead_probs[i, j, j] = 0
                new_head = np.argmax(semhead_probs[i, j, 1:length]) + 1
                masked_semhead_preds[i, j, new_head] = 1
    # print('Corrected List:','\t'.join([key+':' + str(val) for key, val in n_counts.items()]))
    # (n x m x m x c) -> (n x m x m)
    semrel_preds = np.argmax(semgraph_probs, axis=-1)
    # (n x m x m) (*) (n x m x m) -> (n x m x m)
    semgraph_preds = masked_semhead_preds * semrel_preds
    result = masked_semhead_preds + semgraph_preds
    return result


def parse_semgraph(semgraph, sentlens):
    semgraph = semgraph.tolist()
    sents = []
    for s, l in zip(semgraph, sentlens):
        words = []
        for w in s[1:l]:
            arc = []
            for head_idx, deprel in enumerate(w[:l]):
                if deprel == 0: continue
                arc.append([head_idx, deprel - 1])
            words.append(arc)
        sents.append(words)
    return sents


if __name__ == '__main__':
    test = [[[(0, 3), (2, 5)], [(2, 7)]],
            [[(1, 21)]]]
    sentlens = [3, 2]
    unlabeltarget = make_unlabeltarget(test, sentlens)
    print(unlabeltarget)
    labeltarget = make_labeltarget(test, sentlens)
    print(labeltarget)

    '''test sdp_decoder'''
    '''biaffine modle can get (n, m, m) score matrix'''
    a = torch.randn(2, 3, 3)
    b = torch.randn(2, 3, 3, 4)
    a = torch.sigmoid(a)
    sentlens = [3, 2]
    res = sdp_decoder(a.numpy(), b.numpy(), sentlens)
    print('''result''')
    print(res)
    graph = parse_semgraph(res, sentlens)
    print('''graph''')
    print(graph)
