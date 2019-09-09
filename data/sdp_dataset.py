# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     sdp_dataset
   Description :
   Author :       Liangs
   date：          2019/4/23
-------------------------------------------------
   Change Activity:
                   2019/4/23:
-------------------------------------------------
"""
import sys

sys.path.append('..')
import random
import torch
import conll
from doc import Document
from vocab import PAD_ID, ROOT_ID
from dep_vocab import CharVocab, WordVocab, GraphVocab, MultiVocab
from data_utils import get_long_tensor, sort_all, get_cuda_long_tensor

from pretrain import Pretrain

from common.options import parse_args
from common.utils import get_logger
import torch
from torch.utils.data import Dataset, DataLoader


class SDPDataset(Dataset):
    def __init__(self, input_src, batch_size, args, pretrain, vocab=None, evaluation=False):
        """
        eg:
            sdp_dataset = SDPDataset(args['train_file'], args['batch_size'], args, pretrain, evaluation=False)
        :param input_src:
        :param batch_size:
        :param args:
        :param pretrain:
        :param vocab:
        :param evaluation:
        """
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.logger = get_logger(args['logger_name'])

        # check if input source is a file or a Document object
        if isinstance(input_src, str):
            filename = input_src
            assert filename.endswith('conllu'), "Loaded file must be conllu file."
            # 加载所有句子：['word', 'upos', 'deps']
            # 4	总统	总统	NN	NN	_	6	Agt	6:Agt|12:Agt	_
            # [总统, NN, 6:Agt|12:Agt]
            self.conll, data = self.load_file(filename, evaluation=self.eval)
        elif isinstance(input_src, Document):
            filename = None
            doc = input_src
            self.conll, data = self.load_doc(doc)

        # handle vocab
        if vocab is None:
            self.vocab = self.init_vocab(data)
        else:
            self.vocab = vocab
        # token2id:
        self.pretrain_vocab = pretrain.vocab

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)
            self.logger.info("Subsample training set with rate {}".format(args['sample_train']))

        data = self.preprocess(data, self.vocab, self.pretrain_vocab, args)

        if self.shuffled:
            random.shuffle(data)
        self.num_examples = len(data)

        # chunk into batches
        # 先按照句长排序，然后再切分为batches
        self.data = self.chunk_batches(data)
        if filename is not None:
            self.logger.info("{} batches created for {}.".format(len(self.data), filename))

    def init_vocab(self, data):
        assert self.eval == False  # for eval vocab must exist
        charvocab = CharVocab(data, self.args['shorthand'])
        wordvocab = WordVocab(data, self.args['shorthand'], cutoff=self.args['min_occur_count'], lower=True)
        uposvocab = WordVocab(data, self.args['shorthand'], idx=1)
        graphvocab = GraphVocab(data, self.args['shorthand'], idx=2)
        vocab = MultiVocab({'char': charvocab,
                            'word': wordvocab,
                            'upos': uposvocab,
                            'graph': graphvocab})
        return vocab

    def preprocess(self, data, vocab, pretrain_vocab, args):
        # token2id
        processed = []
        for sent in data:
            processed_sent = [[ROOT_ID] + vocab['word'].map([w[0] for w in sent])]
            processed_sent += [[[ROOT_ID]] + [vocab['char'].map([x for x in w[0]]) for w in sent]]
            processed_sent += [[ROOT_ID] + vocab['upos'].map([w[1] for w in sent])]
            processed_sent += [[ROOT_ID] + pretrain_vocab.map([w[0] for w in sent])]
            processed_sent += [vocab['graph'].get_arc(sent, 2)]
            processed.append(processed_sent)
        return processed

    def chunk_batches(self, data):
        res = []

        if not self.eval:
            # sort sentences (roughly) by length for better memory utilization
            data = sorted(data, key=lambda x: len(x[0]), reverse=random.random() > .5)

        current = []
        currentlen = 0
        for x in data:
            if len(x[0]) + currentlen > self.batch_size:
                res.append(current)
                current = []
                currentlen = 0
            current.append(x)
            currentlen += len(x[0])

        if currentlen > 0:
            res.append(current)

        return res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        """
        原始batch的数据格式：
        sent=[
                word_id_list,   # 包括ROOT_ID；0
                char_id_list,   # 包括ROOT_ID；1
                upos_id_list,   # 包括ROOT_ID；2
                pretrain_vocab_list,    # 包括ROOT_ID；3
                graph_arc_list,
            ]
            graph_arc_list=[[head_id,red_id],...]
        """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)

        batch = list(zip(*batch))

        # assert len(batch) == 5  # word_idx_list, char_idx_list, pos_id_list, pretrain_list, arc_list

        # sort sentences by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]  # 统计所有句长
        batch, orig_idx = sort_all(batch, lens)  # 按照句长排序（降序），易于RNN处理

        # sort words by lens for easy char-RNN operations
        # 对char排序，易于char-RNN处理
        batch_words = [w for sent in batch[1] for w in sent]
        word_lens = [len(x) for x in batch_words]

        batch_words, word_orig_idx = sort_all([batch_words], word_lens)
        batch_words = batch_words[0]  # 去掉外层括号
        word_lens = [len(x) for x in batch_words]

        # convert to tensors and pad to max length
        words = batch[0]
        words = get_cuda_long_tensor(words, batch_size)
        words_mask = torch.eq(words, PAD_ID)

        wordchars = get_cuda_long_tensor(batch_words, len(word_lens))
        wordchars_mask = torch.eq(wordchars, PAD_ID)

        upos = get_cuda_long_tensor(batch[2], batch_size)
        pretrained = get_cuda_long_tensor(batch[3], batch_size)
        sentlens = [len(x) for x in batch[0]]
        arcs = batch[4]
        return words, words_mask, wordchars, wordchars_mask, upos, pretrained, arcs, orig_idx, word_orig_idx, sentlens, word_lens

    def load_file(self, filename, evaluation=False):
        conll_file = conll.CoNLLFile(filename)
        data = conll_file.get(['word', 'upos', 'deps'], as_sentences=True)
        return conll_file, data

    def load_doc(self, doc):
        data = doc.conll_file.get(['word', 'upos', 'deps'], as_sentences=True)
        return doc.conll_file, data

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def reshuffle(self):
        data = [y for x in self.data for y in x]
        self.data = self.chunk_batches(data)
        random.shuffle(self.data)


if __name__ == '__main__':
    conll_file_path = '../SDP/sdp_text_train.conllu'
    conll_file = conll.CoNLLFile(conll_file_path)
    data = conll_file.get(['word', 'upos', 'deps'], as_sentences=True)
    args = parse_args()
    args.train_file = conll_file_path
    args.batch_size = 1000
    # print(args)
    args = vars(args)
    vec_file = '../Embeds/sdp_vec.pkl'
    pretrain_file = '../save/sdp.pretrain.pt'
    pretrain = Pretrain(pretrain_file, vec_file)
    sdp_dataset = SDPDataset(args['train_file'], args['batch_size'], args, pretrain, evaluation=False)
    dataloader = DataLoader(sdp_dataset, batch_size=1, num_workers=4)


    # word = sdp_dataset[0][0]
    # words_mask = sdp_dataset[0][1]
    # wordchars = sdp_dataset[0][2]
    # wordchars_mask = sdp_dataset[0][3]
    # upos = sdp_dataset[0][4]
    # pretrained = sdp_dataset[0][5]
    # arcs = sdp_dataset[0][6]
    #
    # print(arcs[0])
    # orig_idx = sdp_dataset[0][7]
    # sentlens = sdp_dataset[0][9]
    # print(orig_idx)
    # print(arcs[0])
    # sent = sdp_dataset.vocab['graph'].parse_to_sent(arcs[0])
    # print(sent)

    def check(data, mode, idx, flag='pretrain'):
        check = []
        for d in data[idx]:
            if flag == 'pretrain':
                check.append(mode.vocab.id2unit(d.item()))
            else:
                check.append(mode.id2unit(d.item()))
        print(check)


    # print('|word size: {}'.format(word.size()))
    # print('|words mask size: {}'.format(words_mask.size()))
    # print('|wordchars size: {}'.format(wordchars.size()))
    #
    # print('|pretrained: {}'.format(pretrained.dtype))
    # check(word, sdp_dataset.vocab['word'], 0, flag='word')
    # check(word, sdp_dataset.vocab['word'], -1, flag='word')
    # check(pretrained, pretrain)
    # from pprint import pprint
    #
    # pprint(sdp_dataset[0])
    # print("===" * 10)
    # for i, data in enumerate(dataloader):
    #     if i != 0:
    #         break
    #     pprint(data)
    from common.timer import Timer

    with Timer('dataset:'):
        for _, data in enumerate(sdp_dataset):
            pass
    # with Timer('dataloader:'):
    #     for _, data in enumerate(dataloader):
    #         pass
