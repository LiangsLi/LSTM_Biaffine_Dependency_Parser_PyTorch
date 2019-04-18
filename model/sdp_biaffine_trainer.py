# -*- coding: utf-8 -*-\
"""
A trainer class to handle training and testing of models.
"""
import sys

sys.path.append('..')
sys.path.append('../data')
import torch

from common.trainer import Trainer as BaseTrainer
from common import utils

from sdp_lstm_biaffine import Parser
from data.dep_vocab import MultiVocab
from sdp_utils import sdp_decoder, parse_semgraph
from modules.optimization import BertAdam


def unpack_batch(batch, use_cuda):
    """ Unpack a batch from the data loader.
    batch=(
            words,  0
            words_mask,     1
            wordchars,      2
            wordchars_mask,     3
            upos,       4
            pretrained,     5
            arcs,       6
            orig_idx,       7
            word_orig_idx,      8
            sentlens,       9
            word_lens       10
          )
    """
    if use_cuda:
        device = torch.device("cuda:1")
        inputs = [b.cuda(device) if b is not None else None for b in batch[:6]]
    else:
        inputs = batch[:6]
    # inputs = [words, words_mark, wordchars, wordchars_mark, upos, pretrained]
    arcs = batch[6]
    orig_idx = batch[7]
    word_orig_idx = batch[8]
    sentlens = batch[9]
    wordlens = batch[10]
    return inputs, arcs, orig_idx, word_orig_idx, sentlens, wordlens


class Trainer(BaseTrainer):
    """ A trainer for training models. """

    # eg: trainer = Trainer(args=args, vocab=vocab, pretrain=pretrain, use_cuda=args['cuda'])
    def __init__(self, args=None, vocab=None, pretrain=None, model_file=None, use_cuda=False):
        self.use_cuda = use_cuda
        if model_file is not None:
            # load everything from file
            self.load(pretrain, model_file)
        else:
            assert all(var is not None for var in [args, vocab, pretrain])
            # build model from scratch
            self.args = args
            self.vocab = vocab
            self.model = Parser(args, vocab, emb_matrix=pretrain.emb)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.use_cuda:
            device = torch.device("cuda:1")
            self.model.cuda(device)
        else:
            self.model.cpu()
        if self.args['optim'] == 'BertAdam':

            # 对 bias、gamma、beta变量不使用权重衰减
            # 权重衰减是一种正则化手段
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.parameters if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in self.parameters if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
            self.optimizer = BertAdam(optimizer_grouped_parameters,
                                      lr=self.args['learning_rate'],
                                      warmup=self.args['warmup_proportion'],
                                      t_total=self.args['max_steps'])
        else:
            self.optimizer = utils.get_optimizer(self.args['optim'], self.parameters, self.args['lr'],
                                                 betas=(0.9, self.args['beta2']), eps=1e-6)

    def update(self, batch, eval=False):
        # inputs = [words, words_mark, wordchars, wordchars_mark, upos, pretrained]
        inputs, arcs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, upos, pretrained = inputs

        if eval:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()
        loss, _ = self.model(word, word_mask, wordchars, wordchars_mask, upos, pretrained, arcs, word_orig_idx,
                             sentlens, wordlens)
        loss_val = loss.data.item()
        if eval:
            return loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, arcs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        print(sentlens)
        word, word_mask, wordchars, wordchars_mask, upos, pretrained = inputs
        self.model.eval()

        _, preds = self.model(word, word_mask, wordchars, wordchars_mask, upos, pretrained, arcs, word_orig_idx,
                              sentlens, wordlens)

        semgraph = sdp_decoder(preds[0], sentlens)
        sents = parse_semgraph(semgraph, sentlens)
        pred_sents = self.vocab['graph'].parse_to_sent_batch(sents)
        if unsort:
            pred_sents = utils.unsort(pred_sents, orig_idx)
        return pred_sents

    def save(self, filename, skip_modules=True):
        model_state = self.model.state_dict()
        # skip saving modules like pretrained embeddings, because they are large and will be saved in a separate file
        if skip_modules:
            skipped = [k for k in model_state.keys() if k.split('.')[0] in self.model.unsaved_modules]
            for k in skipped:
                del model_state[k]
        params = {
            'model': model_state,
            'vocab': self.vocab.state_dict(),
            'config': self.args
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, pretrain, filename):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.args = checkpoint['config']
        self.vocab = MultiVocab.load_state_dict(checkpoint['vocab'])
        self.model = Parser(self.args, self.vocab, emb_matrix=pretrain.emb)
        self.model.load_state_dict(checkpoint['model'], strict=False)


if __name__ == '__main__':
    pass
