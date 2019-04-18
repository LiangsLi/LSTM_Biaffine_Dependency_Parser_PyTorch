import sys

sys.path.append('..')
sys.path.append('../modules')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

from sdp_utils import make_unlabeltarget, make_labeltarget
# import modules
from modules.biaffine import DeepBiaffineScorer
from modules.hlstm import HighwayLSTM
from modules.char_model import CharacterModel

from common.dropout import WordDropout


class Parser(nn.Module):
    def __init__(self, args, vocab, emb_matrix=None, share_hid=False):
        super().__init__()

        self.vocab = vocab
        self.args = args
        self.share_hid = share_hid
        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)

        # input layers
        input_size = 0
        if self.args['word_emb_dim'] > 0:
            self.word_emb = nn.Embedding(len(vocab['word']), self.args['word_emb_dim'], padding_idx=0)
            input_size += self.args['word_emb_dim']

        if self.args['tag_emb_dim'] > 0:
            self.upos_emb = nn.Embedding(len(vocab['upos']), self.args['tag_emb_dim'], padding_idx=0)
            input_size += self.args['tag_emb_dim']

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            self.charmodel = CharacterModel(args, vocab)
            self.trans_char = nn.Linear(self.args['char_hidden_dim'], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']

        if self.args['pretrain']:
            add_unsaved_module('pretrained_emb',
                               nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
            self.trans_pretrained = nn.Linear(emb_matrix.shape[1], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']

        # recurrent layers
        self.parserlstm = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True,
                                      bidirectional=True, dropout=self.args['dropout'],
                                      rec_dropout=self.args['rec_dropout'], highway_func=torch.tanh)
        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        self.parserlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        self.parserlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))

        # classifiers
        # 先预测边：
        self.unlabeled = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'],
                                            self.args['deep_biaff_hidden_dim'], 1, pairwise=True,
                                            dropout=args['dropout'])
        # 再预测标签：
        self.deprel = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'],
                                         self.args['deep_biaff_hidden_dim'], len(vocab['graph']), pairwise=True,
                                         dropout=args['dropout'])

        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, pretrained, arcs, word_orig_idx, sentlens,
                wordlens):
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)

        inputs = []
        if self.args['pretrain']:
            pretrained_emb = self.pretrained_emb(pretrained)
            pretrained_emb = self.trans_pretrained(pretrained_emb)
            pretrained_emb = pack(pretrained_emb)
            inputs += [pretrained_emb]

        # def pad(x):
        #    return pad_packed_sequence(PackedSequence(x, pretrained_emb.batch_sizes), batch_first=True)[0]

        if self.args['word_emb_dim'] > 0:
            word_emb = self.word_emb(word)
            word_emb = pack(word_emb)
            inputs += [word_emb]

        if self.args['tag_emb_dim'] > 0:
            pos_emb = self.upos_emb(upos)
            pos_emb = pack(pos_emb)
            inputs += [pos_emb]

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
            char_reps = PackedSequence(self.trans_char(self.drop(char_reps.data)), char_reps.batch_sizes)
            inputs += [char_reps]

        lstm_inputs = torch.cat([x.data for x in inputs], 1)    # x 是PackedSequence .data 取出tensor

        lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)

        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)

        lstm_outputs, _ = self.parserlstm(lstm_inputs, sentlens, hx=(
            self.parserlstm_h_init.expand(2 * self.args['num_layers'], word.size(0),
                                          self.args['hidden_dim']).contiguous(),
            self.parserlstm_c_init.expand(2 * self.args['num_layers'], word.size(0),
                                          self.args['hidden_dim']).contiguous()))
        lstm_outputs, _ = pad_packed_sequence(lstm_outputs, batch_first=True)

        unlabeled_scores = self.unlabeled(self.drop(lstm_outputs), self.drop(lstm_outputs)).squeeze(3)
        deprel_scores = self.deprel(self.drop(lstm_outputs), self.drop(lstm_outputs))

        preds = []

        if self.training:
            '''
            head_target -- type:tensor, shape:(n, m, m)
            weights -- type:tensor, shape:(n, m, m)
            word_mask -- type:tensor, shape:(n, m)
            unlabelscore -- type:tensor, shape:(n, m, m)
            '''
            batch_size, seq_len = word.size()
            head_target = make_unlabeltarget(arcs, sentlens, self.args['cuda'])
            weights = torch.ones(batch_size, seq_len, seq_len, dtype=torch.float, device=word.device)
            weights = weights.masked_fill(word_mask.unsqueeze(1), 0)
            weights = weights.masked_fill(word_mask.unsqueeze(2), 0)
            crit_head = nn.BCEWithLogitsLoss(weight=weights, reduction='sum')
            head_loss = crit_head(unlabeled_scores, head_target)

            '''
            deprel_target -- type:tensor, shape:(n, m, m)
            deprel_scores -- type:tensor, shape:(n, m, m, c)
            deprel_mask -- tyep:tensor, shape:(n, m)
            '''
            self.crit_rel = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
            deprel_target = make_labeltarget(arcs, sentlens, self.args['cuda'])
            deprel_mask = deprel_target.eq(0)
            deprel_target = deprel_target.masked_fill(deprel_mask, -1)
            deprel_scores = deprel_scores.contiguous().view(-1, len(self.vocab['graph']))
            rel_loss = self.crit_rel(deprel_scores, deprel_target.view(-1))
            loss = (1 - 0.025) * head_loss + 0.025 * rel_loss

            loss /= wordchars.size(0)  # number of words
        else:
            loss = 0
            # head_target = make_unlabeltarget(arcs, sentlens, self.args['cuda'])
            batch_size, seq_len = word.size()
            weights = torch.ones(batch_size, seq_len, seq_len, dtype=torch.float, device=word.device)
            weights = weights.masked_fill(word_mask.unsqueeze(1), 0)
            weights = weights.masked_fill(word_mask.unsqueeze(2), 0)
            weights = weights.unsqueeze(3)
            print(unlabeled_scores[0])
            head_probs = torch.sigmoid(unlabeled_scores).unsqueeze(3)
            label_probs = torch.softmax(deprel_scores, dim=3)
            semgraph_probs = head_probs * label_probs * weights
            preds.append(semgraph_probs.detach().cpu().numpy())

        return loss, preds


if __name__ == '__main__':
    pass
