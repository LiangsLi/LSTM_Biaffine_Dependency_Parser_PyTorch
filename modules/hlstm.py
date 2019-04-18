import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence
from packed_lstm import PackedLSTM


# Highway LSTM Cell (Zhang et al. (2018) Highway Long Short-Term Memory RNNs for Distant Speech Recognition)
class HLSTMCell(nn.modules.rnn.RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(HLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM parameters
        self.Wi = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.Wf = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.Wo = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.Wg = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)

        # highway gate parameters
        self.gate = nn.Linear(input_size + 2 * hidden_size, hidden_size, bias=bias)

    def forward(self, input, c_l_minus_one=None, hx=None):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = (hx, hx)
        if c_l_minus_one is None:
            c_l_minus_one = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)

        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        self.check_forward_hidden(input, c_l_minus_one, 'c_l_minus_one')

        # vanilla LSTM computation
        rec_input = torch.cat([input, hx[0]], 1)
        i = F.sigmoid(self.Wi(rec_input))
        f = F.sigmoid(self.Wi(rec_input))
        o = F.sigmoid(self.Wi(rec_input))
        g = F.tanh(self.Wi(rec_input))

        # highway gates
        gate = F.sigmoid(self.gate(torch.cat([c_l_minus_one, hx[1], input], 1)))

        c = gate * c_l_minus_one + f * hx[1] + i * g
        h = o * F.tanh(c)

        return h, c


# Highway LSTM network, does NOT use the HLSTMCell above
# 用法：
#         self.parserlstm = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True,
#                                       bidirectional=True, dropout=self.args['dropout'],
#                                       rec_dropout=self.args['rec_dropout'], highway_func=torch.tanh)
class HighwayLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, rec_dropout=0, highway_func=None, pad=False):
        super(HighwayLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.highway_func = highway_func
        self.pad = pad

        self.lstm = nn.ModuleList()
        self.highway = nn.ModuleList()
        self.gate = nn.ModuleList()
        self.drop = nn.Dropout(dropout, inplace=True)

        in_size = input_size
        for l in range(num_layers):
            # 过一层PackedLSTM，不对输入做dropout（dropout=0），对每个展开的中间状态做dropout（rec_dropout）
            # in_size --> hidden_size*num_directions
            self.lstm.append(PackedLSTM(in_size, hidden_size, num_layers=1, bias=bias,
                                        batch_first=batch_first, dropout=0, bidirectional=bidirectional,
                                        rec_dropout=rec_dropout))
            # in_size --> hidden_size*num_directions
            self.highway.append(nn.Linear(in_size, hidden_size * self.num_directions))
            # in_size --> hidden_size*num_directions
            self.gate.append(nn.Linear(in_size, hidden_size * self.num_directions))
            self.highway[-1].bias.data.zero_()
            self.gate[-1].bias.data.zero_()
            # 除了第一层的in_size==input_size之外，其他层的in_size==hidden_size*num_directions
            in_size = hidden_size * self.num_directions

    def forward(self, input, seqlens, hx=None):
        highway_func = (lambda x: x) if self.highway_func is None else self.highway_func

        hs = []
        cs = []

        if not isinstance(input, PackedSequence):
            input = pack_padded_sequence(input, seqlens, batch_first=self.batch_first)

        for l in range(self.num_layers):
            if l > 0:
                # 对每层的输入做dropout
                # 代替了PackedLSTM中的层间dropout
                input = PackedSequence(self.drop(input.data), input.batch_sizes)
            layer_hx = (hx[0][l * self.num_directions:(l + 1) * self.num_directions],
                        hx[1][l * self.num_directions:(l + 1) * self.num_directions]) if hx is not None else None
            # 执行每层的PackedLSTM
            h, (ht, ct) = self.lstm[l](input, seqlens, layer_hx)

            hs.append(ht)
            cs.append(ct)
            # highway连接：（残差连接）
            input = PackedSequence(
                h.data + torch.sigmoid(self.gate[l](input.data)) * highway_func(self.highway[l](input.data)),
                input.batch_sizes)

        if self.pad:
            input = pad_packed_sequence(input, batch_first=self.batch_first)[0]
        return input, (torch.cat(hs, 0), torch.cat(cs, 0))


if __name__ == "__main__":
    T = 10
    bidir = True
    num_dir = 2 if bidir else 1
    rnn = HighwayLSTM(10, 20, num_layers=2, bidirectional=True)
    input = torch.randn(T, 3, 10)
    hx = torch.randn(2 * num_dir, 3, 20)
    cx = torch.randn(2 * num_dir, 3, 20)
    lengths = [3 for _ in range(3)]
    print(lengths)
    output = rnn(input, lengths, (hx, cx))
    print(output)
