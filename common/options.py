# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:52:02 2019

@author: mypc
"""
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--data_dir', type=str, default='SDP', help='Root dir for saving models.')
    # parser.add_argument('--wordvec_dir', type=str, default='Embeds', help='Directory of word vectors')
    parser.add_argument('--vec_file', type=str, default='Embeds/sem16_tencent.pkl', help='词向量文件')

    parser.add_argument('--save_dir', type=str, default='saved_models', help='Root dir for saving models.')
    parser.add_argument('--save_name_suffix', type=str, help="File name to save the model", default='save_name_suffix')

    parser.add_argument('--train_merge_file', type=str, default='SDP/train/text_news.train.conllu',
                        help='教材+新闻的训练集（仅训练集）')
    parser.add_argument('--dev_text_file', type=str, default='SDP/dev/sdp_text_dev.conllu',
                        help='验证集（仅用text的dev）.')
    parser.add_argument('--dev_news_file', type=str, default='SDP/dev/sdp_news_dev.conllu',
                        help='验证集（仅用news的dev）.')
    parser.add_argument('--output_file_path', type=str, default='Eval',
                        help='模型输出文件')
    parser.add_argument('--test_text_file', type=str, default='SDP/test/sdp_text_test.conllu',
                        help='TEXT Test 标准答案.')
    parser.add_argument('--test_news_file', type=str, default='SDP/test/sdp_news_test.conllu',
                        help='NEWS Test 标准答案.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')
    parser.add_argument('--shorthand', type=str, help="Treebank shorthand")

    parser.add_argument('--hidden_dim', type=int, default=600, help='')
    parser.add_argument('--char_hidden_dim', type=int, default=400, help='')
    parser.add_argument('--deep_biaff_hidden_dim', type=int, default=600, help='')
    parser.add_argument('--composite_deep_biaff_hidden_dim', type=int, default=600, help='')
    parser.add_argument('--word_emb_dim', type=int, default=100, help='')
    parser.add_argument('--char_emb_dim', type=int, default=100, help='')
    parser.add_argument('--tag_emb_dim', type=int, default=100, help='')
    parser.add_argument('--transformed_dim', type=int, default=125, help='')

    parser.add_argument('--num_layers', type=int, default=3, help='')
    parser.add_argument('--char_num_layers', type=int, default=1, help='')

    parser.add_argument('--word_dropout', type=float, default=0.2, help='进入LSTM前的word drop比率')
    parser.add_argument('--dropout', type=float, default=0.33, help='层间dropout')
    parser.add_argument('--rec_dropout', type=float, default=0.25, help="DropoutRNN（沿时间步展开的dropout）")
    parser.add_argument('--char_rec_dropout', type=float, default=0.33, help="字符LSTM：DropoutRNN（沿时间步展开的dropout）")

    parser.add_argument('--no_char', dest='char', action='store_false', help="Turn off character model.")
    parser.add_argument('--no_pretrain', dest='pretrain', action='store_false', help="Turn off pretrained embeddings.")
    parser.add_argument('--no_linearization', dest='linearization', action='store_false',
                        help="Turn off linearization term.")
    parser.add_argument('--no_distance', dest='distance', action='store_false', help="Turn off distance term.")

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    # optim:
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0, help="adam beta1,normal=0.9")
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--L2_penalty', type=float, default=3e-9, help='normal=0')
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument('--rel_loss_ratio', type=float, default=0.5)
    # big batch:
    parser.add_argument('--big_batch', type=bool, default=False)
    parser.add_argument('--accumulation_steps', type=int, default=3, help='loss accumulation steps')
    # self attention,head_att,highway:
    parser.add_argument('--self_att', type=bool, default=False)
    parser.add_argument('--head_abs_att', type=bool, default=True)
    parser.add_argument('--real_highway', type=bool, default=False)

    parser.add_argument('--split_loss', default=False, type=bool)
    # bert adam args: -->
    # parser.add_argument("--bert_adam_lr",
    #                     default=5e-5,
    #                     type=float,
    #                     help="The initial learning rate for BERTAdam.")
    # parser.add_argument("--warmup_proportion",
    #                     default=0.1,
    #                     type=float,
    #                     help="Proportion of training to perform linear learning rate warmup for. "
    #                          "E.g., 0.1 = 10%% of training.")
    # <--

    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--log_step', type=int, default=50, help='Print log every k steps.')
    parser.add_argument('--max_steps_before_stop', type=int, default=3000)

    parser.add_argument('--batch_size', type=int, default=3000)
    parser.add_argument('--word_cutoff', type=int, default=0)
    parser.add_argument('--label_cutoff', type=int, default=0)

    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

    parser.add_argument('--logger_name', type=str, default='sdp_logger')
    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--nlpcc', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
