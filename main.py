import sys

sys.path.append('./data/')
sys.path.append('./common/')
sys.path.append('./modules/')
sys.path.append('./model/')
print(sys.path)
import os
import shutil
import time
from datetime import datetime
import numpy as np
import random
import torch
from torch import nn, optim

from common.options import parse_args
from data.sdp_data_loader import DataLoader
from model.sdp_biaffine_trainer import Trainer
from model import sdp_simple_scorer
from common import utils
from data.pretrain import Pretrain


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)

    args = vars(args)
    print("Running parser in {} mode".format(args['mode']))

    if args['mode'] == 'train':
        train(args)
    else:
        evaluate(args)


def train(args):
    utils.ensure_dir(args['save_dir'])
    model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
        else '{}/{}_parser.pt'.format(args['save_dir'], args['shorthand'])

    # load pretrained vectors
    vec_file = './Embeds/sdp_vec.pkl'
    pretrain_file = './save/sdp.pretrain.pt'
    # 传入torch预训练的模型文件（pt）和Python二进制持久化文件（pkl）
    pretrain = Pretrain(pretrain_file, vec_file)

    # load data
    print("Loading data with batch size {}...".format(args['batch_size']))
    train_batch = DataLoader(args['train_file'], args['batch_size'], args, pretrain, evaluation=False)
    vocab = train_batch.vocab
    dev_batch = DataLoader(args['eval_file'], args['batch_size'], args, pretrain, vocab=vocab, evaluation=True)

    # pred and gold path
    system_pred_file = args['output_file']
    gold_file = args['gold_file']

    # skip training if the language does not have training or dev data
    if len(train_batch) == 0 or len(dev_batch) == 0:
        print("Skip training because no data available...")
        exit()

    print("Training parser...")
    trainer = Trainer(args=args, vocab=vocab, pretrain=pretrain, use_cuda=args['cuda'])

    global_step = 0
    max_steps = args['max_steps']
    dev_score_history = []
    best_dev_preds = []
    current_lr = args['lr']
    global_start_time = time.time()
    format_str = '{}: step {}/{}, loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'

    using_amsgrad = False
    last_best_step = 0
    # start training
    train_loss = 0
    while True:
        do_break = False
        for i, batch in enumerate(train_batch):
            # batch=(words, words_mask, wordchars, wordchars_mask,
            #        upos, pretrained, arcs, orig_idx, word_orig_idx,
            #        sentlens, word_lens)
            start_time = time.time()
            global_step += 1
            loss = trainer.update(batch, eval=False)  # update step
            train_loss += loss
            if global_step % args['log_step'] == 0:
                duration = time.time() - start_time
                print(format_str.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), global_step, \
                                        max_steps, loss, duration, current_lr))

            if global_step % args['eval_interval'] == 0:
                # eval on dev
                print("Evaluating on dev set...")
                dev_preds = []
                for batch in dev_batch:
                    preds = trainer.predict(batch)
                    dev_preds += preds

                dev_batch.conll.set(['deps'], [y for x in dev_preds for y in x])
                dev_batch.conll.write_conll(system_pred_file)
                _, dev_score = sdp_simple_scorer.score(system_pred_file, gold_file)

                train_loss = train_loss / args['eval_interval']  # avg loss per batch
                print("step {}: train_loss = {:.6f}, dev_score = {:.4f}".format(global_step, train_loss, dev_score))
                train_loss = 0

                # save best model
                if len(dev_score_history) == 0 or dev_score > max(dev_score_history):
                    last_best_step = global_step
                    trainer.save(model_file)
                    print("new best model saved.")
                    best_dev_preds = dev_preds

                dev_score_history += [dev_score]
                print("")

            if global_step - last_best_step >= args['max_steps_before_stop']:
                if not using_amsgrad:
                    print("Switching to AMSGrad")
                    last_best_step = global_step
                    using_amsgrad = True
                    trainer.optimizer = optim.Adam(trainer.model.parameters(), amsgrad=True, lr=args['lr'],
                                                   betas=(.9, args['beta2']), eps=1e-6)
                else:
                    do_break = True
                    break

            if global_step >= args['max_steps']:
                do_break = True
                break

        if do_break: break

        train_batch.reshuffle()

    print("Training ended with {} steps.".format(global_step))

    best_f, best_eval = max(dev_score_history) * 100, np.argmax(dev_score_history) + 1
    print("Best dev F1 = {:.2f}, at iteration = {}".format(best_f, best_eval * args['eval_interval']))


def evaluate(args):
    args = vars(args)
    # file paths
    system_pred_file = args['output_file']
    gold_file = args['gold_file']
    model_file = args['save_dir'] + '/' + args['save_name'] if args['save_name'] is not None \
        else '{}/{}_parser.pt'.format(args['save_dir'], args['shorthand'])
    pretrain_file = './save/sdp.pretrain.pt'

    # load pretrain
    pretrain = Pretrain(pretrain_file)

    # load model
    use_cuda = args['cuda'] and not args['cpu']
    trainer = Trainer(pretrain=pretrain, model_file=model_file, use_cuda=use_cuda)
    loaded_args, vocab = trainer.args, trainer.vocab

    # load config
    for k in args:
        if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand'] or k == 'mode':
            loaded_args[k] = args[k]

    # load data
    print("Loading data with batch size {}...".format(args['batch_size']))
    batch = DataLoader(args['eval_file'], args['batch_size'], loaded_args, pretrain, vocab=vocab, evaluation=True)

    if len(batch) > 0:
        print("Start evaluation...")
        preds = []
        for i, b in enumerate(batch):
            preds += trainer.predict(b)
    else:
        # skip eval if dev data does not exist
        preds = []

    # write to file and score
    batch.conll.set(['deps'], [y for x in preds for y in x])
    batch.conll.write_conll(system_pred_file)

    if gold_file is not None:
        _, score = sdp_simple_scorer.score(system_pred_file, gold_file)

        print("Parser score:")
        print("{} {:.2f}".format(args['shorthand'], score * 100))


if __name__ == '__main__':
    args = parse_args()
    args.save_name = 'sdp_parser.pt'
    args.train_file = './SDP/sdp_text_train.conllu'
    args.eval_file = './SDP/sdp_text_test.conllu'
    args.gold_file = './SDP/sdp_text_test.conllu'
    args.output_file = './Eval/sdp_text_test_predict.conllu'
    args.batch_size = 3000
    # args.cpu = True
    print(args)
    # main(args)
    evaluate(args)
