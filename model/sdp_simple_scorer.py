"""
Utils and wrappers for scoring parsers.
"""
import sys

sys.path.append('..')

'''
def score(system_conllu_file, gold_conllu_file, verbose=True):
    """ Wrapper for UD parser scorer. """
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)
    el = evaluation['LAS']
    p = el.precision
    r = el.recall
    f = el.f1
    if verbose:
        scores = [evaluation[k].f1 * 100 for k in ['LAS', 'MLAS', 'BLEX']]
        print("LAS\tMLAS\tBLEX")
        print("{:.2f}\t{:.2f}\t{:.2f}".format(*scores))
    return p, r, f
'''


def parse_conllu(f_object):
    sents = []
    sent = []
    for line in f_object:
        line = line.strip()
        if len(line) == 0:
            sents.append(sent)
            sent = []
        else:
            line = line.split('\t')
            sent.append(line[8])
    return sents


def score(system_conllu_file, gold_conllu_file):
    arc_total, arc_correct, arc_predict, label_total, label_correct, label_predict = 0, 0, 0, 0, 0, 0
    with open(system_conllu_file, 'r', encoding='utf-8') as f_system, open(gold_conllu_file, 'r',
                                                                           encoding='utf-8') as f_gold:

        sys_sents = []
        sys_sent = []
        for line in f_system:
            line = line.strip()
            if len(line) == 0:
                sys_sents.append(sys_sent)
                sys_sent = []
            else:
                line = line.split('\t')
                sys_sent.append(line[8])

        gold_sents = []
        gold_sent = []
        for line in f_gold:
            line = line.strip()
            if len(line) == 0:
                gold_sents.append(gold_sent)
                gold_sent = []
            else:
                line = line.split('\t')
                gold_sent.append(line[8])

        for sys_sent, gold_sent in zip(sys_sents, gold_sents):
            for system, gold in zip(sys_sent, gold_sent):
                gold = gold.split('|')
                system = system.split('|')

                label_total += len(gold)
                label_predict += len(system)
                label_correct += len(list(set(gold) & set(system)))

                gold_head = [arc.split(':')[0] for arc in gold]
                sys_head = [arc.split(':')[0] for arc in system]

                arc_total += len(gold_head)
                arc_predict += len(sys_head)
                arc_correct += len(list(set(gold_head) & set(sys_head)))

    arc_recall = arc_correct / arc_total
    arc_precison = arc_correct / arc_predict
    arc_f = 2 * arc_precison * arc_recall / (arc_precison + arc_recall)

    label_recall = label_correct / label_total
    label_precison = label_correct / label_predict
    label_f = 2 * label_precison * label_precison / (label_precison + label_recall)
    UAS = arc_f
    LAS = label_f
    print('UAS Score:{}'.format(UAS))
    print('LAS Score:{}'.format(LAS))
    return UAS, LAS


if __name__ == '__main__':
    system_conllu_file = '../Eval/sdp_text_dev_predict.conllu'
    gold_conllu_file = '../SDP/sdp_text_dev.conllu'
    score(system_conllu_file, gold_conllu_file)
