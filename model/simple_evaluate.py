# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     simple_evaluate.py
   Description :
   Author :       Liangs
   date：          2019/5/6
-------------------------------------------------
   Change Activity:
                   2019/5/6:
-------------------------------------------------
"""
INF = float('inf')


def stat_one_tree(lines):
    stat_data = {}
    for line in lines:
        payload = line.strip().split("\t")
        if (len(payload) < 7):
            print(lines)
        id_val = int(payload[0])
        form_val = payload[1]
        postag_val = payload[3]
        head_val = payload[6]
        deprel_val = payload[7]
        # if not opts.punctuation and engine(form_val, postag_val):
        #     continue
        if id_val not in stat_data:
            stat_data[id_val] = {
                "id": id_val,
                "form": form_val,
                "heads": [head_val],
                "deprels": [deprel_val]
            }
        else:
            assert (form_val == stat_data[id_val]["form"])
            stat_data[id_val]["heads"].append(head_val)
            stat_data[id_val]['deprels'].append(deprel_val)
    return stat_data


def stat_one_node_heads_and_deprels(gold_heads, gold_deprels, test_heads, test_deprels):
    gold_len = len(gold_heads)  # ! assert( len(gold_heads) == len(gold_deprels))
    test_len = len(test_heads)
    nr_right_heads = 0
    nr_right_deprels = 0

    assert gold_len != 0 and test_len != 0
    if gold_len == 1 and test_len == 1:
        # ! normal situation
        if gold_heads[0] == test_heads[0]:
            nr_right_heads = 1
            if gold_deprels[0] == test_deprels[0]:
                nr_right_deprels = 1
    else:
        for gold_head, gold_deprel in zip(gold_heads, gold_deprels):
            if gold_head in test_heads:
                nr_right_heads += 1
                head_idx = test_heads.index(gold_head)
                if gold_deprel == test_deprels[head_idx]:  # !! head_idx == deprel_idx
                    nr_right_deprels += 1
    return (gold_len, test_len,
            nr_right_heads, nr_right_deprels)


def stat_gold_and_test_data(gold_stat_data, test_stat_data):
    nr_gold_rels = 0
    nr_test_rels = 0
    nr_head_right = 0
    nr_deprel_right = 0

    for idx in gold_stat_data.keys():
        gold_node = gold_stat_data[idx]
        test_node = test_stat_data[idx]
        assert (gold_node['id'] == test_node['id'])

        (
            gold_rels_len, test_rels_len,
            nr_one_node_right_head, nr_one_node_right_deprel
        ) = (
            stat_one_node_heads_and_deprels(gold_node['heads'], gold_node['deprels'],
                                            test_node['heads'], test_node['deprels'])
        )

        nr_gold_rels += gold_rels_len
        nr_test_rels += test_rels_len
        nr_head_right += nr_one_node_right_head
        nr_deprel_right += nr_one_node_right_deprel

    return (nr_gold_rels, nr_test_rels,
            nr_head_right, nr_deprel_right)


def score(system_conllu_file, gold_conllu_file):
    reference_dataset = open(gold_conllu_file, "r", encoding='utf-8').read().strip().split("\n\n")
    answer_dataset = open(system_conllu_file, "r", encoding='utf-8').read().strip().split("\n\n")

    assert len(reference_dataset) == len(answer_dataset), "Number of instance unequal."

    nr_total_gold_rels = 0
    nr_total_test_rels = 0
    nr_total_right_heads = 0
    nr_total_right_deprels = 0

    nr_sentence = len(reference_dataset)

    length_error_num = 0

    for reference_data, answer_data in zip(reference_dataset, answer_dataset):
        reference_lines = reference_data.split("\n")
        answer_lines = answer_data.split("\n")

        reference_stat_data = stat_one_tree(reference_lines)
        answer_stat_data = stat_one_tree(answer_lines)
        if len(reference_stat_data) != len(answer_stat_data):
            length_error_num += 1
            continue

        (
            nr_one_gold_rels, nr_one_test_rels,
            nr_one_head_right, nr_one_deprel_right
        ) \
            = stat_gold_and_test_data(reference_stat_data, answer_stat_data)

        nr_total_gold_rels += nr_one_gold_rels
        nr_total_test_rels += nr_one_test_rels
        nr_total_right_heads += nr_one_head_right
        nr_total_right_deprels += nr_one_deprel_right

    nr_sentence -= length_error_num

    LAS = float(2 * nr_total_right_deprels) / (nr_total_test_rels + nr_total_gold_rels) \
        if (nr_total_gold_rels + nr_total_test_rels) != 0 else INF

    UAS = float(2 * nr_total_right_heads) / (nr_total_test_rels + nr_total_gold_rels) \
        if (nr_total_gold_rels + nr_total_test_rels) != 0 else INF

    return UAS, LAS


if __name__ == '__main__':
    UAS, LAS = score(system_conllu_file='../process_SAE_output/yizhixing_merge/sorted/wyz.sdp_sorted.sdp.conllu',
                     gold_conllu_file='../process_SAE_output/yizhixing_merge/sorted/wlh.sdp_sorted.sdp.conllu')
    print(f'LAS:{LAS:0.5f}\nUAS:{UAS:0.5f}')
