from utils.data import Node
from collections import Counter


def eval_assembly_tree(gt_tree: Node, pred_tree: Node, round_n_digit=-1):
    '''
    return a dict of:
     1. no_children
     2. children_set
     3. children_order
    '''
    gt_nodes = []
    pred_nodes = []

    def traverse_tree(tree, func):
        func(tree)
        for c in tree.children:
            traverse_tree(c, func)
        return

    for tree, li in zip([gt_tree, pred_tree], [gt_nodes, pred_nodes]):
        traverse_tree(tree, lambda n: li.append(n) if len(n.children) > 0 else None)

    ct = {'no_children': 0, 'children_set': 0, 'children_order': 0}

    for n_gt in gt_nodes:
        for n_pred in pred_nodes:
            if n_gt.part_idxs == n_pred.part_idxs:
                ct['no_children'] += 1
                # remove leaf nodes as their order should be ignored in both metrics
                child_list_gt = [c.part_idxs for c in n_gt.children
                                 if len(c.part_idxs) > 1]
                child_list_pred = [c.part_idxs for c in n_pred.children
                                   if len(c.part_idxs) > 1]
                if set(child_list_gt) == set(child_list_pred):
                    ct['children_set'] += 1
                    if child_list_gt == child_list_pred:
                        ct['children_order'] += 1

    result = {}

    for k, v in ct.items():
        result[k] = {}
        pre = v / len(pred_nodes)
        rec = v / len(gt_nodes)
        result[k]['precision'] = pre
        result[k]['recall'] = rec

        if pre > 0 and rec > 0:
            result[k]['f1'] = 2 * pre * rec / (pre + rec)
        else:
            result[k]['f1'] = 0

    if round_n_digit > 0:
        for k1, d in result.items():
            for k2, v in d.items():
                result[k1][k2] = round(result[k1][k2], round_n_digit)

    return result
