import numpy as np
import torch

import knowledge_graph as knwlgrh
import pandas as pd
import logging
import dgl
from torch_scatter import scatter_sum
import random
import os
def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    # indices = torch.gather(indices, 1, target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def sort_and_rank_multi(score, target):
    target_c = target.unsqueeze(0).repeat(score.shape[0],1)
    _, indices = torch.sort(score, dim=2, descending=True)
    indices = torch.nonzero(indices == target_c.view(score.shape[0],-1, 1))
    # indices = torch.gather(indices, 1, target.view(-1, 1))
    indices = indices[:, 2].view(score.shape[0],-1)
    return indices

def sort_and_rank_tuple(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    # indices = torch.gather(indices, 1, target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def filter_score_sub(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans.keys())
        ans.remove(h.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -100000  #
    return score

def filter_score_obj(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    score_c = score.clone()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -100000  #
    return score,score_c

def filter_score_obj_multi(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    score_c = score.clone()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[:,_,ans] = -100000  #
    return score,score_c

def filter_score_rel(test_triples, score, all_ans,num_rel):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t = triple
        ans = list(all_ans[h.item()][t.item()])
        ans.remove(r.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score


def get_total_rank(test_triples, score, all_ans, eval_bz, select_type,num_rel,filer):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz

    filter_rank = []
    rank =[]
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]

        target = test_triples[batch_start:batch_end, select_type-1]
        # if filer:
        if select_type == 1:
            filter_score_batch = filter_score_sub(triples_batch, score_batch, all_ans)
        elif select_type == 2:
            filter_score_batch = filter_score_rel(triples_batch, score_batch, all_ans,num_rel)
        else:
            filter_score_batch,score_c = filter_score_obj(triples_batch, score_batch, all_ans)
        filter_rank.append(sort_and_rank(filter_score_batch, target))
        # else:
        rank.append(sort_and_rank(score_c, target))
            
    filter_rank = torch.cat(filter_rank)
    filter_rank += 1 # change to 1-indexed
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    
    rank = torch.cat(rank)
    rank += 1 # change to 1-indexed
    mrr = torch.mean(1.0 / rank.float())
    return filter_mrr.item(), filter_rank, mrr.item(),rank
    # filter_rank = torch.cat(filter_rank,-1)
    # filter_rank += 1 # change to 1-indexed
    # filter_mrr = torch.mean(1.0 / filter_rank.float(),-1)
    
    # rank = torch.cat(rank,-1)
    # rank += 1 # change to 1-indexed
    # mrr = torch.mean(1.0 / rank.float(),-1)
    
    # max_index = torch.argmax(mrr)

    # return filter_mrr[max_index].item(), filter_rank[max_index], mrr[max_index].item(),rank[max_index]
    
def get_total_rank_multi(test_triples, score, all_ans, eval_bz, select_type,num_rel,filer):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz

    filter_rank = []
    rank =[]
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]

        target = test_triples[batch_start:batch_end, select_type-1]
        # if filer:
        if select_type == 1:
            filter_score_batch = filter_score_sub(triples_batch, score_batch, all_ans)
        elif select_type == 2:
            filter_score_batch = filter_score_rel(triples_batch, score_batch, all_ans,num_rel)
        else:
            filter_score_batch,score_c = filter_score_obj_multi(triples_batch, score_batch, all_ans)
        filter_rank.append(sort_and_rank_multi(filter_score_batch, target))
        # else:
        rank.append(sort_and_rank_multi(score_c, target))
            
    # filter_rank = torch.cat(filter_rank)
    # filter_rank += 1 # change to 1-indexed
    # filter_mrr = torch.mean(1.0 / filter_rank.float())
    
    # rank = torch.cat(rank)
    # rank += 1 # change to 1-indexed
    # mrr = torch.mean(1.0 / rank.float())
    # return filter_mrr.item(), filter_rank, mrr.item(),rank
    filter_rank = torch.cat(filter_rank,-1)
    filter_rank += 1 # change to 1-indexed
    filter_mrr = torch.mean(1.0 / filter_rank.float(),-1)
    
    rank = torch.cat(rank,-1)
    rank += 1 # change to 1-indexed
    mrr = torch.mean(1.0 / rank.float(),-1)
    
    max_index = torch.argmax(mrr)

    return filter_mrr[max_index].item(), filter_rank[max_index], mrr[max_index].item(),rank[max_index]

def get_total_rank_test(test_triples, score, all_ans, eval_bz, select_type,num_rel,filer):
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz

    filter_rank = []
    rank =[]
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :]
        score_batch = score[batch_start:batch_end, :]

        target = test_triples[batch_start:batch_end, select_type-1]
        # if filer:
        if select_type == 1:
            filter_score_batch = filter_score_sub(triples_batch, score_batch, all_ans)
        elif select_type == 2:
            filter_score_batch = filter_score_rel(triples_batch, score_batch, all_ans,num_rel)
        else:
            filter_score_batch,score_c = filter_score_obj(triples_batch, score_batch, all_ans)
        filter_rank.append(sort_and_rank(filter_score_batch, target))
        # else:
        rank.append(sort_and_rank(score_c, target))
            
    filter_rank = torch.cat(filter_rank)
    filter_rank += 1 # change to 1-indexed
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    
    rank = torch.cat(rank)
    rank += 1 # change to 1-indexed
    mrr = torch.mean(1.0 / rank.float())

    return filter_mrr.item(), filter_rank, mrr.item(),rank

# def get_filtered_score(test_triples, score, all_ans, eval_bz, rel_predict=0):
#     num_triples = len(test_triples)
#     n_batch = (num_triples + eval_bz - 1) // eval_bz

#     filtered_score = []
#     for idx in range(n_batch):
#         batch_start = idx * eval_bz
#         batch_end = min(num_triples, (idx + 1) * eval_bz)
#         triples_batch = test_triples[batch_start:batch_end, :]
#         score_batch = score[batch_start:batch_end, :]
#         filtered_score.append(filter_score(triples_batch, score_batch, all_ans))
#     filtered_score = torch.cat(filtered_score,dim =0)

#     return filtered_score


def popularity_map(tuple_tensor, head_ents,select_type):
    tag = 'head' if tuple_tensor[select_type].item() in head_ents else 'other'
    return tag


def cal_ranks(rank_list, mode):

    hits_log = []
    hits = [1, 3, 10]
    rank_list = torch.cat(rank_list)

    mrr_debiased = torch.mean(1.0 / rank_list.float())

    for hit in hits:
        avg_count_ent_debiased = torch.mean((rank_list <= hit).float())
        hits_log.append(avg_count_ent_debiased.item())

    return mrr_debiased,hits_log


def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    def add_relation(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not e2 in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)

    def add_subject(e1, e2, r, d, num_rel):
        if not e2 in d:
            d[e2] = {}
        if not r + num_rel in d[e2]:
            d[e2][r + num_rel] = set()
        d[e2][r + num_rel].add(e1)

    def add_object(e1, e2, r, d, num_rel):
        if not e1 in d:
            d[e1] = {}
        if not r in d[e1]:
            d[e1][r] = set()
        d[e1][r].add(e2)

    all_ans = {}
    for line in total_data:
        s, r, o = line[: 3]
        if rel_p:
            add_relation(s, o, r, all_ans)
            add_relation(o, s, r + num_rel, all_ans)
        else:
            add_subject(s, o, r, all_ans, num_rel=num_rel)
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    all_snap = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_list.append(all_ans_t)

    return all_ans_list # [ { e1: {e2: (r) / r: (e2) } } ], len = uniq_t in given dataset


def calculate_f1(ground_truth, predicted):
    """
    计算 True Positives (tp), False Positives (fp), 和 False Negatives (fn) 的数量。

    参数:
    ground_truth_set (set): 实际存在的三元组集合，每个三元组为 (S, R, O)。
    predicted_set (set): 模型预测的三元组集合，每个三元组为 (S, R, O)。

    返回:
    int, int, int: tp, fp, fn 的数量。
    """
    ground_truth_set = set(ground_truth)
    predicted_set = set(predicted)
    tp = len(ground_truth_set.intersection(predicted_set))
    fp = len(predicted_set - ground_truth_set)
    fn = len(ground_truth_set - predicted_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1_score



def split_by_time(data):
    snapshot_list = []
    snapshot = [] # ((s, r, o))
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i] # [s, r, o, t]
        if latest_t != t:
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:3])
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # relabel
        # edges: old order = [list of s, then list of o]; the new index of each old element in new order = [uniq_v]
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list


def split_context_by_time_onehot(data, k_contexts):
    onehot_matrix = []
    for context in range(k_contexts):
        onehot = [0] * k_contexts
        onehot[context] = 1
        onehot_matrix.append(onehot.copy())

    snapshot_list = []
    snapshot = [] # (contextid, ...)
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][1]
        train = data[i] # [contextid, t]
        if latest_t != t:
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(onehot_matrix[train[0]])
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1
    return snapshot_list

def split_context_by_time_avg(data, k_contexts):
    avg_vector = np.ones(k_contexts) / k_contexts

    snapshot_list = []
    snapshot = [] # (contextid, ...)
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][1]
        train = data[i] # [contextid, t]
        if latest_t != t:
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(avg_vector.copy())
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1
    return snapshot_list


def load_data(dataset):
    if dataset in ['EG', 'IS', 'IR','ICEWS14','ICEWS14_REL','ICEWS05_15','ICEWS18','WIKI','GDELT','YAGO']:
        return knwlgrh.load_from_local("./data", dataset)
    else:
        return knwlgrh.load_from_local("./data_disentangled", dataset, load_context=True)


def ccorr(a, b):
    """
    Compute circular correlation of two tensors.
    Parameters
    ----------
    a: Tensor, 1D or 2D
    b: Tensor, 1D or 2D
    Notes
    -----
    Input a and b should have the same dimensions. And this operation supports broadcasting.
    Returns
    -------
    Tensor, having the same dimension as the input a.
    """
    return torch.fft.irfftn(torch.conj(torch.fft.rfftn(a, (-1))) * torch.fft.rfftn(b, (-1)), (-1))

def seed_everything(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def getstate():
    return random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state(),torch.cuda.get_rng_state_all()
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    
def setstate(ran_state,np_state,torch_state,cuda_state,cuda_state_all):
    random.setstate(ran_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
    torch.cuda.set_rng_state(cuda_state)
    torch.cuda.set_rng_state_all(cuda_state_all)
    
def log_sum_exp(value,weight_energy, dim=None, keepdim=False,mask_matrix=None):
        """Numerically stable implementation of the operation

        value.exp().sum(dim, keepdim).log()
        """
        import math
        # TODO: torch.max(value, dim=None) threw an error at time of writing
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            # value_replace_0 = scatter_sum(value0.exp(),mask_matrix.long(),dim=-1)[:,1]
            if keepdim is False:
                m = m.squeeze(dim)
            part_energe = torch.log(scatter_sum(torch.relu(weight_energy)*value0.exp(),mask_matrix.long(),dim=-1)[:,1]+1e-8)
            if torch.isnan(part_energe).any():
                print("sum_exp_values contain NaN or inf")
            return m + part_energe
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            # if isinstance(sum_exp, Number):
            #     return m + math.log(sum_exp)
            # else:
            return m + torch.log(sum_exp)
        
def build_sub_graph(num_nodes, num_rels, triples, use_cuda, gpu):
    """
    :param node_id: node id in the large graph
    :param num_rels: number of relation
    :param src: relabeled src id
    :param rel: original rel id
    :param dst: relabeled dst id
    :param use_cuda:
    :return:
    """
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm

    src, rel, dst = triples.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)

    uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    g.uniq_r = uniq_r
    g.r_to_e = r_to_e
    g.r_len = r_len
    if use_cuda:
        g.to(gpu)
        g.r_to_e = torch.from_numpy(np.array(r_to_e))
    return g

from collections import defaultdict

def r2e(triplets, num_rels):
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r+num_rels))
    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel+num_rels].add(src)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx



