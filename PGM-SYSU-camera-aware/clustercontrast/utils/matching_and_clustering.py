import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
import copy 
from scipy.optimize import linear_sum_assignment as linear_assignment


def cluster_acc(pred_label, gt_label):
    D = max(pred_label.max(), gt_label.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)  # w=[[1 5 0],[4 1 1],[0 2 3]]
    for i in range(pred_label.size):
        w[pred_label[i], gt_label[i]] += 1
    ind = linear_assignment(w.max() - w) # assignment= [[0 1],[1 0],[2 2]]
    ind = np.vstack(ind).T
    acc = sum([w[i, j] for i, j in ind]) * 1.0 / len(pred_label)
    return acc



def two_step_hungarian_matching(proxy_features_rgb, proxy_features_ir, obtain_pseudo_label=False):
    i2r = {}
    r2i = {}
    R = []
    assert(len(proxy_features_rgb) >= len(proxy_features_ir))
    similarity = torch.mm(proxy_features_rgb, proxy_features_ir.T).exp().cpu()
    cost = 1 / similarity
    tmp = torch.zeros(cost.shape[0], cost.shape[0] - cost.shape[1])
    cost = (torch.cat((cost, tmp), 1))
    unmatched_row = []
    row_ind, col_ind = linear_sum_assignment(cost)

    for idx, item in enumerate(row_ind):
        if col_ind[idx] < similarity.shape[1]:
            R.append((row_ind[idx], col_ind[idx]))
            r2i[row_ind[idx]] = col_ind[idx]
            i2r[col_ind[idx]] = row_ind[idx]
        else:
            unmatched_row.append(row_ind[idx])

    print(f'hungarian matching: after step-1 matching, unmatched row num= {len(unmatched_row)}, total ir feat num= {len(proxy_features_ir)}')

    # step-2
    #assert(len(unmatched_row) < len(proxy_features_ir))
    if len(unmatched_row) > len(proxy_features_ir):
        unmatched_row_new = []
        unmatched_cost = cost[unmatched_row][:, :similarity.shape[1]]
        tmp = torch.zeros(unmatched_cost.shape[0], unmatched_cost.shape[0] - unmatched_cost.shape[1])
        unmatched_cost = (torch.cat((unmatched_cost, tmp), 1))
        unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost)
        for idx, item in enumerate(unmatched_row_ind):
            if unmatched_col_ind[idx] < similarity.shape[1]:
                R.append((unmatched_row[idx], unmatched_col_ind[idx]))
                r2i[unmatched_row[idx]] = unmatched_col_ind[idx]
            else:
                unmatched_row_new.append(unmatched_row[idx])
        unmatched_row = copy.deepcopy(unmatched_row_new)

    # step-3
    if len(unmatched_row) > 0:
        unmatched_cost = cost[unmatched_row][:, :similarity.shape[1]]
        unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost)
        for idx, item in enumerate(unmatched_row_ind):
            if unmatched_col_ind[idx] < similarity.shape[1]:
                R.append((unmatched_row[idx], unmatched_col_ind[idx]))
                r2i[unmatched_row[idx]] = unmatched_col_ind[idx]
   
    tensor_r2i = -1 * torch.ones(len(proxy_features_rgb)).long().cuda()
    for k,v in r2i.items():
        tensor_r2i[k] = v

    tensor_i2r = -1 * torch.ones(len(proxy_features_ir)).long().cuda()
    for k,v in i2r.items():
        tensor_i2r[k] = v

    if obtain_pseudo_label:
        pseudo_label = -1*torch.ones(len(proxy_features_rgb)+len(proxy_features_ir))
        rgbNum = len(proxy_features_rgb)
        cnt = 0
        for k,v in r2i.items():
            v = v + rgbNum
            if pseudo_label[v] == -1:
                pseudo_label[k] = cnt
                pseudo_label[v] = cnt
                cnt += 1
            else:
                pseudo_label[k] = pseudo_label[v]
        print('hungarian matched pseudo label number= ', len(torch.unique(pseudo_label)))
        return tensor_i2r, tensor_r2i, pseudo_label.cuda()
    else:
        return tensor_i2r, tensor_r2i

