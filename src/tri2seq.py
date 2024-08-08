import os
os.chdir("..")
import utils_entity
import numpy as np
from tqdm import trange
from scipy import sparse

def process_triples(matrix, max_sequence_length=-1,m=-1,r=-1,history_times=-1):
    t = len(matrix)
    sequences_dict = np.full((m,max_sequence_length,3),-1,dtype=int)
    # sequences_dict_one = np.full((m,512,3),1,dtype=int)
    for ti in trange(0,t):
        inverse_test_triplets = matrix[ti][:, [2, 1, 0]] # inverse triplets 
        inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + r
        matrix[ti] = np.concatenate((matrix[ti], inverse_test_triplets))
    for ti in trange(0,t):
        if ti == 0:
            np.save('./data/{}/history_seq/h_r_seen_triple_{}.npy'.format(dataset,str(history_times[ti])), sparse.csr_matrix((sequences_dict+1)[matrix[ti][:,0]].reshape(len(matrix[ti]),-1)))
            continue
        for mi in matrix[ti-1]:
            s, r, o = mi
            current_sequence = sequences_dict[s]
            # 将当前（o, r）添加到序列中
            current_sequence = np.insert(current_sequence, len(current_sequence), np.array([o, r,ti-1],int), axis=0)
            # 限制序列长度为max_sequence_length
            if len(current_sequence) > max_sequence_length:
                current_sequence = current_sequence[-max_sequence_length:]
            sequences_dict[s] = current_sequence.copy()
        np.save('./data/{}/history_seq/h_r_seen_triple_{}'.format(dataset,str(history_times[ti])), sparse.csr_matrix((sequences_dict+1)[matrix[ti][:,0]].reshape(len(matrix[ti]),-1)))
for dataset in ['ICEWS14','ICEWS18','GDELT','ICEWS05_15']:
    data = utils_entity.load_data(dataset)
    train_list = utils_entity.split_by_time(data.train)  # [((s, r, o), ...), ...] len = num_date, do not contain inverse triplets
    valid_list = utils_entity.split_by_time(data.valid)
    test_list = utils_entity.split_by_time(data.test)
    train_times = np.array(sorted(set(data.train[:, 3])))
    val_times = np.array(sorted(set(data.valid[:, 3])))
    test_times = np.array(sorted(set(data.test[:, 3])))
    history_times = np.concatenate((train_times, val_times, test_times), axis=None)
    train_list.extend(valid_list)
    train_list.extend(test_list)
    process_triples(train_list, max_sequence_length=128,m=data.num_nodes,r=data.num_rels,history_times=history_times)