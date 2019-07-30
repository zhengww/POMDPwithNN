import numpy as np
import torch
import torch.utils.data as data_utils

def one_hot_encode(data, dict_size, seq_len, sample_num):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((sample_num, seq_len, dict_size))

    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(sample_num):
        for u in range(seq_len):
            features[i, u, data[i][u]] = 1
    return features


def preprocessData(obsN, latN, nq, na, nr, nl, Ncol1, Ncol2):
    Ns = obsN.shape[0]
    Nt = obsN.shape[1] - 1
    Nf = na + nr + nl + Ncol1 + Ncol2
    xMatFull = np.zeros((Ns, Nt, Nf), dtype=int)

    act_onehot = one_hot_encode(obsN[:, 0:-1, 0].astype(int), na, Nt, Ns)
    rew_onehot = one_hot_encode(obsN[:, 1:, 1].astype(int), nr, Nt, Ns)
    loc_onehot = one_hot_encode(obsN[:, 1:, 2].astype(int), nl, Nt, Ns)
    col1_onehot = one_hot_encode(obsN[:, 1:, 3].astype(int), Ncol1, Nt, Ns)
    col2_onehot = one_hot_encode(obsN[:, 1:, 4].astype(int), Ncol2, Nt, Ns)
    xMatFull[:, :, :] = np.concatenate((act_onehot, rew_onehot, loc_onehot, col1_onehot, col2_onehot),
                                       axis=2)  # cascade all the input
    # 5  + 2 + 3 + 5 + 5

    belief = (latN[:, 1:, 0:2] + 0.5) / nq
    actout = obsN[:, 1:, 0:1]
    act_dist = obsN[:, 1:, 5:]
    yMatFull = np.concatenate((belief, actout, act_dist), axis=2)  # cascade output

    return xMatFull, yMatFull


def splitData(xMatFull, yMatFull, train_ratio, batch_size):
    Ns = xMatFull.shape[0]
    dataset = data_utils.TensorDataset(torch.tensor(xMatFull, dtype=torch.float),
                                       torch.tensor(yMatFull, dtype=torch.float))

    train_set, test_set = data_utils.random_split(dataset, [int(Ns * train_ratio), Ns - int(Ns * train_ratio)])
    train_loader = data_utils.DataLoader(train_set, batch_size, shuffle=True)
    test_loader = data_utils.DataLoader(test_set, batch_size)

    return train_loader, test_loader
