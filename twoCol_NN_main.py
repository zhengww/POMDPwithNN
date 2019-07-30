import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data_utils

from POMDP_generate import *
from twoCol_NN_data import *
from twoCol_NN_model import *


path = os.getcwd()
datestring_run = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')


######################################################
#
#  Generate Data based on POMDP model
#
######################################################

# parametersAgent = [gamma1, gamma2, epsilon1, epsilon2, groom, travelCost, pushButtonCost, NumCol, qmin, qmax, temperature]
#parametersExp = [gamma1, gamma2, epsilon1, epsilon2, qmin, qmax]
parametersAgent = [0.2,0.15,0.1,0.08,0.2,0.15,0.3,5,0.42,0.66, 0.1]
#parametersExp = [0.15,0.1,0.05,0.04,0.4,0.6]
parametersExp = [0.2,0.15,0.1,0.08,0.42,0.66]

nq = 10
nl = 3
nr = 2
na = 5
Ncol1 = parametersAgent[7]
Ncol2 = parametersAgent[7]
Nf = na + nr + nl + Ncol1 + Ncol2

# sample_length = 500
# sample_number = 200
# obsN, latN, truthN, datestring = twoboxColGenerate(parametersAgent, parametersExp, sample_length, sample_number,
#                                                    nq, nr, nl, na, belief1Initial=np.random.randint(nq), rewInitial=np.random.randint(nr),
#                                                    belief2Initial=np.random.randint(nq), locationInitial=np.random.randint(nl))

datestring_data = '07172019(1604)'
dataN_pkl_file = open(path + '/Results/' + datestring_data + '_dataN_twoboxCol.pkl', 'rb')
dataN_pkl = pickle.load(dataN_pkl_file)
dataN_pkl_file.close()
obsN = dataN_pkl['observations']
latN = dataN_pkl['beliefs']


######################################################
#
#  data from two box with color
#  consider color as discrete with one-hot encoding
#
######################################################

# def one_hot_encode(data, dict_size, seq_len, sample_num):
#     # Creating a multi-dimensional array of zeros with the desired output shape
#     features = np.zeros((sample_num, seq_len, dict_size))
#
#     # Replacing the 0 at the relevant character index with a 1 to represent that character
#     for i in range(sample_num):
#         for u in range(seq_len):
#             features[i, u, data[i][u]] = 1
#     return features
#
# Ns = obsN.shape[0]
# Nt = obsN.shape[1] - 1
# xMatFull = np.zeros((Ns, Nt, Nf), dtype=int)
#
# act_onehot = one_hot_encode(obsN[:, 0:-1, 0].astype(int), na, Nt, Ns)
# rew_onehot = one_hot_encode(obsN[:, 1:, 1].astype(int), nr, Nt, Ns)
# loc_onehot = one_hot_encode(obsN[:, 1:, 2].astype(int), nl, Nt, Ns)
# col1_onehot = one_hot_encode(obsN[:, 1:, 3].astype(int), Ncol1, Nt, Ns)
# col2_onehot = one_hot_encode(obsN[:, 1:, 4].astype(int), Ncol2, Nt, Ns)
# xMatFull[:, :, :] =  np.concatenate((act_onehot, rew_onehot, loc_onehot, col1_onehot, col2_onehot), axis = 2)  # cascade all the input
# # 5  + 2 + 3 + 5 + 5
#
#
# belief = (latN[:, 1:, 0:2] + 0.5)/nq
# actout = obsN[:, 1:, 0:1]
# act_dist = obsN[:, 1:, 5:]
# yMatFull = np.concatenate((belief, actout, act_dist), axis = 2)  # cascade output

xMatFull, yMatFull = preprocessData(obsN, latN, nq, na, nr, nl, Ncol1, Ncol2)

######################################################
#
#  RNN model for belief network
#
######################################################

"""
set parameters for training
"""
input_size = Nf     # na+nr+nl+Ncol1+Ncol2
hidden_size_bel = 300   # number of neurons
output_size_bel = 2    # belief for the box (continuous)
hidden_size_act = 100
output_size_act = na
batch_size = 5
sequence_length =  obsN.shape[1] - 1
num_layers = 1


######################################################
#
#  Convert ground truth data to torch tensors
#  Split training and testing data
#
######################################################
train_ratio = 0.9
# dataset = data_utils.TensorDataset(torch.tensor(xMatFull, dtype = torch.float),
#                                     torch.tensor(yMatFull, dtype = torch.float))
#
# train_set, test_set = data_utils.random_split(dataset, [int(Ns * train_ratio), Ns - int(Ns * train_ratio)])
# train_loader = data_utils.DataLoader(train_set, batch_size, shuffle = True)
# test_loader = data_utils.DataLoader(test_set, batch_size)

train_loader, test_loader = splitData(xMatFull, yMatFull, train_ratio, batch_size)



"""
Create RNN module and set loss and optimization function 
"""
rnn = RNN(input_size, hidden_size_bel, output_size_bel, num_layers)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)


"""
Train belief network module
"""
NEpochs_bel = 2
train_loss_bel = np.zeros([NEpochs_bel])

for epoch in range(NEpochs_bel):
    for i, data in enumerate(train_loader, 0):
        in_batch, target_batch = data

        out_bel_batch, _ = rnn(in_batch)

        loss = criterion(out_bel_batch.squeeze(), target_batch[:, :, 0:2])

        optimizer.zero_grad()  # zero-gradients at the start of each epoch
        loss.backward()
        optimizer.step()

    train_loss_bel[epoch] = loss.item()

    if epoch % 10 == 9:
        print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.item()))
        torch.save({
            'epoch': epoch,
            'belNN_state_dict': rnn.state_dict(),
            'bel_optimizer_state_dict': optimizer.state_dict(),
            'bel_loss': loss,
        }, path + '/Results/' + datestring_run + '_belNN' + '_epoch' + str(epoch+1) + '_data' + datestring_data)


print("Belief network learning finished!")




net = net(hidden_size_bel, hidden_size_act, output_size_act)

criterion_act = torch.nn.KLDivLoss(reduction = "batchmean")
optimizer_act = torch.optim.Adam(rnn.parameters(), lr=0.001)

"""
Train action network module
"""
NEpochs_act = 3
train_loss_act = np.zeros([NEpochs_act])

for epoch in range(NEpochs_act):
    count = 0
    correct_count = 0

    for i, data in enumerate(train_loader, 0):
        in_batch, target_batch = data
        target_act_batch = target_batch[:, :, 2]
        target_actDist_batch = target_batch[:, :, 3:]

        out_bel_batch, hidden_batch = rnn(in_batch)
        out_act_batch = net(hidden_batch)

        loss = criterion_act(torch.log(out_act_batch), target_actDist_batch)

        optimizer_act.zero_grad()  # zero-gradients at the start of each epoch
        loss.backward()
        optimizer_act.step()

    train_loss_act[epoch] = loss.item()

    if epoch % 10 == 9:
        print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.item()))
        torch.save({
            'epoch': epoch,
            'actNN_state_dict': net.state_dict(),
            'act_optimizer_state_dict': optimizer_act.state_dict(),
            'act_loss': loss,
        }, path + '/Results/' + datestring_run + '_actNN' + '_epoch' + str(epoch + 1) + '_data' + datestring_data)

print("Action network learning finished!")