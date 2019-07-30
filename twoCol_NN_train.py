import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data_utils

from twoCol_NN_params import *
from POMDP_generate import *
from twoCol_NN_POMDPdata import *
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
# parametersAgent = [0.2,0.15,0.1,0.08,0.2,0.15,0.3,5,0.42,0.66, 0.1]
# #parametersExp = [0.15,0.1,0.05,0.04,0.4,0.6]
# parametersExp = [0.2,0.15,0.1,0.08,0.42,0.66]
#
# nq = 10
# nl = 3
# nr = 2
# na = 5
# Ncol1 = parametersAgent[7]
# Ncol2 = parametersAgent[7]
# Nf = na + nr + nl + Ncol1 + Ncol2

# sample_length = 500
# sample_number = 200
# obsN, latN, truthN, datestring_data = twoboxColGenerate(parametersAgent, parametersExp, sample_length, sample_number,
#                                                    nq, nr, nl, na, belief1Initial=np.random.randint(nq), rewInitial=np.random.randint(nr),
#                                                    belief2Initial=np.random.randint(nq), locationInitial=np.random.randint(nl))

#datestring_data = '07172019(1604)'
dataN_pkl_file = open(path + '/Results/' + datestring_data + '_dataN_twoboxCol.pkl', 'rb')
dataN_pkl = pickle.load(dataN_pkl_file)
dataN_pkl_file.close()
obsN = dataN_pkl['observations']
latN = dataN_pkl['beliefs']



xMatFull, yMatFull = preprocessData(obsN, latN, nq, na, nr, nl, Ncol1, Ncol2)


# """
# set parameters for training
# """
# input_size = Nf     # na+nr+nl+Ncol1+Ncol2
# hidden_size_bel = 300   # number of neurons
# output_size_bel = 2    # belief for the box (continuous)
# hidden_size_act = 100
# output_size_act = na
# batch_size = 5
# num_layers = 1
# train_ratio = 0.9
#
# NEpochs_bel = 60
# NEpochs_act = 300

train_loader, test_loader = splitData(xMatFull, yMatFull, train_ratio, batch_size)



"""
Create RNN module and set loss and optimization function 
"""
rnn = rnn(input_size, hidden_size_bel, output_size_bel, num_layers)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)


"""
Train belief network module
"""
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