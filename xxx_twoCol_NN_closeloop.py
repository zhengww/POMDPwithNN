import os
import pickle
from datetime import datetime

path = os.getcwd()
datestring_NNagent = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')

from twoboxCol import *
from twoCol_NN_params import *
from twoCol_NN_model import *
from twoCol_NN_data import *
from POMDP_generate import *


checkpoint_bel = torch.load(path + '/Results/' + datestring_train + '_belNN' + '_epoch' + str(NEpochs_bel) + '_data' + datestring_data)
model_bel = rnn(input_size, hidden_size_bel, output_size_bel, num_layers)
model_bel.load_state_dict(checkpoint_bel['belNN_state_dict'])
optimizer_bel = torch.optim.Adam(model_bel.parameters(), lr=0.001)
optimizer_bel.load_state_dict(checkpoint_bel['bel_optimizer_state_dict'])
epoch_bel = checkpoint_bel['epoch']
loss_bel = checkpoint_bel['bel_loss']
model_bel.train()

checkpoint_act = torch.load(path + '/Results/' + datestring_train + '_actNN' + '_epoch' + str(NEpochs_act) + '_data' + datestring_data)
model_act = net(hidden_size_bel, hidden_size_act, output_size_act)
model_act.load_state_dict(checkpoint_act['actNN_state_dict'])
optimizer_act = torch.optim.Adam(model_act.parameters(), lr=0.001)
optimizer_act.load_state_dict(checkpoint_act['act_optimizer_state_dict'])
epoch_act = checkpoint_act['epoch']
loss_act = checkpoint_act['act_loss']
model_act.train()


dataN_pkl_file = open(path + '/Results/07172019(1604)_dataN_twoboxCol.pkl', 'rb')
dataN_pkl = pickle.load(dataN_pkl_file)
dataN_pkl_file.close()
obs = dataN_pkl['observations']
act = obs[:,:, 0]
actionSel = np.array([len(np.where(act == 0)[0]),
          len(np.where(act == 1)[0]),
          len(np.where(act == 2)[0]),
          len(np.where(act == 3)[0]),
          len(np.where(act == 4)[0])])
actionSel = actionSel / np.sum(actionSel)


obsN, latN, truthN, datestring_closeloop = twoboxColGenerate_offpolicy(actionSel, parametersAgent, parametersExp, sample_length, sample_number,
                                                    nq, nr, nl, na, belief1Initial=np.random.randint(nq), rewInitial=np.random.randint(nr),
                                                    belief2Initial=np.random.randint(nq), locationInitial=np.random.randint(nl))

xMatFull, yMatFull = preprocessData(obsN, latN, nq, na, nr, nl, Numcol)
train_loader, test_loader = splitData(xMatFull, yMatFull, train_ratio, batch_size)

criterion = torch.nn.MSELoss()
"""
Train belief network module
"""
train_loss_bel = np.zeros([NEpochs_bel])

for epoch in range(NEpochs_bel):
    for i, data in enumerate(train_loader, 0):
        in_batch, target_batch = data

        out_bel_batch, _ = model_bel(in_batch)

        loss = criterion(out_bel_batch.squeeze(), target_batch[:, :, 0:2])

        optimizer_bel.zero_grad()  # zero-gradients at the start of each epoch
        loss.backward()
        optimizer_bel.step()

    train_loss_bel[epoch] = loss.item()

    if epoch % 10 == 9:
        print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.item()))
    if epoch == NEpochs_bel - 1:
        torch.save({
            'epoch': epoch,
            'belNN_state_dict': model_bel.state_dict(),
            'bel_optimizer_state_dict': optimizer_bel.state_dict(),
            'bel_loss': loss,
        }, path + '/Results/' + datestring_train + '_belNN' +'_CLtraining' + datestring_closeloop
            + '_data' + datestring_data)


print("Belief network learning finished!")


criterion_act = torch.nn.KLDivLoss(reduction = "batchmean")
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

        out_bel_batch, hidden_batch = model_bel(in_batch)
        out_act_batch = model_act(hidden_batch)

        loss = criterion_act(torch.log(out_act_batch), target_actDist_batch)

        optimizer_act.zero_grad()  # zero-gradients at the start of each epoch
        loss.backward()
        optimizer_act.step()

    train_loss_act[epoch] = loss.item()

    if epoch % 10 == 9:
        print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.item()))
    if epoch == epoch_act:
        torch.save({
            'epoch': epoch,
            'actNN_state_dict': model_act.state_dict(),
            'act_optimizer_state_dict': optimizer_act.state_dict(),
            'act_loss': loss,
        }, path + '/Results/' + datestring_train + '_actNN' + '_CLtraining' + datestring_closeloop + '_data' + datestring_data)

print("Action network learning finished!")
