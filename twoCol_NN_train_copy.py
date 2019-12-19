import matplotlib.pyplot as plt

from POMDP_generate import *
from twoCol_NN_data_utils import *
from twoCol_NN_model import *

"""
######################################################
#
#  train nn network (two phases: belief nn first, then action network)
#  NN Architecture:
#  Input layer --> Hidden recurrent layer --> Linear readout -- > belief
#                                         --> Linear with softmax -- > action 
######################################################   
"""

def training(obsN, latN, POMDP_params, training_params, nn_params, datestring_data, datestring_train):

    sample_length = obsN.shape[1]
    nq, na, nr, nl, Numcol, discount, parametersAgent, parametersExp, parametersExp_test= POMDP_params
    input_size, hidden_size_bel, output_size_bel, hidden_size_act, output_size_act, num_layers = nn_params
    batch_size, train_ratio, NEpochs_bel, NEpochs_act, lr_bel, lr_act = training_params

    # save neural network and training parameters
    nn_train_para_dict = {'POMDP_params': POMDP_params,
                          'nn_params': nn_params,
                          'training_params': training_params
                         }
    nn_train_para_output = open(path + '/Results/' + datestring_train + '_data' + datestring_data + '_nn_train_params_twoboxCol.pkl', 'Wb')
    pickle.dump(nn_train_para_dict, nn_train_para_output)
    nn_train_para_output.close()

    xMatFull, yMatFull = preprocessData(obsN, latN, nq, na, nr, nl, Numcol)
    train_loader, test_loader = splitData(xMatFull, yMatFull, train_ratio, batch_size)

    """
    Create modules and set loss and optimization function 
    """
    rnn = rnn_bel(input_size, hidden_size_bel, output_size_bel, num_layers)  #belief network
    net = net_act(hidden_size_bel, hidden_size_act, output_size_act)         #action network

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr = lr_bel)

    criterion_act = torch.nn.KLDivLoss(reduction="batchmean")
    optimizer_act = torch.optim.Adam(net.parameters(), lr = lr_act)

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
            }, path + '/Results/' + datestring_train + '_belNN' + '_epoch' + str(epoch + 1) + '_data' + datestring_data)

    print("Belief network learning finished!")

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
            }, path + '/Results/' + datestring_train + '_actNN' + '_epoch' + str(epoch + 1) + '_data' + datestring_data)

    print("Action network learning finished!")

    # ###### FIGURES #############
    # """
    # Plot the training error as a function of epochs
    # """
    # plt.plot(train_loss_bel)
    # plt.title('Training error of beliefs')
    # plt.ylabel('mse of beliefs')
    # plt.xlabel('Epochs')
    # plt.show()
    #
    # """
    # Plot the training error as a function of epochs
    # """
    # plt.plot(train_loss_act / (sample_length - 1))
    # plt.title('Training error of policy')
    # plt.ylabel('KL of action')
    # plt.xlabel('Epochs')
    # plt.show()
    #
    # """
    # Cross-validation with testing data
    # """
    #
    # count = 0
    # correct_count = 0
    # with torch.no_grad():
    #     for i, batch in enumerate(test_loader):
    #         in_batch, target_batch = batch
    #         target_bel_batch = target_batch[:, :, 0:2]
    #         target_act_batch = target_batch[:, :, 2]
    #         target_actDist_batch = target_batch[:, :, 3:]
    #
    #         out_bel_batch, hidden_batch = rnn(in_batch)
    #         out_act_batch = net(hidden_batch)
    #
    #         act_predicted = np.zeros((batch_size, sample_length - 1))
    #         for i in range(batch_size):
    #             for j in range(sample_length - 1):
    #                 act_predicted[i, j] = np.argmax(np.random.multinomial(1, out_act_batch[i, j, :]))
    #         act_predicted = torch.from_numpy(act_predicted).to(torch.long)
    #
    #         count += in_batch.shape[0] * in_batch.shape[1]
    #         correct_count += (act_predicted == target_act_batch.to(torch.long)).sum().item()
    #
    # print('MSE of belief: %f' % criterion(out_bel_batch.squeeze(), target_batch[:, :, 0:2]).item())
    #
    # T_st = 100
    # T_end = 150
    # time_range = np.arange(T_end - T_st)
    #
    # fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    # bef1_true = target_bel_batch[1:2, T_st:T_end, 0].float().squeeze().detach().numpy()
    # bef1_est = out_bel_batch[1:2, T_st:T_end, 0].detach().squeeze().numpy()
    # ax[0].plot(time_range, bef1_est, time_range, bef1_true)
    # ax[0].legend(('bef_est', 'bef_true'))
    # ax[0].set_yticks(np.arange(0.5 / nq, 1, step=1 / nq))
    # ax[0].set_title('belief one box 1')
    # ax[0].set_ylabel('belief')
    # ax[0].set_xlabel('time')
    # bef2_true = target_bel_batch[1:2, T_st:T_end, 1].float().squeeze().detach().numpy()
    # bef2_est = out_bel_batch[1:2, T_st:T_end, 1].detach().squeeze().numpy()
    # ax[1].plot(time_range, bef2_est, time_range, bef2_true)
    # ax[1].legend(('bef_est', 'bef_true'))
    # ax[1].set_yticks(np.arange(0.5 / nq, 1, step=1 / nq))
    # ax[1].set_title('belief on box 2')
    # ax[1].set_ylabel('belief')
    # ax[1].set_xlabel('time')
    # plt.show()
    #
    # fig, ax = plt.subplots(2, 1, figsize=(12, 3), sharex=True)
    # ax[0].imshow(target_batch[0, 0:100:, 3:].numpy().T, vmin=0, vmax=1)
    # ax[0].set(xlabel='time', ylabel='true policy')
    # ax[1].imshow(out_act_batch[0, 0:100, :].numpy().T, vmin=0, vmax=1)
    # ax[1].set(xlabel='time', ylabel='est policy')
    # plt.show()

    return rnn, net, test_loader, train_loader

