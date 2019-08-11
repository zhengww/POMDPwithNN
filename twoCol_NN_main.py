from twoCol_NN_train import *
from twoCol_NN_agent import *

def main():
    existingPOMDP = True
    trainedNN = False
    additionalTraining = True

    parametersAgent = [0.2, 0.15, 0.1, 0.08, 0.2, 0.15, 0.3, 5, 0.42, 0.66, 0.1]
    parametersExp = [0.15, 0.1, 0.05, 0.04, 0.4, 0.6]
    parametersExp_test = parametersExp

    nq = 10
    nl = 3
    nr = 2
    na = 5
    discount = 0.99
    Numcol = parametersAgent[7]  # number of colors
    Ncol = Numcol - 1  # number value: 0 top Numcol-1
    Nf = na + nr + nl + Numcol * 2

    """
    set parameters for training
    """
    input_size = Nf  # na+nr+nl+Ncol1+Ncol2
    hidden_size_bel = 300  # number of neurons
    output_size_bel = 2  # belief for the box (continuous)
    hidden_size_act = 100
    output_size_act = na
    batch_size = 5
    num_layers = 1
    train_ratio = 0.9

    NEpochs_bel = 60
    NEpochs_act = 60

    nn_params = [input_size, hidden_size_bel, output_size_bel, hidden_size_act, output_size_act, num_layers]
    POMDP_params = [nq, na, nr, nl, Numcol, discount]
    training_params = [batch_size, train_ratio, NEpochs_bel, NEpochs_act]

    if existingPOMDP:
        datestring_data = '07172019(1604)'
        dataN_pkl_file = open(path + '/Results/' + datestring_data + '_dataN_twoboxCol.pkl', 'rb')
        dataN_pkl = pickle.load(dataN_pkl_file)
        dataN_pkl_file.close()
        obsN = dataN_pkl['observations']
        latN = dataN_pkl['beliefs']

        sample_number = obsN.shape[0]
        sample_length = obsN.shape[1]
    else:
        sample_length = 500
        sample_number = 200

        obsN, latN, truthN, datestring_data = twoboxColGenerate(parametersAgent, parametersExp, sample_length, sample_number,
                                                       nq, nr, nl, na, belief1Initial=np.random.randint(nq), rewInitial=np.random.randint(nr),
                                                       belief2Initial=np.random.randint(nq), locationInitial=np.random.randint(nl))

    act = obsN[:, :, 0]
    actionSel = np.array([len(np.where(act == 0)[0]),
                          len(np.where(act == 1)[0]),
                          len(np.where(act == 2)[0]),
                          len(np.where(act == 3)[0]),
                          len(np.where(act == 4)[0])])
    actionSel = actionSel / np.sum(actionSel)
    obsN1, latN1, truthN1, datestring_closeloop = twoboxColGenerate_offpolicy(actionSel, parametersAgent, parametersExp, sample_length,
                                                                              sample_number, nq, nr, nl, na, belief1Initial=np.random.randint(nq), rewInitial=np.random.randint(nr),
                                                         belief2Initial=np.random.randint(nq), locationInitial=np.random.randint(nl))


    if trainedNN:
        datestring_train = '08102019(2344)'
        checkpoint_bel = torch.load(path + '/Results/' + datestring_train + '_belNN' + '_epoch' + str(NEpochs_bel) + '_data' + datestring_data)
        rnn = rnn_bel(input_size, hidden_size_bel, output_size_bel, num_layers)
        rnn.load_state_dict(checkpoint_bel['belNN_state_dict'])
        rnn.eval()

        checkpoint_act = torch.load(path + '/Results/' + datestring_train + '_actNN' + '_epoch' + str(NEpochs_act) + '_data' + datestring_data)
        net = net_act(hidden_size_bel, hidden_size_act, output_size_act)
        net.load_state_dict(checkpoint_act['actNN_state_dict'])
        net.eval()
    else:
        datestring_train = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')
        rnn = rnn_bel(input_size, hidden_size_bel, output_size_bel, num_layers)
        net = net_act(hidden_size_bel, hidden_size_act, output_size_act)

        if additionalTraining:
            rnn, net = training_double(obsN, latN, obsN1, latN1, POMDP_params, training_params, rnn, net,
                                       datestring_data, datestring_train)
        else:
            rnn, net = training(obsN, latN, POMDP_params, training_params, rnn, net, datestring_data, datestring_train)


        data_dict = agent_NNandPOMDP_NN(rnn, net, parametersExp_test, POMDP_params, parametersAgent, nn_params, N = 20, T = sample_length - 1)
        datestring_NNagent = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')
        data_output = open(path + '/Results/' + datestring_train  + '_data' + datestring_data + '_agentNNdriven' + datestring_NNagent + '_twoboxCol' + '.pkl', 'wb')
        pickle.dump(data_dict, data_output)
        data_output.close()

        data_dict1 = agent_NNandPOMDP_POMDP(rnn, net, parametersExp_test, N=20, T=sample_length - 1)
        datestring_NNagent1 = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')
        data_output1 = open(
            path + '/Results/' + datestring_train + '_data' + datestring_data + '_agentPOMDPdriven' + datestring_NNagent1 + '_twoboxCol' + '.pkl',
            'wb')
        pickle.dump(data_dict1, data_output1)
        data_output.close()

if __name__ == "__main__":
    main()