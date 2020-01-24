from twoCol_NN_train import *
from twoCol_NN_agent import *

def main():
    # parametersAgent = [0.2, 0.18, 0.1, 0.08, 0.2, 0.15, 0.3, 5, 0.45, 0.55, 0.1]
    # parametersExp = [0.15, 0.1, 0.05, 0.04, 0.4, 0.6]
    #
    # nq = 10
    # nl = 3
    # nr = 2
    # na = 5
    # sample_length = 1000
    # sample_number = 1
    # Numcol = parametersAgent[7]
    #
    # obsN, latN, truthN, datestring = twoboxColGenerate(parametersAgent, parametersExp, sample_length, sample_number,
    #                                                    nq, nr, nl)
    #

    existingPOMDP = True
    trainedNN = False

    if existingPOMDP:
        datestring_data = '08132019(0013)'
        dataN_pkl_file = open(path + '/Results/oldResults/' + datestring_data + '_dataN_twoboxCol.pkl', 'rb')
        dataN_pkl = pickle.load(dataN_pkl_file)
        dataN_pkl_file.close()
        obsN = dataN_pkl['observations']
        latN = dataN_pkl['beliefs']

        para_pkl_file = open(path + '/Results/' + datestring_data + '_para_twoboxCol.pkl', 'rb')
        para_pkl = pickle.load(para_pkl_file)
        para_pkl_file.close()
        nq = para_pkl['nq']
        nl = para_pkl['nl']
        nr = para_pkl['nr']
        na = para_pkl['na']
        discount = para_pkl['discount']
        sample_length = obsN.shape[1]
        sample_number = obsN.shape[0]

        gamma1 = para_pkl['appRate1']
        gamma2 = para_pkl['appRate1']
        epsilon1 = para_pkl['disappRate1']
        epsilon2 = para_pkl['disappRate1']
        groom = para_pkl['groom']
        travelCost = para_pkl['travelCost']
        pushButtonCost = para_pkl['pushButtonCost']
        Numcol = para_pkl['ColorNumber']
        qmin = para_pkl['qmin']
        qmax = para_pkl['qmax']
        temperature = para_pkl['temperatureQ']
        parametersAgent = [gamma1, gamma2, epsilon1, epsilon2, groom, travelCost, pushButtonCost, Numcol, qmin, qmax,
                           temperature]

        gamma1_e = para_pkl['appRateExperiment1']
        gamma2_e = para_pkl['appRateExperiment2']
        epsilon1_e = para_pkl['disappRateExperiment1']
        epsilon2_e = para_pkl['disappRateExperiment2']
        qmin_e = para_pkl['qminExperiment']
        qmax_e = para_pkl['qmaxExperiment']
        parametersExp = [gamma1_e, gamma2_e, epsilon1_e, epsilon2_e, qmin_e, qmax_e]
    else:
        # parameters = [gamma1, gamma2, epsilon1, epsilon2, groom, travelCost, pushButtonCost, NumCol, qmin, qmax, temperature]
        parametersAgent = [0.2, 0.15, 0.1, 0.08, 0.2, 0.15, 0.3, 5, 0.42, 0.66, 0.1]
        parametersExp = [0.15, 0.1, 0.05, 0.04, 0.4, 0.6]
        #parametersExp_test = parametersExp

        nq = 10
        nl = 3
        nr = 2
        na = 5
        discount = 0.99
        Numcol = parametersAgent[7]  # number of colors
        Ncol = Numcol - 1  # number value: 0 top Numcol-1

        sample_length = 500
        sample_number = 200

        obsN, latN, truthN, datestring_data = twoboxColGenerate(parametersAgent, parametersExp, sample_length, sample_number,
                                                       nq, nr, nl, na, discount)

    POMDP_params = [nq, na, nr, nl, Numcol, discount, parametersAgent, parametersExp]

    if trainedNN:
        datestring_train = '08132019(0014)'

        nn_train_pkl_file = open(path + '/Results/' + datestring_train + '_data' + datestring_data + '_nn_train_params_twoboxCol.pkl', 'rb')
        nn_train_pkl = pickle.load(nn_train_pkl_file)
        nn_train_pkl_file.close()

        training_params = nn_train_pkl['training_params']
        nn_params = nn_train_pkl['nn_params']

        batch_size, train_ratio, NEpochs_bel, NEpochs_act, lr_bel, lr_act = training_params
        input_size, hidden_size_bel, output_size_bel, hidden_size_act, output_size_act, num_layers = nn_params

        checkpoint_bel = torch.load(path + '/Results/' + datestring_train + '_belNN' + '_epoch' + str(
            NEpochs_bel) + '_data' + datestring_data)
        checkpoint_act = torch.load(path + '/Results/' + datestring_train + '_actNN' + '_epoch' + str(
            NEpochs_act) + '_data' + datestring_data)

        rnn = rnn_bel(input_size, hidden_size_bel, output_size_bel, num_layers)
        rnn.load_state_dict(checkpoint_bel['belNN_state_dict'])
        rnn.eval()

        net = net_act(hidden_size_bel, hidden_size_act, output_size_act)
        net.load_state_dict(checkpoint_act['actNN_state_dict'])
        net.eval()
    else:
        """
        set parameters for training
        """
        Nf = na + nr + nl + Numcol * 2
        input_size = Nf  # na+nr+nl+Ncol1+Ncol2
        hidden_size_bel = 300  # number of neurons
        output_size_bel = 2  # belief for the box (continuous)
        hidden_size_act = 100
        output_size_act = na
        batch_size = 5
        num_layers = 1
        train_ratio = 0.9

        NEpochs_bel = 80
        NEpochs_act = 60

        lr_bel = 0.001
        lr_act = 0.001

        nn_params = [input_size, hidden_size_bel, output_size_bel, hidden_size_act, output_size_act, num_layers]
        training_params = [batch_size, train_ratio, NEpochs_bel, NEpochs_act, lr_bel, lr_act]

        datestring_train = datetime.strftime(datetime.now(), '%m%d%Y(%H%M%S)')
        rnn, net, _, _ = training(obsN, latN, POMDP_params, training_params, nn_params, datestring_data, datestring_train)


    test_N = 10
    test_T = 40000

    data_dict = agent_NNandPOMDP_NN(rnn, net, POMDP_params, nn_params, N = test_N, T = test_T)
    datestring_NNagent = datetime.strftime(datetime.now(), '%m%d%Y(%H%M%S)')
    data_output = open(path + '/Results/' + datestring_train  + '_data' + datestring_data + '_agentNNdriven' + datestring_NNagent + '_twoboxCol' + '.pkl', 'wb')
    pickle.dump(data_dict, data_output)
    data_output.close()

    data_dict1 = agent_NNandPOMDP_POMDP(rnn, net, POMDP_params, nn_params, N = test_N, T = test_T)
    datestring_NNagent1 = datetime.strftime(datetime.now(), '%m%d%Y(%H%M%S)')
    data_output1 = open(path + '/Results/' + datestring_train + '_data' + datestring_data + '_agentPOMDPdriven' + datestring_NNagent1 + '_twoboxCol' + '.pkl', 'wb')
    pickle.dump(data_dict1, data_output1)
    data_output1.close()

    data_dict2 = agent_NN(rnn, net, POMDP_params, nn_params, N = test_N, T = test_T)
    datestring_NNagent2 = datetime.strftime(datetime.now(), '%m%d%Y(%H%M%S)')
    data_output2 = open(
        path + '/Results/' + datestring_train + '_data' + datestring_data + '_agentNN' + datestring_NNagent2 + '_twoboxCol' + '.pkl',
        'wb')
    pickle.dump(data_dict2, data_output2)
    data_output2.close()

    nn_para_dict = {'nn_params': nn_params,
                 'training_params': training_params,
                 'POMDP_params': POMDP_params
                 }

    # create a file that saves the parameter dictionary using pickle
    para_output = open(path + '/Results/' + datestring_train + '_data' + datestring_data +  '_mainPara_twoboxCol' + '.pkl', 'wb')
    pickle.dump(nn_para_dict, para_output)
    para_output.close()


if __name__ == "__main__":
    main()