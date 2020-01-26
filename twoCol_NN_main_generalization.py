
from twoCol_NN_train_generalization import *
from twoCol_NN_agent_generalization import *

def main():
    existingPOMDP = True
    trainedNN = True
    generateNN = True

    if existingPOMDP:
        datestring_start = '01252020(154652)'
        datestring_end = '01252020(154909)'

        dataEnsemble_pkl_file = open(
            path + '/Results/' + datestring_start + '_' + datestring_end + '_dataEnsemble_twoboxCol' + '.pkl', 'rb')
        dataEnsemble = pickle.load(dataEnsemble_pkl_file)
        dataEnsemble_pkl_file.close()

        parametersAgent_set = dataEnsemble['parametersAgent_set']
        parameterSetFull = dataEnsemble['parameterSetFull']
        obsNFull = dataEnsemble['obsNFull']
        latNFull = dataEnsemble['latNFull']
        truthNFull = dataEnsemble['truthNFull']
        datastringSet = dataEnsemble['datastringSet']
        yMatFull = dataEnsemble['yMatFull']

        para_pkl_file = open(path + '/Results/' + datestring_start + '_para_twoboxCol.pkl', 'rb')
        para_pkl = pickle.load(para_pkl_file)
        para_pkl_file.close()
        nq = para_pkl['nq']
        nl = para_pkl['nl']
        nr = para_pkl['nr']
        na = para_pkl['na']
        discount = para_pkl['discount']
        Numcol = para_pkl['ColorNumber']
        sample_length = para_pkl['sample_length']
        sample_number = para_pkl['sample_number']
    else:
        # parameters = [gamma1, gamma2, epsilon1, epsilon2,
        #              groom, travelCost, pushButtonCost, NumCol, qmin, qmax, temperature]
        parametersAgent_set = [[0.15, 0.1, 0.1, 0.05, 0.2, 0.15, 0.3, 5, 0.3, 0.6, 0.1],
                               [0.15, 0.1, 0.1, 0.05, 0.2, 0.15, 0.3, 5, 0.45, 0.55, 0.1],
                               [0.15, 0.1, 0.1, 0.05, 0.2, 0.2, 0.4, 5, 0.45, 0.55, 0.1],
                               [0.15, 0.1, 0.1, 0.05, 0.2, 0.15, 0.3, 5, 0.3, 0.6, 0.1],
                               [0.15, 0.1, 0.1, 0.05, 0.3, 0.12, 0.3, 5, 0.48, 0.52, 0.1],
                               [0.15, 0.1, 0.1, 0.05, 0.2, 0.4, 0.5, 5, 0.45, 0.55, 0.1],
                               [0.15, 0.1, 0.1, 0.05, 0.1, 0.3, 0.1, 5, 0.45, 0.55, 0.1],
                               [0.15, 0.1, 0.1, 0.05, 0.2, 0.15, 0.3, 5, 0.48, 0.52, 0.1],
                               [0.12, 0.1, 0.08, 0.03, 0.2, 0.15, 0.3, 5, 0.45, 0.55, 0.1],
                               [0.12, 0.1, 0.08, 0.03, 0.2, 0.15, 0.3, 5, 0.2, 0.8, 0.1],
                               [0.2, 0.3, 0.15, 0.15, 0.2, 0.15, 0.3, 5, 0.45, 0.55, 0.1],
                               [0.1, 0.1, 0.05, 0.05, 0.2, 0.15, 0.3, 5, 0.45, 0.55, 0.1],
                               [0.12, 0.08, 0.1, 0.05, 0.2, 0.15, 0.3, 5, 0.45, 0.55, 0.1],
                               ]

        nq = 10
        nl = 3
        nr = 2
        na = 5
        discount = 0.99
        sample_length = 1000
        sample_number = 100
        Numcol = parametersAgent_set[0][7]  # number of colors
        Ncol = Numcol - 1  # number value: 0 top Numcol-1

        for i, parametersAgent in enumerate(parametersAgent_set):
            print("The %d -th set of parameter." % (i + 1))

            # parametersAgent = [0.15,0.1,0.1,0.05,0.2,0.15,0.3,5,0.45,0.55, 0.1]
            parametersExp = parametersAgent[0:4] + parametersAgent[-3:-1]
            # parametersExp = [0.2,0.15,0.05,0.04,0.45,0.5]

            Numcol = parametersAgent[7]

            obsN, latN, truthN, datestring = twoboxColGenerate(parametersAgent, parametersExp, sample_length,
                                                               sample_number,
                                                               nq, nr, nl, na, discount)

            if i == 0:
                parameterSetFull = np.tile(parametersAgent, (sample_number, 1))
                obsNFull = np.copy(obsN)
                latNFull = np.copy(latN)
                truthNFull = np.copy(truthN)
                datastringSet = [datestring]
            else:
                parameterSetFull = np.concatenate((parameterSetFull, np.tile(parametersAgent, (sample_number, 1))),
                                                  axis=0)
                obsNFull = np.concatenate((obsNFull, obsN), axis=0)
                latNFull = np.concatenate((latNFull, latN), axis=0)
                truthNFull = np.concatenate((truthNFull, truthN), axis=0)
                datastringSet.append(datestring)

            belief = (latN[:, 1:, 0:2] + 0.5) / nq
            actout = obsN[:, 1:, 0:1]
            act_dist = obsN[:, 1:, 5:]
            yMat = np.concatenate((belief, actout, act_dist), axis=2)  # cascade output
            if i == 0:
                yMatFull = np.copy(yMat)
            else:
                yMatFull = np.concatenate((yMatFull, yMat), axis=0)

        datestring_start = datastringSet[0]
        print('datestring_start is', datestring_start)

        datestring_end = datastringSet[-1]
        print('datestring_end is', datestring_end)
        data_ensemble_dict = {'parametersAgent_set': parametersAgent_set,
                              'parameterSetFull': parameterSetFull,
                              'obsNFull': obsNFull,
                              'latNFull': latNFull,
                              'truthNFull': truthNFull,
                              'datastringSet': datastringSet,
                              'yMatFull': yMatFull
                              }

        data_output = open(
            path + '/Results/' + datastringSet[0] + '_' + datastringSet[-1] + '_dataEnsemble_twoboxCol' + '.pkl', 'wb')
        pickle.dump(data_ensemble_dict, data_output)
        data_output.close()

    POMDP_params = [nq, na, nr, nl, Numcol, discount, parametersAgent_set]

    if trainedNN:
        datestring_train =  '01252020(154926)'

        nn_train_pkl_file = open(path + '/Results/' + datestring_train + '_data' +
                                 datestring_start + '_nn_train_params_twoboxCol_generalization.pkl', 'rb')
        nn_train_pkl = pickle.load(nn_train_pkl_file)
        nn_train_pkl_file.close()

        training_params = nn_train_pkl['training_params']
        nn_params = nn_train_pkl['nn_params']
        train_loss = nn_train_pkl['train_loss']


        input_size, hidden_size1, hidden_size2, output_size_act, num_layers = nn_params
        batch_size, train_ratio, NEpochs, lr = training_params

        checkpoint = torch.load(path + '/Results/' + datestring_train + '_rnn' + '_epoch' + str(
            NEpochs) + '_dataEnsemble' + datestring_start)
        rnn = RNN(input_size, hidden_size1, hidden_size2, output_size_act, num_layers, len(parametersAgent_set[0]))
        rnn.load_state_dict(checkpoint['model_state_dict'])
        rnn.eval()
    else:
        """
        set parameters for NN & training
        """
        Nf = na + nr + nl + Numcol + Numcol
        input_size = Nf  # nna+nr+nl+Numcol+Numcol
        hidden_size1 = 100  # number of neurons
        hidden_size2 = 50
        output_size_act = na
        num_layers = 1

        batch_size = 5
        train_ratio = 0.9
        NEpochs = 50
        lr = 0.0003

        nn_params = [input_size, hidden_size1, hidden_size2, output_size_act, num_layers]
        training_params = [batch_size, train_ratio, NEpochs, lr]

        act_onehot = one_hot_encode(obsNFull[:, 0:-1, 0].astype(int), na)
        rew_onehot = one_hot_encode(obsNFull[:, 1:, 1].astype(int), nr)
        loc_onehot = one_hot_encode(obsNFull[:, 1:, 2].astype(int), nl)
        col1_onehot = one_hot_encode(obsNFull[:, 1:, 3].astype(int), Numcol)
        col2_onehot = one_hot_encode(obsNFull[:, 1:, 4].astype(int), Numcol)
        xMatFull = np.concatenate((act_onehot, rew_onehot, loc_onehot, col1_onehot, col2_onehot),
                                  axis=2)  # cascade all the input

        datestring_train = datetime.strftime(datetime.now(), '%m%d%Y(%H%M%S)')
        print('datestring_train is:', datestring_train)
        rnn, test_loader, _, train_loss= training_generalization(parameterSetFull, xMatFull, yMatFull, POMDP_params, training_params, nn_params,
                                          datestring_start, datestring_train)

        nn_dict = {'nn_params': nn_params,
                   'training_params': training_params,
                   'train_loss': train_loss

        }
        data_output = open(path + '/Results/' + datestring_train + '_data' +
                                 datestring_start + '_nn_train_params_twoboxCol_generalization.pkl', 'wb')
        pickle.dump(nn_dict, data_output)
        data_output.close()

    """
    generate NN and POMDP behavior data with observation from NN, compare policy 
    """
    if generateNN:
        test_N = 5 #1
        test_T = 20000
        # parametersAgent_test = [0.15, 0.1, 0.1, 0.05, 0.2, 0.15, 0.3, 5, 0.48, 0.53, 0.1]
        # parametersExp_test = [0.2, 0.15, 0.05, 0.04, 0.45, 0.5]

        parametersAgent_test = [0.2, 0.18, 0.1, 0.08, 0.2, 0.15, 0.3, 5, 0.45, 0.55, 0.1]
        parametersExp_test = [0.15, 0.1, 0.05, 0.04, 0.4, 0.6]

        #parametersExp_test = parametersAgent[0:4] + parametersAgent[-3:-1]


        NNtest_params = [nq, na, nr, nl, Numcol, discount, parametersAgent_test, parametersExp_test]
        """
        discrepancy, if we increase the parameter difference in the same way 
        """

        data_dict = agent_NNandPOMDP_NN(rnn, NNtest_params, nn_params, N=test_N, T=test_T)

        # para = np.array(parametersAgent_test)
        # obs_IRC = data_dict['observations'][0, :5000, :5].astype(int)  # NN agent behavior
        # twoboxCol = twoboxColMDP(discount, 10, nr, na, nl, para)
        # twoboxCol.setupMDP()
        # twoboxCol.solveMDP_sfm()
        # ThA = twoboxCol.ThA
        # policy = twoboxCol.softpolicy
        # pi = np.ones(nq * nq) / nq / nq  # initialize the estimation of the belief state
        # Trans_hybrid_obs12 = twoboxCol.Trans_hybrid_obs12
        # Obs_emis_trans1 = twoboxCol.Obs_emis_trans1
        # Obs_emis_trans2 = twoboxCol.Obs_emis_trans2
        # twoboxColHMM = HMMtwoboxCol(ThA, policy, Trans_hybrid_obs12, Obs_emis_trans1, Obs_emis_trans2, pi, 4)
        #
        # Qaux2= twoboxColHMM.computeQaux(obs_IRC, ThA, policy, Trans_hybrid_obs12, Obs_emis_trans1,
        #                                        Obs_emis_trans2)
        # Qaux3 = twoboxColHMM.latent_entr(obs_IRC)
        # print(Qaux2 + Qaux3)


        datestring_NNagent = datetime.strftime(datetime.now(), '%m%d%Y(%H%M%S)')
        data_output = open(path + '/Results/' + datestring_train  + '_data' + datestring_start + '_agentNNdriven' + datestring_NNagent + '_twoboxCol' + '.pkl', 'wb')
        pickle.dump(data_dict, data_output)
        data_output.close()

        # data_dict1 = agent_NNandPOMDP_POMDP(rnn, NNtest_params, nn_params, N=test_N, T=test_T)
        # datestring_NNagent1 = datetime.strftime(datetime.now(), '%m%d%Y(%H%M%S)')
        # data_output1 = open(path + '/Results/' + datestring_train + '_data' + datestring_start + '_agentPOMDPdriven' + datestring_NNagent1 + '_twoboxCol' + '.pkl', 'wb')
        # pickle.dump(data_dict1, data_output1)
        # data_output1.close()

        # data_dict2 = agent_NN(rnn, NNtest_params, nn_params, N=test_N, T=test_T)
        # datestring_NNagent2 = datetime.strftime(datetime.now(), '%m%d%Y(%H%M%S)')
        # data_output2 = open(
        #     path + '/Results/' + datestring_train + '_data' + datestring_start + '_agentNN' + datestring_NNagent2 + '_twoboxCol' + '.pkl',
        #     'wb')
        # pickle.dump(data_dict2, data_output2)
        # data_output2.close()


        nn_para_dict = {'nn_params': nn_params,
                        'training_params': training_params,
                        'NNtest_params': NNtest_params
                        }

        # create a file that saves the parameter dictionary using pickle
        para_output = open(
            path + '/Results/' + datestring_train + '_data' + datestring_start + '_agent' + datestring_NNagent +'_mainPara_twoboxCol' + '.pkl', 'wb')
        pickle.dump(nn_para_dict, para_output)
        para_output.close()

    print(1)

if __name__ == "__main__":
    main()