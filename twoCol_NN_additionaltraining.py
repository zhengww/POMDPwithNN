import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

path = os.getcwd()
#datestring_additional_training = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')
datestring_NNagent = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')


from twoboxCol import *
from twoCol_NN_params import *
from twoCol_NN_model import *
from twoCol_NN_data import *

bel_model = rnn(input_size, hidden_size_bel, output_size_bel, num_layers)
bel_model.load_state_dict(torch.load(
    path + '/Results/' + datestring_train + '_belNN' + '_epoch' + str(NEpochs_bel) + '_data' + datestring_data)[
                              'belNN_state_dict'])
bel_model.eval()

act_model = net(hidden_size_bel, hidden_size_act, output_size_act)
act_model.load_state_dict(torch.load(
    path + '/Results/' + datestring_train + '_actNN' + '_epoch' + str(NEpochs_act) + '_data' + datestring_data)[
                              'actNN_state_dict'])
act_model.eval()


beta = 0  # available food dropped back into box after button press
delta = 0  # animal trips, doesn't go to target location
direct = 0  # animal goes right to target, skipping location 0
rho = 1  # food in mouth is consumed

parametersExp_test = parametersExp

gamma1_e_test = parametersExp_test[0]
gamma2_e_test = parametersExp_test[1]
epsilon1_e_test = parametersExp_test[2]
epsilon2_e_test = parametersExp_test[3]
qmin_e_test = parametersExp_test[4]
qmax_e_test = parametersExp_test[5]

N = 1  # sample_number
T = 5000 #sample_length

action = np.empty((N, T), dtype=int)
location = np.empty((N, T), dtype=int)
belief1 = np.empty((N, T))
belief2 = np.empty((N, T))
reward = np.empty((N, T), dtype=int)
trueState1 = np.empty((N, T), dtype=int)
trueState2 = np.empty((N, T), dtype=int)
color1 = np.empty((N, T), dtype=int)
color2 = np.empty((N, T), dtype=int)
neural_response = np.empty((N, T, hidden_size_bel))
actionDist = np.zeros((N, T, na))

twoboxColdata = twoboxColMDPdata(discount, nq, nr, na, nl, parametersAgent, parametersExp, T, N)
softpolicy = twoboxColdata.softpolicy
den1 = twoboxColdata.den1
den2 = twoboxColdata.den2
belief1_POMDP = np.empty((N, T), int)
belief2_POMDP = np.empty((N, T), int)
action_POMDP = np.empty((N, T), dtype=int)
hybrid_POMDP = np.empty((N, T), int)
actionDist_POMDP = np.zeros((N, T, na))
belief1Dist_POMDP = np.zeros((N, T, nq))
belief2Dist_POMDP = np.zeros((N, T, nq))


for n in range(N):
    actionInitial = 0  # at time t = -1
    belief1Initial_POMDP = np.random.randint(nq)
    rewInitial = np.random.randint(nr)
    belief2Initial_POMDP = np.random.randint(nq)
    locationInitial = np.random.randint(nl)

    for t in range(T):
        if t == 0:
            trueState1[n, t] = np.random.binomial(1, gamma1_e_test)
            trueState2[n, t] = np.random.binomial(1, gamma2_e_test)
            q1 = trueState1[n, t] * qmin_e_test + (1 - trueState1[n, t]) * qmax_e_test
            color1[n, t] = np.random.binomial(Ncol, q1)  # color for box 1
            q2 = trueState2[n, t] * qmin_e_test + (1 - trueState2[n, t]) * qmax_e_test
            color2[n, t] = np.random.binomial(Ncol, q2)  # color for box 2

            location[n, t], reward[n, t] = locationInitial, rewInitial

            input_belNN = np.concatenate(
                (one_hot_encode(np.array([[actionInitial]]), na, 1, 1),
                 one_hot_encode(np.array([[reward[n, t]]]), nr, 1, 1),
                 one_hot_encode(np.array([[location[n, t]]]), nl, 1, 1),
                 one_hot_encode(np.array([[color1[n, t]]]), Numcol, 1, 1),
                 one_hot_encode(np.array([[color2[n, t]]]), Numcol, 1, 1)), axis=2)  # cascade all the input
            input_belNN = torch.tensor(input_belNN, dtype=torch.float)

            with torch.no_grad():
                out_bel_batch, hidden_batch = bel_model(input_belNN)
                out_act_batch = act_model(hidden_batch)  # policy

                act_predicted = np.argmax(np.random.multinomial(1, out_act_batch[0, 0, :]))

            belief1[n, t] = out_bel_batch[:, :, 0]
            belief2[n, t] = out_bel_batch[:, :, 1]
            neural_response[n, t] = hidden_batch
            action[n, t] = act_predicted

            belief1_POMDP[n, t], belief2_POMDP[n, t] = belief1Initial_POMDP, belief2Initial_POMDP
            hybrid_POMDP[n, t] = location[n, t] * (nq * nr * nq) + belief1_POMDP[n, t] * (
                    nr * nq) + reward[n, t] * nq + belief2_POMDP[n, t]  # hybrid state, for policy choosing
            action_POMDP[n, t] = np.argmax(np.random.multinomial
                                           (1, softpolicy.T[hybrid_POMDP[n, t]]))
            actionDist_POMDP[n, t] = softpolicy.T[hybrid_POMDP[n, t]]
            belief1Dist_POMDP[n, t, belief1_POMDP[n, t]] = 1
            belief2Dist_POMDP[n, t, belief2_POMDP[n, t]] = 1

            
        else:
            if action[n, t - 1] == pb and location[n, t - 1] == 0:
                action[n, t - 1] = a0

            # variables evolve with dynamics
            if action[n, t - 1] != pb:
                # button not pressed, then true world dynamic is not affected by actions
                if trueState1[n, t - 1] == 0:
                    trueState1[n, t] = np.random.binomial(1, gamma1_e_test)
                else:
                    trueState1[n, t] = 1 - np.random.binomial(1, epsilon1_e_test)

                if trueState2[n, t - 1] == 0:
                    trueState2[n, t] = np.random.binomial(1, gamma2_e_test)
                else:
                    trueState2[n, t] = 1 - np.random.binomial(1, epsilon2_e_test)

                q1 = trueState1[n, t] * qmin_e_test + (1 - trueState1[n, t]) * qmax_e_test
                color1[n, t] = np.random.binomial(Ncol, q1)  # color for box 1
                q2 = trueState2[n, t] * qmin_e_test + (1 - trueState2[n, t]) * qmax_e_test
                color2[n, t] = np.random.binomial(Ncol, q2)  # color for box 2

                belief1Dist_POMDP[n, t] = den1[color1[n, t], :, int(find_closest((np.arange(nq) + 0.5) / nq, belief1[n, t-1]) * nq -0.5)]
                belief1_POMDP[n, t] = np.argmax(
                    np.random.multinomial(1, den1[color1[n, t], :, int(find_closest((np.arange(nq) + 0.5) / nq, belief1[n, t-1]) * nq -0.5)], size=1))
                belief1Dist_POMDP[n, t] = den2[color2[n, t], :, int(find_closest((np.arange(nq) + 0.5) / nq, belief2[n, t-1]) * nq -0.5)]
                belief2_POMDP[n, t] = np.argmax(
                    np.random.multinomial(1, den2[color2[n, t], :, int(find_closest((np.arange(nq) + 0.5) / nq, belief2[n, t-1]) * nq -0.5)], size=1))


                if reward[n, t - 1] == 0:
                    reward[n, t] = 0
                else:
                    reward[n, t] = np.random.binomial(1, 1 - rho)

                if action[n, t - 1] == a0:
                    location[n, t] = location[n, t - 1]
                if action[n, t - 1] == g0:
                    Tl0 = np.array(
                        [[1, 1 - delta, 1 - delta], [0, delta, 0],
                         [0, 0, delta]])  # go to loc 0 (with error of delta)
                    location[n, t] = np.argmax(np.random.multinomial(1, Tl0[:, location[n, t - 1]], size=1))
                if action[n, t - 1] == g1:
                    Tl1 = np.array([[delta, 0, 1 - delta - direct], [1 - delta, 1, direct],
                                    [0, 0, delta]])  # go to box 1 (with error of delta)
                    location[n, t] = np.argmax(np.random.multinomial(1, Tl1[:, location[n, t - 1]], size=1))
                if action[n, t - 1] == g2:
                    Tl2 = np.array([[delta, 1 - delta - direct, 0], [0, delta, 0],
                                    [1 - delta, direct, 1]])  # go to box 2 (with error of delta)
                    location[n, t] = np.argmax(np.random.multinomial(1, Tl2[:, location[n, t - 1]], size=1))

            if action[n, t - 1] == pb:  # press button
                location[n, t] = location[n, t - 1]  # pressing button does not change location

                ### for pb action, wait for usual time and then pb  #############
                if trueState1[n, t - 1] == 0:
                    trueState1[n, t - 1] = np.random.binomial(1, gamma1_e_test)
                else:
                    trueState1[n, t - 1] = 1 - np.random.binomial(1, epsilon1_e_test)

                if trueState2[n, t - 1] == 0:
                    trueState2[n, t - 1] = np.random.binomial(1, gamma2_e_test)
                else:
                    trueState2[n, t - 1] = 1 - np.random.binomial(1, epsilon2_e_test)
                ### for pb action, wait for usual time and then pb  #############

                if location[n, t] == 1:  # consider location 1 case

                    # belief on box 2 is independent on box 1
                    if trueState2[n, t - 1] == 0:
                        trueState2[n, t] = np.random.binomial(1, gamma2_e_test)
                    else:
                        trueState2[n, t] = 1 - np.random.binomial(1, epsilon2_e_test)
                    q2 = trueState2[n, t] * qmin_e_test + (1 - trueState2[n, t]) * qmax_e_test
                    color2[n, t] = np.random.binomial(Ncol, q2)  # color for box 2


                    belief2_POMDP[n, t] = np.argmax(
                        np.random.multinomial(1, den2[color2[n, t], :, int(find_closest((np.arange(nq) + 0.5) / nq, belief2[n, t-1]) * nq -0.5)],
                                              size=1))

                    belief1_POMDP[n, t] = 0

                    belief1Dist_POMDP[n, t, belief1_POMDP[n, t]] = 1
                    belief2Dist_POMDP[n, t] = den2[color2[n, t], :, int(find_closest((np.arange(nq) + 0.5) / nq, belief2[n, t-1]) * nq -0.5)]

                    if trueState1[n, t - 1] == 0:
                        trueState1[n, t] = 0
                        color1[n, t] = Ncol

                        if reward[n, t - 1] == 0:  # reward depends on previous time frame
                            reward[n, t] = 0
                        else:
                            reward[n, t] = np.random.binomial(1, 1 - rho)  # have not consumed food
                    else:
                        trueState1[n, t] = 0  # if true world is one, pb resets it to zero
                        color1[n, t] = Ncol

                        reward[n, t] = 1  # give some reward

                if location[n, t] == 2:  # consider location 2 case

                    # belief on box 1 is independent on box 2
                    if trueState1[n, t - 1] == 0:
                        trueState1[n, t] = np.random.binomial(1, gamma1_e_test)
                    else:
                        trueState1[n, t] = 1 - np.random.binomial(1, epsilon1_e_test)
                    q1 = trueState1[n, t] * qmin_e_test + (1 - trueState1[n, t]) * qmax_e_test
                    color1[n, t] = np.random.binomial(Ncol, q1)  # color for box 2

                    belief1_POMDP[n, t] = np.argmax(
                        np.random.multinomial(1, den1[color1[n, t], :, int(find_closest((np.arange(nq) + 0.5) / nq, belief1[n, t-1]) * nq -0.5)],
                                              size=1))
                    belief2_POMDP[n, t] = 0

                    belief2Dist_POMDP[n, t, belief2_POMDP[n, t]] = 1
                    belief1Dist_POMDP[n, t] = den1[color1[n, t], :, int(
                        find_closest((np.arange(nq) + 0.5) / nq, belief1[n, t - 1]) * nq - 0.5)]

                    if trueState2[n, t - 1] == 0:
                        trueState2[n, t] = trueState2[n, t - 1]
                        color2[n, t] = Ncol
                        # if true world is zero, pb does not change real state
                        # assume that the real state does not change during button press

                        if reward[n, t - 1] == 0:  # reward depends on previous time frame
                            reward[n, t] = 0
                        else:
                            reward[n, t] = np.random.binomial(1, 1 - rho)  # have not consumed food
                    else:
                        trueState2[n, t] = 0  # if true world is one, pb resets it to zero
                        color2[n, t] = Ncol

                        reward[n, t] = 1  # give some reward

            input_belNN = np.concatenate(
                (one_hot_encode(np.array([[action[n, t - 1]]]), na, 1, 1),
                 one_hot_encode(np.array([[reward[n, t]]]), nr, 1, 1),
                 one_hot_encode(np.array([[location[n, t]]]), nl, 1, 1),
                 one_hot_encode(np.array([[color1[n, t]]]), Numcol, 1, 1),
                 one_hot_encode(np.array([[color2[n, t]]]), Numcol, 1, 1)), axis=2)  # cascade all the input
            input_belNN = torch.tensor(input_belNN, dtype=torch.float)

            with torch.no_grad():
                out_bel_batch, hidden_batch = bel_model(input_belNN)
                out_act_batch = act_model(hidden_batch)

            act_predicted = np.argmax(np.random.multinomial(1, out_act_batch[0, 0, :]))

            belief1[n, t] = out_bel_batch[:, :, 0]
            belief2[n, t] = out_bel_batch[:, :, 1]
            neural_response[n, t] = hidden_batch
            action[n, t] = act_predicted

            hybrid_POMDP[n, t] = location[n, t] * (nq * nr * nq) + belief1_POMDP[n, t] * (
                    nr * nq) + reward[n, t] * nq + belief2_POMDP[n, t]  # hybrid state, for policy choosing
            actionDist_POMDP[n, t] = softpolicy.T[hybrid_POMDP[n, t]]
            action_POMDP[n, t] = np.argmax(np.random.multinomial
                                           (1, softpolicy.T[hybrid_POMDP[n, t]]))

obsN = np.dstack([action, reward, location, color1, color2])  # includes the action and the observable states
latN = np.dstack([belief1, belief2])
truthN = np.dstack([trueState1, trueState2])
neuralNN = np.dstack([neural_response])
dataN = np.dstack([obsN, latN, neuralNN, truthN])
dataN_NN = np.dstack([action_POMDP, belief1_POMDP, belief2_POMDP])
dataN_NN_dist = np.dstack([actionDist_POMDP, belief1Dist_POMDP, belief2Dist_POMDP])
### write data to file
data_dict = {'observations': obsN,
             'beliefs': latN,
             'trueStates': truthN,
             'neural_response': neuralNN,
             'allData': dataN,
             'POMDP_agent': dataN_NN,
             'POMDP_agent_dist': dataN_NN_dist}
# data_output = open(
#     path + '/Results/' + datestring_train + '_Atrain' + datestring_additional_training + '_data' + datestring_data + '_agent'  + '_twoboxCol' + '.pkl',
#     'wb')

data_output = open(
    path + '/Results/' + datestring_train + '_data' + datestring_data + '_agent' + datestring_NNagent + '_twoboxCol' + '.pkl',
    'wb')

pickle.dump(data_dict, data_output)
data_output.close()





