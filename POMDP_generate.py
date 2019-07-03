from twoboxCol import *
from onebox import *
from datetime import datetime
import os
import pickle

path = os.getcwd()

def twoboxColGenerate(parameters, parametersExp, sample_length, sample_number, nq, nr = 2, nl = 3, na = 5,
                      discount = 0.99, belief1Initial=0, rewInitial=0, belief2Initial=0, locationInitial=0):

    datestring = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')  # current time used to set file name

    print("\nSet the parameters of the model... \n")

    beta = 0  # available food dropped back into box after button press
    gamma1 = parameters[0]  # reward becomes available in box 1
    gamma2 = parameters[1]  # reward becomes available in box 2
    delta = 0  # animal trips, doesn't go to target location
    direct = 0  # animal goes right to target, skipping location 0
    epsilon1 = parameters[2]  # available food disappears from box 1
    epsilon2 = parameters[3]  # available food disappears from box 2
    rho = 1  # food in mouth is consumed
    # State rewards
    Reward = 1  # reward per time step with food in mouth
    groom = parameters[4]  # location 0 reward
    # Action costs
    travelCost = parameters[5]
    pushButtonCost = parameters[6]

    NumCol = np.rint(parameters[7]).astype(int)  # number of colors
    Ncol = NumCol - 1  # max value of color
    qmin = parameters[8]
    qmax = parameters[9]

    gamma1_e = parametersExp[0]
    gamma2_e = parametersExp[1]
    epsilon1_e = parametersExp[2]
    epsilon2_e = parametersExp[3]
    qmin_e = parametersExp[4]
    qmax_e = parametersExp[5]

    # parameters = [gamma1, gamma2, epsilon1, epsilon2,
    #              groom, travelCost, pushButtonCost, NumCol, qmin, qmax]

    ### Gnerate data"""
    print("Generating data...")
    T = sample_length
    N = sample_number
    twoboxColdata = twoboxColMDPdata(discount, nq, nr, na, nl, parameters, parametersExp, T, N)
    twoboxColdata.dataGenerate_sfm(belief1Initial, rewInitial, belief2Initial, locationInitial)

    hybrid = twoboxColdata.hybrid
    action = twoboxColdata.action
    location = twoboxColdata.location
    belief1 = twoboxColdata.belief1
    belief2 = twoboxColdata.belief2
    reward = twoboxColdata.reward
    trueState1 = twoboxColdata.trueState1
    trueState2 = twoboxColdata.trueState2
    color1 = twoboxColdata.color1
    color2 = twoboxColdata.color2

    # sampleNum * sampleTime * dim of observations(=3 here, action, reward, location)
    # organize data
    obsN = np.dstack([action, reward, location, color1, color2])  # includes the action and the observable states
    latN = np.dstack([belief1, belief2])
    truthN = np.dstack([trueState1, trueState2])
    dataN = np.dstack([obsN, latN, truthN])

    ### write data to file
    data_dict = {'observations': obsN,
                 'beliefs': latN,
                 'trueStates': truthN,
                 'allData': dataN}
    data_output = open(path + '/Results/' + datestring + '_dataN_twoboxCol' + '.pkl', 'wb')
    pickle.dump(data_dict, data_output)
    data_output.close()

    ### write all model parameters to file
    para_dict = {'discount': discount,
                 'nq': nq,
                 'nr': nr,
                 'nl': nl,
                 'na': na,
                 'foodDrop': beta,
                 'appRate1': gamma1,
                 'appRate2': gamma2,
                 'disappRate1': epsilon1,
                 'disappRate2': epsilon2,
                 'consume': rho,
                 'reward': Reward,
                 'groom': groom,
                 'travelCost': travelCost,
                 'pushButtonCost': pushButtonCost,
                 'ColorNumber': NumCol,
                 'qmin': qmin,
                 'qmax': qmax,
                 'appRateExperiment1': gamma1_e,
                 'disappRateExperiment1': epsilon1_e,
                 'appRateExperiment2': gamma2_e,
                 'disappRateExperiment2': epsilon2_e,
                 'qminExperiment': qmin_e,
                 'qmaxExperiment': qmax_e
                 }

    # create a file that saves the parameter dictionary using pickle
    para_output = open(path + '/Results/' + datestring + '_para_twoboxCol' + '.pkl', 'wb')
    pickle.dump(para_dict, para_output)
    para_output.close()

    print('Data stored in files')

    return obsN, latN, truthN, datestring

def oneboxGenerate(parameters, parametersExp, sample_length, sample_number, nq, nr = 2, na = 2,
                   discount = 0.99, beliefInitial = 0, rewInitial = 0):
    #datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    datestring = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')   # current time used to set file name

    beta = parameters[0]     # available food dropped back into box after button press
    gamma = parameters[1]   # reward becomes available
    epsilon = parameters[2]   # available food disappears
    rho = parameters[3]   # .99      # food in mouth is consumed
    pushButtonCost = parameters[4]
    Reward = 1

    gamma_e = parametersExp[0]
    epsilon_e = parametersExp[1]
    #parameters = [beta, gamma, epsilon, rho, pushButtonCost]

    ### Gnerate data"""
    print("\nGenerating data...")
    T = sample_length
    N = sample_number
    oneboxdata = oneboxMDPdata(discount, nq, nr, na, parameters, parametersExp, T, N)
    oneboxdata.dataGenerate_sfm(beliefInitial, rewInitial)  # softmax policy

    belief = oneboxdata.belief
    action = oneboxdata.action
    reward = oneboxdata.reward
    trueState = oneboxdata.trueState

    # sampleNum * sampleTime * dim of observations(=3 here, action, reward, location)
    # organize data
    obsN = np.dstack([action, reward])  # includes the action and the observable states
    latN = np.dstack([belief])
    truthN = np.dstack([trueState])
    dataN = np.dstack([obsN, latN, truthN])

    ### write data to file
    data_dict = {'observations': obsN,
                 'beliefs': latN,
                 'trueStates': truthN,
                 'allData': dataN}
    data_output = open(path + '/Results/' + datestring + '_dataN_onebox' + '.pkl', 'wb')
    pickle.dump(data_dict, data_output)
    data_output.close()

    ### write all model parameters to file
    para_dict = {'discount': discount,
                 'nq': nq,
                 'nr': nr,
                 'na': na,
                 'foodDrop': beta,
                 'appRate': gamma,
                 'disappRate': epsilon,
                 'consume': rho,
                 'reward': Reward,
                 'pushButtonCost': pushButtonCost,
                 'appRateExperiment': gamma_e,
                 'disappRateExperiment': epsilon_e
                 }

    # create a file that saves the parameter dictionary using pickle
    para_output = open(path + '/Results/' + datestring + '_para_onebox' + '.pkl', 'wb')
    pickle.dump(para_dict, para_output)
    para_output.close()

    print('Data stored in files')

    return obsN, latN, truthN, datestring