'''
This incorporates the oneboxtask_ini and oneboxMDPsolver and oneboxGenerate into one file with oneboxMDP object

'''

from __future__ import division
from boxtask_func import *
from MDPclass import *


# we need two different transition matrices, one for each of the following actions:
a0 = 0  # a0 = do nothing
pb = 1  # pb  = push button
sigmaTb = 0.1    # variance for the gaussian approximation in belief transition matrix
temperatureQ = 0.061  # temperature for soft policy based on Q value


class oneboxMDP:
    """
    model onebox problem, set up the transition matrices and reward based on the given parameters,
    and solve the MDP problem, return the optimal policy
    """
    def __init__(self, discount, nq, nr, na, parameters):
        self.discount = discount
        self.nq = nq
        self.nr = nr
        self.na = na
        self.n = self.nq * self.nr  # total number of states
        self.parameters = parameters  # [beta, gamma, epsilon, rho]
        self.ThA = np.zeros((self.na, self.n, self.n))
        self.R = np.zeros((self.na, self.n, self.n))

    def setupMDP(self):
        """
        Based on the parameters, create transition matrices and reward function.
        Implement the codes in file 'oneboxtask_ini.py'
        :return:
                ThA: transition probability,
                     shape: (# of action) * (# of states, old state) * (# of states, new state)
                R: reward function
                   shape: (# of action) * (# of states, old state) * (# of states, new state)
        """

        beta = self.parameters[0]   # available food dropped back into box after button press
        gamma = self.parameters[1]    # reward becomes available
        epsilon = self.parameters[2]   # available food disappears
        rho = self.parameters[3]    # food in mouth is consumed
        pushButtonCost = self.parameters[4]
        Reward = 1

        # initialize probability distribution over states (belief and world)
        pr0 = np.array([1, 0])  # (r=0, r=1) initially no food in mouth p(R=0)=1.
        pb0 = np.insert(np.zeros(self.nq - 1), 0, 1)  # initial belief states (here, lowest availability)

        ph0 = kronn(pr0, pb0)
        # kronecker product of these initial distributions
        # Note that this ordering makes the subsequent products easiest

        # setup single-variable transition matrices
        Tr = np.array([[1, rho], [0, 1 - rho]])  # consume reward
        # Tb = beliefTransitionMatrix(gamma, epsilon, nq, eta)
        # belief transition matrix

        Tb = beliefTransitionMatrixGaussian(gamma, epsilon, self.nq, sigmaTb)
        # softened the belief transition matrix with 2-dimensional Gaussian distribution

        # ACTION: do nothing
        self.ThA[a0, :, :] = kronn(Tr, Tb)
        # kronecker product of these transition matrices

        # ACTION: push button
        bL = (np.array(range(self.nq)) + 1 / 2) / self.nq

        Trb = np.concatenate((np.array([np.insert(np.zeros(self.nq), 0, 1 - bL)]),
                              np.zeros((self.nq - 2, 2 * self.nq)),
                              np.array([np.insert([np.zeros(self.nq)], 0, beta * bL)]),
                              np.array([np.insert([(1 - beta) * bL], self.nq, 1 - bL)]),
                              np.zeros(((self.nq - 2), 2 * self.nq)),
                              np.array([np.insert([np.zeros(self.nq)], self.nq, bL)])), axis=0)
        self.ThA[pb, :, :] = Trb.dot(self.ThA[a0, :, :])
        #self.ThA[pb, :, :] = Trb

        Reward_h = tensorsumm(np.array([[0, Reward]]), np.zeros((1, self.nq)))
        Reward_a = - np.array([0, pushButtonCost])

        [R1, R2, R3] = np.meshgrid(Reward_a.T, Reward_h, Reward_h, indexing='ij')
        Reward = R1 + R3
        self.R = Reward

        for i in range(self.na):
            self.ThA[i, :, :] = self.ThA[i, :, :].T

    def solveMDP_op(self, epsilon = 10**-6, niterations = 10000):
        """
        Solve the MDP problem with value iteration
        Implement the codes in file 'oneboxMDPsolver.py'

        :param discount: temporal discount
        :param epsilon: stopping criterion used in value iteration
        :param niterations: value iteration
        :return:
                Q: Q value function
                   shape: (# of actions) * (# of states)
                policy: the optimal policy based on the maximum Q value
                        shape: # of states, take integer values indicating the action
                softpolicy: probability of choosing each action
                            shape: (# of actions) * (# of states)
        """

        vi = ValueIteration_opZW(self.ThA, self.R, self.discount, epsilon, niterations)
        vi.run()
        self.Q = self._QfromV(vi)   # shape na * number of state, use value to calculate Q value
        self.policy = np.array(vi.policy)

        #pi = mdp.ValueIteration(self.ThA, self.R, self.discount, epsilon, niterations)
        #pi.run()
        #self.Q = self._QfromV(pi)
        #self.policy = np.array(pi.policy)


    def solveMDP_sfm(self, epsilon = 10**-6, niterations = 10000, initial_value=0):
        """
        Solve the MDP problem with value iteration
        Implement the codes in file 'oneboxMDPsolver.py'

        :param discount: temporal discount
        :param epsilon: stopping criterion used in value iteration
        :param niterations: value iteration
        :return:
                Q: Q value function
                   shape: (# of actions) * (# of states)
                policy: softmax policy
        """

        vi = ValueIteration_sfmZW(self.ThA, self.R, self.discount, epsilon, niterations, initial_value)
        vi.run(temperatureQ)
        self.Qsfm = self._QfromV(vi)   # shape na * number of state, use value to calculate Q value
        self.softpolicy = np.array(vi.softpolicy)
        #print self.Qsfm

        return  vi.V


    def _QfromV(self, ValueIteration):
        Q = np.zeros((ValueIteration.A, ValueIteration.S)) # Q is of shape: na * n
        for a in range(ValueIteration.A):
            Q[a, :] = ValueIteration.R[a] + ValueIteration.discount * \
                                            ValueIteration.P[a].dot(ValueIteration.V)
        return Q


class oneboxMDPdata(oneboxMDP):
    """
    This class generates the data based on the object oneboxMDP. The parameters, and thus the transition matrices and
    the rewrd function, are shared for the oneboxMDP and this data generator class.
    """
    def __init__(self, discount, nq, nr, na, parameters, parametersExp,
                 sampleTime, sampleNum):
        oneboxMDP.__init__(self, discount, nq, nr, na, parameters)

        self.parametersExp = parametersExp
        self.sampleNum = sampleNum
        self.sampleTime = sampleTime

        self.action = np.empty((self.sampleNum, self.sampleTime), int)  # initialize action
        self.hybrid = np.empty((self.sampleNum, self.sampleTime), int)  # initialize hybrid state.
        # Here it is the joint state of reward and belief
        self.belief = np.empty((self.sampleNum, self.sampleTime), int)  # initialize hidden state, belief state
        self.reward = np.empty((self.sampleNum, self.sampleTime), int)  # initialize reward state
        self.trueState = np.zeros((self.sampleNum, self.sampleTime))

        self.actionDist = np.empty((self.sampleNum, self.sampleTime, self.na), int)
        self.beliefDist = np.empty((self.sampleNum, self.sampleTime, self.nq), int)

        self.setupMDP()
        self.solveMDP_op()
        self.solveMDP_sfm()

    def dataGenerate_op(self, beliefInitial, rewInitial):
        """
        This is a function that belongs to the oneboxMDP class. In the oneboxGenerate.py file, this function is implemented
        as a separate class, since at that time, the oneboxMDP class was not defined.
        In this file, all the functions are implemented under a single class.

        :return: the obseravations
        """

        beta = self.parameters[0]  # available food dropped back into box after button press
        gamma = self.parameters[1]  # reward becomes available
        epsilon = self.parameters[2]  # available food disappears
        rho = self.parameters[3]  # food in mouth is consumed

        gamma_e = self.parametersExp[0]
        epsilon_e = self.parametersExp[1]


        for i in range(self.sampleNum):
            for t in range(self.sampleTime):
                if t == 0:
                    self.trueState[i, t] = np.random.binomial(1, gamma_e)

                    self.reward[i, t], self.belief[i, t] = rewInitial, beliefInitial
                    self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]    # This is for one box only
                    self.action[i, t] = self.policy[self.hybrid[i, t]]
                            # action is based on optimal policy
                else:
                    if self.action[i, t-1] != pb:
                        stattemp = np.random.multinomial(1, self.ThA[self.action[i, t - 1], self.hybrid[i, t - 1], :], size = 1)
                        self.hybrid[i, t] = np.argmax(stattemp)
                        self.reward[i, t], self.belief[i, t] = divmod(self.hybrid[i, t], self.nq)
                        self.action[i, t] = self.policy[self.hybrid[i, t]]

                        if self.trueState[i, t - 1] == 0:
                            self.trueState[i, t] = np.random.binomial(1, gamma_e)
                        else:
                            self.trueState[i, t] = 1 - np.random.binomial(1, epsilon_e)
                    else:
                        #### for pb action, wait for usual time and then pb  #############
                        if self.trueState[i, t - 1] == 0:
                            self.trueState[i, t - 1] = np.random.binomial(1, gamma_e)
                        else:
                            self.trueState[i, t - 1] = 1 - np.random.binomial(1, epsilon_e)
                        #### for pb action, wait for usual time and then pb  #############

                        if self.trueState[i, t - 1] == 0:
                            self.trueState[i, t] = self.trueState[i, t-1]
                            self.belief[i, t] = 0
                            if self.reward[i, t-1]==0:
                                self.reward[i, t] = 0
                            else:
                                self.reward[i, t] = np.random.binomial(1, 1 - rho)
                        else:
                            self.trueState[i, t] = np.random.binomial(1, beta)

                            if self.trueState[i, t] == 1: # is dropped back after bp
                                self.belief[i, t] = self.nq - 1
                                if self.reward[i, t - 1] == 0:
                                    self.reward[i, t] = 0
                                else:
                                    self.reward[i, t] = np.random.binomial(1, 1 - rho)
                            else: # not dropped back
                                self.belief[i, t] = 0
                                self.reward[i, t] = 1  # give some reward

                            #self.trueState[i, t] = 0  # if true world is one, pb resets it to zero
                            #self.belief[i, t] = 0
                            #self.reward[i, t] = 1  # give some reward

                        self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]
                        self.action[i, t] = self.policy[self.hybrid[i, t]]


    def dataGenerate_sfm(self, beliefInitial, rewInitial):
        """
        This is a function that belongs to the oneboxMDP class. In the oneboxGenerate.py file, this function is implemented
        as a separate class, since at that time, the oneboxMDP class was not defined.
        In this file, all the functions are implemented under a single class.

        :return: the observations
        """

        beta = self.parameters[0]  # available food dropped back into box after button press
        gamma = self.parameters[1]  # reward becomes available
        epsilon = self.parameters[2]  # available food disappears
        rho = self.parameters[3]  # food in mouth is consumed

        gamma_e = self.parametersExp[0]
        epsilon_e = self.parametersExp[1]

        for i in range(self.sampleNum):
            for t in range(self.sampleTime):
                if t == 0:
                    self.trueState[i, t] = np.random.binomial(1, gamma_e)

                    self.reward[i, t], self.belief[i, t] = rewInitial, beliefInitial
                    self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]    # This is for one box only
                    self.action[i, t] = self._chooseAction(np.vstack(self.softpolicy).T[self.hybrid[i, t]])
                            # action is based on softmax policy
                else:
                    if self.action[i, t-1] != pb:
                        stattemp = np.random.multinomial(1, self.ThA[self.action[i, t - 1], self.hybrid[i, t - 1], :], size = 1)
                        self.hybrid[i, t] = np.argmax(stattemp)
                            # not pressing button, hybrid state evolves probabilistically
                        self.reward[i, t], self.belief[i, t] = divmod(self.hybrid[i, t], self.nq)
                        self.action[i, t] = self._chooseAction(np.vstack(self.softpolicy).T[self.hybrid[i, t]])

                        if self.trueState[i, t - 1] == 0:
                            self.trueState[i, t] = np.random.binomial(1, gamma_e)
                        else:
                            self.trueState[i, t] = 1 - np.random.binomial(1, epsilon_e)
                    else:   # press button
                        #### for pb action, wait for usual time and then pb  #############
                        if self.trueState[i, t - 1] == 0:
                            self.trueState[i, t - 1] = np.random.binomial(1, gamma_e)
                        else:
                            self.trueState[i, t - 1] = 1 - np.random.binomial(1, epsilon_e)
                        #### for pb action, wait for usual time and then pb  #############

                        if self.trueState[i, t - 1] == 0:
                            self.trueState[i, t] = self.trueState[i, t-1]
                            self.belief[i, t] = 0
                            if self.reward[i, t-1]==0:
                                self.reward[i, t] = 0
                            else:
                                self.reward[i, t] = np.random.binomial(1, 1 - rho)
                                        # With probability 1- rho, reward is 1, not consumed
                                        # with probability rho, reward is 0, consumed
                        # if true world is one, pb resets it to zero with probability
                        else:
                            self.trueState[i, t] = np.random.binomial(1, beta)

                            if self.trueState[i, t] == 1: # is dropped back after bp
                                self.belief[i, t] = self.nq - 1
                                if self.reward[i, t - 1] == 0:
                                    self.reward[i, t] = 0
                                else:
                                    self.reward[i, t] = np.random.binomial(1, 1 - rho)
                            else: # not dropped back
                                self.belief[i, t] = 0
                                self.reward[i, t] = 1  # give some reward

                        self.hybrid[i, t] = self.reward[i, t] * self.nq + self.belief[i, t]
                        self.action[i, t] = self._chooseAction(np.vstack(self.softpolicy).T[self.hybrid[i, t]])



    def _chooseAction(self, pvec):
        # Generate action according to multinomial distribution
        stattemp = np.random.multinomial(1, pvec)
        return np.argmax(stattemp)




