parametersAgent = [0.2, 0.15, 0.1, 0.08, 0.2, 0.15, 0.3, 5, 0.42, 0.66, 0.1] #parameters of the POMDP agent
#parametersExp = [0.2, 0.15, 0.1, 0.08, 0.42, 0.66]
parametersExp = [0.15, 0.1, 0.05, 0.04, 0.4, 0.6] #experiment parameters
 
sample_length = 500
sample_number = 200

datestring_data = '07172019(1604)'
datestring_train = '08092019(2346)'

nq = 10
nl = 3
nr = 2
na = 5
discount = 0.99
Numcol = parametersAgent[7]  # number of colors
Ncol = Numcol  - 1  # number value: 0 top Numcol-1
Nf = na + nr + nl + Numcol * 2


"""
set parameters for training
"""
input_size = Nf     # na+nr+nl+Ncol1+Ncol2
hidden_size_bel = 300   # number of neurons
output_size_bel = 2    # belief for the box (continuous)
hidden_size_act = 100
output_size_act = na
batch_size = 5
num_layers = 1
train_ratio = 0.9

NEpochs_bel = 60
NEpochs_act = 80
