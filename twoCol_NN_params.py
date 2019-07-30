parametersAgent = [0.2, 0.15, 0.1, 0.08, 0.2, 0.15, 0.3, 5, 0.42, 0.66, 0.1]
# parametersExp = [0.15,0.1,0.05,0.04,0.4,0.6]
parametersExp = [0.2, 0.15, 0.1, 0.08, 0.42, 0.66]

datestring_data = '07172019(1604)'
datestring_run = '07262019(1441)'

nq = 10
nl = 3
nr = 2
na = 5
Ncol1 = parametersAgent[7]
Ncol2 = parametersAgent[7]
Nf = na + nr + nl + Ncol1 + Ncol2

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
NEpochs_act = 300
