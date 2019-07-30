import os
from datetime import datetime

path = os.getcwd()
datestring_run = datetime.strftime(datetime.now(), '%m%d%Y(%H%M)')

from twoCol_NN_params import *
from twoCol_NN_model import *

bel_model = rnn(input_size, hidden_size_bel, output_size_bel, num_layers)
bel_model.load_state_dict(torch.load(path + '/Results/' + datestring_run + '_belNN' + '_' + str(NEpochs_bel) + '_data' + datestring_data))
bel_model.eval()

act_model = net(hidden_size_bel, hidden_size_act, output_size_act)
act_model.load_state_dict(torch.load(path + '/Results/' + datestring_run + '_actNN' + '_' + str(NEpochs_act) + '_data' + datestring_data))
act_model.eval()

