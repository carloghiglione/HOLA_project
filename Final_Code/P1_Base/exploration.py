import copy
import sys

time_horizon = 50
seed = 17021890

sys.stdout.write('\r' + str("Initializing simulation environment"))
from P1_Base.Classes_base import *
from P1_Base.Price_puller import pull_prices_explor
import numpy as np
import matplotlib.pyplot as plt
from data_cruise import data_dict
env = Hyperparameters(data_dict["tr_prob"], data_dict["dir_par"], data_dict["pois_par"],
                      data_dict["conv_rate"], data_dict["margin"], data_dict["meppp"])

sys.stdout.write(str(": Done") + '\n')

np.random.seed(seed)

printer = str(('\r' + str("Finding Clairvoyant solution")))
prof = pull_prices_explor(env=copy.deepcopy(env), conv_rates=copy.deepcopy(env.global_conversion_rate),
                          alpha=copy.deepcopy(env.dir_params), n_buy=copy.deepcopy(env.mepp),
                          trans_prob=copy.deepcopy(env.global_transition_prob), print_message=printer)
sys.stdout.write('\r' + str("Finding Clairvoyant solution: Done") + '\n')

plt.figure(0)
plt.plot(prof, color='red')
plt.tight_layout()
plt.show()

print(np.max(prof))


