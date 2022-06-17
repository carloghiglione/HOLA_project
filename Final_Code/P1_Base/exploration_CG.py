import copy
import sys

time_horizon = 50
seed = 17021890

sys.stdout.write('\r' + str("Initializing simulation environment"))
from P1_Base.Classes_base import *
from P1_Base.Price_puller import pull_prices_explor
import numpy as np
import matplotlib.pyplot as plt
from P7_CG.data_CG import data_dict
env = Hyperparameters(data_dict["tr_prob"], data_dict["dir_par"], data_dict["pois_par"],
                      data_dict["conv_rate"], data_dict["margin"], data_dict["meppp"])

sys.stdout.write(str(": Done") + '\n')

np.random.seed(seed)

printer = str(('\r' + str("Finding Clairvoyant solution")))
prof_1 = pull_prices_explor(env=copy.deepcopy(env), conv_rates=copy.deepcopy(env.global_conversion_rate[0]),
                            alpha=copy.deepcopy(env.dir_params[0]), n_buy=copy.deepcopy(env.mepp[0, :]),
                            trans_prob=copy.deepcopy(env.global_transition_prob), print_message=printer)
sys.stdout.write('\r' + str("Finding Clairvoyant solution: Done") + '\n')
prof_2 = pull_prices_explor(env=copy.deepcopy(env), conv_rates=copy.deepcopy(env.global_conversion_rate[1]),
                            alpha=copy.deepcopy(env.dir_params[1]), n_buy=copy.deepcopy(env.mepp[1, :]),
                            trans_prob=copy.deepcopy(env.global_transition_prob), print_message=printer)
sys.stdout.write('\r' + str("Finding Clairvoyant solution: Done") + '\n')
prof_3 = pull_prices_explor(env=copy.deepcopy(env), conv_rates=copy.deepcopy(env.global_conversion_rate[2]),
                            alpha=copy.deepcopy(env.dir_params[2]), n_buy=copy.deepcopy(env.mepp[2, :]),
                            trans_prob=copy.deepcopy(env.global_transition_prob), print_message=printer)
sys.stdout.write('\r' + str("Finding Clairvoyant solution: Done") + '\n')

plt.figure(0)
plt.plot(prof_1, color='red')
plt.tight_layout()
plt.show()
plt.figure(1)
plt.plot(prof_2, color='red')
plt.tight_layout()
plt.show()
plt.figure(2)
plt.plot(prof_3, color='red')
plt.tight_layout()
plt.show()

print(np.max(prof_1))
print(np.max(prof_2))
print(np.max(prof_3))


