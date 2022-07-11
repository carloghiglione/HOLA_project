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
plt.plot(prof, color='black', label='expected profit')
plt.axvline(x=4**4, color='red', linestyle='--', label='p1 levels')
plt.axvline(x=2*4**4, color='red', linestyle='--')
plt.axvline(x=3*4**4, color='red', linestyle='--')
plt.axvline(x=4**3, color='blue', linestyle='--', label='p2 levels - nested')
plt.axvline(x=2*4**3, color='blue', linestyle='--')
plt.axvline(x=3*4**3, color='blue', linestyle='--')
plt.title('Visualization of single user expected profits')
plt.ylabel('Expected profit [Euro]')
plt.xlabel('Price configurations - nested')
plt.xlim(0, 1024)
plt.ylim(30, 70)
plt.legend(loc='lower right')
plt.xticks(ticks=[])
plt.tight_layout()
plt.show()

plt.figure(1)
plt.plot(prof, color='black', label='expected profit')
plt.axvline(x=4**4 + 2*4**3 + 4**2, color='green', linestyle='--', label='p3 levels - nested')
plt.axvline(x=4**4 + 2*4**3 + 2*4**2, color='green', linestyle='--')
plt.axvline(x=4**4 + 2*4**3 + 3*4**2, color='green', linestyle='--')
plt.axvline(x=4**4 + 2*4**3 + 4, color='purple', linestyle='--', label='p4 levels - nested')
plt.axvline(x=4**4 + 2*4**3 + 2*4, color='purple', linestyle='--')
plt.axvline(x=4**4 + 2*4**3 + 3*4, color='purple', linestyle='--')
plt.title('Visualization of single user expected profits - zoom on p1=1, p2=2')
plt.ylabel('Expected profit [Euro]')
plt.xlabel('Price configurations - nested')
plt.xlim(4**4+2*4**3, 4**4+3*4**3)
plt.ylim(30, 70)
plt.legend(loc='lower right')
plt.xticks(ticks=[])
plt.tight_layout()
plt.show()

print(np.max(prof))


