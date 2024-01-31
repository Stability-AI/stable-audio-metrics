from scipy.special import rel_entr
import numpy as np
from math import log
import torch
import torch.nn.functional as F

# https://machinelearningmastery.com/divergence-between-probability-distributions/ 
def kl_divergence(p, q):
	return sum(p[i] * log(p[i]/q[i]) for i in range(len(p)))

###########################################
# EXAMPLE 1: KL(P||Q) = 0.589885181619163 #
###########################################
P = [.05, .1, .2, .05, .15, .25, .08, .12]
Q = [.3, .1, .2, .1, .1, .02, .08, .1]

print(kl_divergence(P, Q))
# output: 0.589885181619163

# https://www.statology.org/kl-divergence-python/#:~:text=If%20we%20have%20two%20probability,“P%27s%20divergence%20from%20Q.”&text=If%20the%20KL%20divergence%20between,We%20can%20use%20the%20scipy.
print(sum(rel_entr(P, Q)))
# output: 0.589885181619163

# https://github.com/pytorch/pytorch/issues/7337
# https://discuss.pytorch.org/t/kl-divergence-different-results-from-tf/56903/2
print(F.kl_div((torch.tensor(Q) + 1e-6).log(), torch.tensor(P), reduction='sum', log_target=False).item()) # note: F.kl_div(x, y) is KL(y||x)
# output: 0.5898663401603699

###############################
# EXAMPLE 2: KL(p||q) = 1.336 #
###############################
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]

print(kl_divergence(p, q))
# output: 1.3356800935337299

print(sum(rel_entr(p, q)))
# output: 1.3356800935337299

print(F.kl_div((torch.tensor(q) + 1e-6).log(), torch.tensor(p), reduction='sum', log_target=False).item()) # note: F.kl_div(x, y) is KL(y||x)
# output: 1.335667371749878