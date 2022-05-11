# solve the problems of QFI evaluation for randomly sampled rank-2 channels as noise

from parallel import *
from sequential import *
from quantum_switch import *
from causal_superposition import *
from general_indefinite_causal_order import *

# setup
phi = 1.0 # the parameter to be estimated
t = 1.0 # the evolution time
N_steps = 2 # number of steps
d = 2 # Here we suppose all subsystems have the same dimension d = 2. This requirement is only necessary for the quantum switch strategy. 
dims_s = [d for i in range(2 * N_steps)] # list of dimensions [d_2N,...,d_1]
eps = 1.0e-12 # relative tolerance of the gap between primal and dual SDP
num_channels = 1000 # number of random channels

# uncomment the following block of code to generate new random channels; otherwise load the data used in the paper
"""
# generate random rank-2 channels

import qutip as qt
channels_range = [] # list of random channels in the Kraus representation
for i in range(num_channels):
    rand_choi = np_move_subsystem(qt.to_choi(qt.rand_super_bcsz(N=2, enforce_tp=True, rank=2, dims=None)).full(), [2,2], [2,2], [0], [1]) # Choi operator of a random channel
    rand_channel = [np.reshape(np.sqrt(np.linalg.eig(rand_choi)[0][i]) * np.linalg.eig(rand_choi)[1][:,i], (2,2)) for i in range(rand_choi.shape[1])] # list of Kraus operators
    channels_range.append(rand_channel)

# save random channels
np.save('./data/my_{0}_rand_channels.npy'.format(num_channels), channels_range, allow_pickle=True)   
"""

# load the data used in the paper, or alternatively use the new data generated
# channels_range = np.load('my_{0}_rand_channels.npy'.format(num_channels), allow_pickle=True)
channels_range = np.load('./data/1000_rand_channels.npy', allow_pickle=True)

# list of Kraus operators of unitary evolution followed by rank-2 noise channels
def K_phis_rand(rand_channel):
    U = np.array([[np.exp(-1j * (phi*t) / 2), 0],
             [0, np.exp(1j * (phi*t) / 2)]])
    return [rand_channel[i]@U for i in range(len(rand_channel))]

# the derivative of Kraus operators
def dK_phis_rand(rand_channel):
    dU = np.array([[(-1j * t / 2) * np.exp(-1j * (phi*t) / 2), 0],
                 [0, (1j * t / 2) * np.exp(1j * (phi*t) / 2)]])
    return [rand_channel[i]@dU for i in range(len(rand_channel))]

#use Parameter in CVXPY to accelerate repeated optimization of the same problem for different channels
#initialize the problems
rand_channel = channels_range[0]
N_phis_init = Comb(K_phis_rand(rand_channel), dK_phis_rand(rand_channel), dims_s, N_steps)[0]
dN_phis_init = Comb(K_phis_rand(rand_channel), dK_phis_rand(rand_channel), dims_s, N_steps)[1]
N_phis_re = [cp.Parameter(N_phis_init[i].shape) for i in range(len(N_phis_init))]
N_phis_im = [cp.Parameter(N_phis_init[i].shape) for i in range(len(N_phis_init))]
dN_phis_re = [cp.Parameter(dN_phis_init[i].shape) for i in range(len(dN_phis_init))]
dN_phis_im = [cp.Parameter(dN_phis_init[i].shape) for i in range(len(dN_phis_init))]

# define the problems for 5 families of strategies
prob_par = Prob_CombQFI_par(N_phis_re, N_phis_im, dN_phis_re, dN_phis_im, dims_s, N_steps) # parallel strategy
prob_seq = Prob_CombQFI_seq(N_phis_re, N_phis_im, dN_phis_re, dN_phis_im, dims_s, N_steps) # sequential strategy
prob_swi = Prob_CombQFI_swi(N_phis_re, N_phis_im, dN_phis_re, dN_phis_im, d, N_steps) # quantum switch strategy
prob_sup = Prob_CombQFI_sup(N_phis_re, N_phis_im, dN_phis_re, dN_phis_im, dims_s, N_steps) # causal superposition strategy
prob_ico = Prob_CombQFI_ico(N_phis_re, N_phis_im, dN_phis_re, dN_phis_im, dims_s, N_steps) # general indefinite-causal-order strategy

# solve the problems
QFI_par = []
QFI_seq = []
QFI_swi = []
QFI_sup = []
QFI_ico = []
for rand_channel in channels_range:
    N_phis = Comb(K_phis_rand(rand_channel), dK_phis_rand(rand_channel), dims_s, N_steps)[0]
    dN_phis = Comb(K_phis_rand(rand_channel), dK_phis_rand(rand_channel), dims_s, N_steps)[1]
    for i in range(len(N_phis)):
        N_phis_re[i].value = np.real(0.5 * (N_phis[i] + N_phis[i].conj()))
        N_phis_im[i].value = np.real(-0.5j * (N_phis[i] - N_phis[i].conj()))
        dN_phis_re[i].value = np.real(0.5 * (dN_phis[i] + dN_phis[i].conj()))
        dN_phis_im[i].value = np.real(-0.5j * (dN_phis[i] - dN_phis[i].conj()))
    QFI_par.append(prob_par.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.intpnt_co_tol_rel_gap: eps,
                                    mosek.iparam.intpnt_solve_form:   mosek.solveform.dual}))
    QFI_seq.append(prob_seq.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.intpnt_co_tol_rel_gap: eps,
                                    mosek.iparam.intpnt_solve_form:   mosek.solveform.dual}))
    QFI_swi.append(prob_swi.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.intpnt_co_tol_rel_gap: eps,
                                    mosek.iparam.intpnt_solve_form:   mosek.solveform.dual}))
    QFI_sup.append(prob_sup.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.intpnt_co_tol_rel_gap: eps,
                                    mosek.iparam.intpnt_solve_form:   mosek.solveform.dual}))
    QFI_ico.append(prob_ico.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.intpnt_co_tol_rel_gap: eps,
                                    mosek.iparam.intpnt_solve_form:   mosek.solveform.dual}))
    
# save the results of QFI
np.savetxt('./data/QFI_par_rand_channels_N_{0}.txt'.format(N_steps), QFI_par, delimiter=',') # parallel strategy
np.savetxt('./data/QFI_seq_rand_channels_N_{0}.txt'.format(N_steps), QFI_seq, delimiter=',') # sequential strategy
np.savetxt('./data/QFI_swi_rand_channels_N_{0}.txt'.format(N_steps), QFI_swi, delimiter=',') # quantum switch strategy
np.savetxt('./data/QFI_sup_rand_channels_N_{0}.txt'.format(N_steps), QFI_sup, delimiter=',') # causal superposition strategy
np.savetxt('./data/QFI_ico_rand_channels_N_{0}.txt'.format(N_steps), QFI_ico, delimiter=',') # general indefinite-causal-order strategy

# compare QFI of different strategies
comparison_4_strategies = []
comparison_swi_seq = []
comparison_swi_par = []
for i in range(num_channels):
    comparison_list_4_strategies.append(QFI_ico[i] > QFI_sup[i]+1e-8 > QFI_seq[i]+2e-8 > QFI_par[i]+3e-8)
    comparison_list_swi_seq.append(QFI_swi_rand_2[i]> QFI_seq_rand_2[i]+1e-8)
    comparison_list_swi_par.append(QFI_swi_rand_2[i]> QFI_par_rand_2[i]+1e-8)
    
# print the number of channels sarisfying the following conditions
print('ICO > Sup > Seq > Par: ', sum(comparison_4_strategies))
print('Swi > Seq: ', sum(comparison_swi_seq))
print('Swi > Par: ', sum(comparison_swi_par))