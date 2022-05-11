# solve the problems of QFI evaluation with amplitude damping noise for 5 families of strategies

from parallel import *
from sequential import *
from quantum_switch import *
from causal_superposition import *
from general_indefinite_causal_order import *

# setup
phi = 1.0 # the parameter to be estimated
t = 1.0 # the evolution time
N_steps = 2 # number of steps: we consider N_steps = 2 or 3 in the paper 
d = 2 # Here we suppose all subsystems have the same dimension d = 2. This requirement is only necessary for the quantum switch strategy. 
dims_s = [d for i in range(2 * N_steps)] # list of dimensions [d_2N,...,d_1]
eps = 1.0e-12 # relative tolerance of the gap between primal and dual SDP

# use Parameter in CVXPY to accelerate repeated optimization of the same problem for different parameters p or t
# initialize the problems
p = 0.5
N_phis_init = Comb(K_phis_AD(phi, t, p), dK_phis_AD(phi, t, p), dims_s, N_steps)[0]
dN_phis_init = Comb(K_phis_AD(phi, t, p), dK_phis_AD(phi, t, p), dims_s, N_steps)[1]
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

# solve the problems for p ranging from 0 to 1.0
p_range = np.linspace(0, 1.0, 201)
QFI_par = []
QFI_seq = []
QFI_swi = []
QFI_sup = []
QFI_ico = []
for p in p_range:
    N_phis = Comb(K_phis_AD(phi, t, p), dK_phis_AD(phi, t, p), dims_s, N_steps)[0]
    dN_phis = Comb(K_phis_AD(phi, t, p), dK_phis_AD(phi, t, p), dims_s, N_steps)[1]
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
np.savetxt('./data/QFI_par_rangep_0_1_N_{0}.txt'.format(N_steps), QFI_par, delimiter=',') # parallel strategy
np.savetxt('./data/QFI_seq_rangep_0_1_N_{0}.txt'.format(N_steps), QFI_seq, delimiter=',') # sequential strategy
np.savetxt('./data/QFI_swi_rangep_0_1_N_{0}.txt'.format(N_steps), QFI_swi, delimiter=',') # quantum switch strategy
np.savetxt('./data/QFI_sup_rangep_0_1_N_{0}.txt'.format(N_steps), QFI_sup, delimiter=',') # causal superposition strategy
np.savetxt('./data/QFI_ico_rangep_0_1_N_{0}.txt'.format(N_steps), QFI_ico, delimiter=',') # general indefinite-causal-order strategy