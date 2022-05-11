# compute parallel upper bound on QFI for the amplitude damping noise and the SWAP-type noise, and compare with exact values of parallel and sequential QFI

from parallel import *
from sequential import *
import scipy

# setup
phi = 1.0 # the parameter to be estimated
p = 0.5 # the decay parameter of the amplitude damping channel 
g = 1.0 # the interaction strength of the SWAP-type channel
N_steps = 3 # number of steps: we consider N_steps = 2 or 3 in the paper 
d = 2 # Here we suppose all subsystems have the same dimension d = 2. This requirement is only necessary for the quantum switch strategy. 
dims_s = [d for i in range(2 * N_steps)] # list of dimensions [d_2N,...,d_1]
eps = 1.0e-12 # relative tolerance of the gap between primal and dual SDP

# construct the problem of the parallel upper bound on QFI
# dims: a list of integers, containing the dimension of each subsystem, from d_2N to d_1
# N_steps: the number of steps
# K_phis: a list of Kraus operators
# dK_phis: the derivative of K_phis
def upper_bound(K_phis, dK_phis, dims, N_steps):
    
    # variables to be optimized
    dim_h = len(K_phis)
    h = cp.Variable((dim_h, dim_h), hermitian=True)
    lambda1 = cp.Variable()
    lambda2 = cp.Variable()
    block_K_phis = cp.vstack(K_phis)
    block_dK_phis = cp.vstack(dK_phis)
    block = block_dK_phis - 1j * move_subsystem(cp.kron(np.eye(dims[0]), h), [dims[0], dim_h], [dims[0], dim_h], [0], [1]) @ block_K_phis # the derivative of an equivalent Kraus representation
    
    # calculation of the operator norm of alpha can be expressed by SDP
    A = cp.bmat([[lambda1*np.eye(dim_h*dims[0]), block], [block.H, np.eye(dims[1])]])
    beta = 1j * block_K_phis.H @ block 

    #objective
    obj = cp.Minimize(4*(N_steps*lambda1 + N_steps*(N_steps-1)*lambda2**2))
    
    #constraints
    constraints = ([A >> 0, lambda2 * np.eye(dims[1]) - beta >> 0, lambda2 * np.eye(dims[1]) + beta >> 0])

    #problem
    prob = cp.Problem(obj, constraints)
    return prob

# use Parameter in CVXPY to accelerate repeated optimization of the same problem for different parameters t (while computing the exact QFI)
# initialize the problems of the exact QFI for the amplitude damping noise
t = 1.0
N_phis_AD_init = Comb(K_phis_AD(phi, t, p), dK_phis_AD(phi, t, p), dims_s, N_steps)[0]
dN_phis_AD_init = Comb(K_phis_AD(phi, t, p), dK_phis_AD(phi, t, p), dims_s, N_steps)[1]
N_phis_AD_re = [cp.Parameter(N_phis_AD_init[i].shape) for i in range(len(N_phis_AD_init))]
N_phis_AD_im = [cp.Parameter(N_phis_AD_init[i].shape) for i in range(len(N_phis_AD_init))]
dN_phis_AD_re = [cp.Parameter(dN_phis_AD_init[i].shape) for i in range(len(dN_phis_AD_init))]
dN_phis_AD_im = [cp.Parameter(dN_phis_AD_init[i].shape) for i in range(len(dN_phis_AD_init))]

# define the problems of the exact QFI
prob_AD_par = Prob_CombQFI_par(N_phis_AD_re, N_phis_AD_im, dN_phis_AD_re, dN_phis_AD_im, dims_s, N_steps) # parallel strategy
prob_AD_seq = Prob_CombQFI_seq(N_phis_AD_re, N_phis_AD_im, dN_phis_AD_re, dN_phis_AD_im, dims_s, N_steps) # sequential strategy

# solve the problems for t ranging from 0 to 2*pi
times = np.linspace(0, 2*np.pi, 201)
QFI_AD_upper_bound = []
QFI_AD_par = []
QFI_AD_seq = []
for t in times:
    
    # upper bound
    prob_AD_upper_bound = upper_bound(K_phis_AD(phi, t, p), dK_phis_AD(phi, t, p), dims_s, N_steps) 
    QFI_AD_upper_bound.append(prob_AD_upper_bound.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.intpnt_co_tol_rel_gap: eps,
                                    mosek.iparam.intpnt_solve_form:   mosek.solveform.dual}))
    
    # exact QFI
    N_phis_AD = Comb(K_phis_AD(phi, t, p), dK_phis_AD(phi, t, p), dims_s, N_steps)[0]
    dN_phis_AD = Comb(K_phis_AD(phi, t, p), dK_phis_AD(phi, t, p), dims_s, N_steps)[1]
    for i in range(len(N_phis_AD)):
        N_phis_AD_re[i].value = np.real(0.5 * (N_phis_AD[i] + N_phis_AD[i].conj()))
        N_phis_AD_im[i].value = np.real(-0.5j * (N_phis_AD[i] - N_phis_AD[i].conj()))
        dN_phis_AD_re[i].value = np.real(0.5 * (dN_phis_AD[i] + dN_phis_AD[i].conj()))
        dN_phis_AD_im[i].value = np.real(-0.5j * (dN_phis_AD[i] - dN_phis_AD[i].conj()))
    QFI_AD_par.append(prob_AD_par.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.intpnt_co_tol_rel_gap: eps,
                                    mosek.iparam.intpnt_solve_form:   mosek.solveform.dual}))
    QFI_AD_seq.append(prob_AD_seq.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.intpnt_co_tol_rel_gap: eps,
                                    mosek.iparam.intpnt_solve_form:   mosek.solveform.dual}))
    
# save the results of QFI
np.savetxt('./data/QFI_AD_upper_bound_times_0_2pi_N_{0}.txt'.format(N_steps), QFI_AD_upper_bound, delimiter=',') # parallel strategy
np.savetxt('./data/QFI_AD_par_times_0_2pi_N_{0}.txt'.format(N_steps), QFI_AD_par, delimiter=',') # parallel strategy
np.savetxt('./data/QFI_AD_seq_times_0_2pi_N_{0}.txt'.format(N_steps), QFI_AD_seq, delimiter=',') # sequential strategy

# initialize the problems of the exact QFI for the SWAP-type noise
t = 1.0
N_phis_SWAP_init = Comb(K_phis_SWAP(phi, t, t, g), dK_phis_SWAP(phi, t, t, g), dims_s, N_steps)[0]
dN_phis_SWAP_init = Comb(K_phis_SWAP(phi, t, t, g), dK_phis_SWAP(phi, t, t, g), dims_s, N_steps)[1]
N_phis_SWAP_re = [cp.Parameter(N_phis_SWAP_init[i].shape) for i in range(len(N_phis_SWAP_init))]
N_phis_SWAP_im = [cp.Parameter(N_phis_SWAP_init[i].shape) for i in range(len(N_phis_SWAP_init))]
dN_phis_SWAP_re = [cp.Parameter(dN_phis_SWAP_init[i].shape) for i in range(len(dN_phis_SWAP_init))]
dN_phis_SWAP_im = [cp.Parameter(dN_phis_SWAP_init[i].shape) for i in range(len(dN_phis_SWAP_init))]

# define the problems of the exact QFI
prob_SWAP_par = Prob_CombQFI_par(N_phis_SWAP_re, N_phis_SWAP_im, dN_phis_SWAP_re, dN_phis_SWAP_im, dims_s, N_steps) # parallel strategy
prob_SWAP_seq = Prob_CombQFI_seq(N_phis_SWAP_re, N_phis_SWAP_im, dN_phis_SWAP_re, dN_phis_SWAP_im, dims_s, N_steps) # sequential strategy

# solve the problems for t ranging from 0 to 2*pi
QFI_SWAP_upper_bound = []
QFI_SWAP_par = []
QFI_SWAP_seq = []
for t in times:
    
    # upper bound
    prob_SWAP_upper_bound = upper_bound(K_phis_SWAP(phi, t, t, g), dK_phis_SWAP(phi, t, t, g), dims_s, N_steps) 
    QFI_SWAP_upper_bound.append(prob_SWAP_upper_bound.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.intpnt_co_tol_rel_gap: eps,
                                    mosek.iparam.intpnt_solve_form:   mosek.solveform.dual}))
    
    # exact QFI
    N_phis_SWAP = Comb(K_phis_SWAP(phi, t, t, g), dK_phis_SWAP(phi, t, t, g), dims_s, N_steps)[0]
    dN_phis_SWAP = Comb(K_phis_SWAP(phi, t, t, g), dK_phis_SWAP(phi, t, t, g), dims_s, N_steps)[1]
    for i in range(len(N_phis_SWAP)):
        N_phis_SWAP_re[i].value = np.real(0.5 * (N_phis_SWAP[i] + N_phis_SWAP[i].conj()))
        N_phis_SWAP_im[i].value = np.real(-0.5j * (N_phis_SWAP[i] - N_phis_SWAP[i].conj()))
        dN_phis_SWAP_re[i].value = np.real(0.5 * (dN_phis_SWAP[i] + dN_phis_SWAP[i].conj()))
        dN_phis_SWAP_im[i].value = np.real(-0.5j * (dN_phis_SWAP[i] - dN_phis_SWAP[i].conj()))
    QFI_SWAP_par.append(prob_SWAP_par.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.intpnt_co_tol_rel_gap: eps,
                                    mosek.iparam.intpnt_solve_form:   mosek.solveform.dual}))
    QFI_SWAP_seq.append(prob_SWAP_seq.solve(solver=cp.MOSEK, mosek_params = {mosek.dparam.intpnt_co_tol_rel_gap: eps,
                                    mosek.iparam.intpnt_solve_form:   mosek.solveform.dual}))
    
# save the results of QFI
np.savetxt('./data/QFI_SWAP_upper_bound_times_0_2pi_N_{0}.txt'.format(N_steps), QFI_SWAP_upper_bound, delimiter=',') # parallel strategy
np.savetxt('./data/QFI_SWAP_par_times_0_2pi_N_{0}.txt'.format(N_steps), QFI_SWAP_par, delimiter=',') # parallel strategy
np.savetxt('./data/QFI_SWAP_seq_times_0_2pi_N_{0}.txt'.format(N_steps), QFI_SWAP_seq, delimiter=',') # sequential strategy
    