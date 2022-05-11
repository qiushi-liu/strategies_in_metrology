# construct the problem of QFI for general indefinite-causal-order strategies

from comb import *

# N_phis_re/N_phis_im: the real/imaginary part of N_phis, where N_phis is a list of vectors in the ensemble decomposition  
# N_phis_re/N_phis_im: the real/imaginary part of dN_phis, where dN_phis is the derivative of N_phis
# dims: a list of integers, containing the dimension of each subsystem, from d_2N to d_1
# N_steps: the number of steps
def Prob_CombQFI_ico(N_phis_re, N_phis_im, dN_phis_re, dN_phis_im, dims, N_steps):  
    N_phis = [N_phis_re[i] + 1j * N_phis_im[i] for i in range(len(N_phis_re))]
    dN_phis = [dN_phis_re[i] + 1j * dN_phis_im[i] for i in range(len(dN_phis_re))]
    
    # variables to be optimized
    dim_h = len(N_phis)
    h = cp.Variable((dim_h, dim_h), hermitian=True)
    Q = cp.Variable((np.prod(dims), np.prod(dims)), hermitian=True)
    Q_list = [cp.Variable((np.prod(dims[:2*k]+dims[2*k+2:]), np.prod(dims[:2*k]+dims[2*k+2:])), hermitian=True) for k in range(N_steps)]
    lambda0 = cp.Variable()
    
    #construct the non-diagonal block of matrix A
    block_N_phis = cp.hstack(N_phis)
    block_dN_phis = cp.hstack(dN_phis)
    block = cp.conj(block_dN_phis - 1j * block_N_phis @ h) # block @ block.H: peformance operator from an equivalent ensemble decomposition of N_phis

    # objective
    obj = cp.Minimize(lambda0)
    
    # constraints
    A_matrix = cp.bmat([[1/4*lambda0*np.eye(dim_h), block.H], [block, Q]])
    constraints = ([A_matrix >> 0, cp.trace(Q) == np.prod(dims[1::2])] +   
    [partial_trace(Q, dims, axis=2*k) -  
    move_subsystem(cp.kron(1/dims[2*k+1]*np.eye(dims[2*k+1]), Q_list[k]), 
                  [dims[2*k+1]]+dims[:2*k]+dims[2*k+2:], [dims[2*k+1]]+dims[:2*k]+dims[2*k+2:], [0], [2*k]) == 0 
    for k in range(N_steps)])
    
    # problem
    prob = cp.Problem(obj, constraints)
    return prob
