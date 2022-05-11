# construct the problem of QFI for sequential strategies

from comb import *

# N_phis_re/N_phis_im: the real/imaginary part of N_phis, where N_phis is a list of vectors in the ensemble decomposition  
# N_phis_re/N_phis_im: the real/imaginary part of dN_phis, where dN_phis is the derivative of N_phis
# dims: a list of integers, containing the dimension of each subsystem, from d_2N to d_1
# N_steps: the number of steps
def Prob_CombQFI_seq(N_phis_re, N_phis_im, dN_phis_re, dN_phis_im, dims, N_steps):  
    N_phis = [N_phis_re[i] + 1j * N_phis_im[i] for i in range(len(N_phis_re))]
    dN_phis = [dN_phis_re[i] + 1j * dN_phis_im[i] for i in range(len(dN_phis_re))]
    
    # variables to be optimized
    dim_h = len(N_phis)
    h = cp.Variable((dim_h, dim_h), hermitian=True)
    if N_steps > 1:
        dims_Q = [np.prod(dims[2*(k+1):]) for k in range(N_steps-1)]
        Q_list = [cp.Variable((dims_Q[k], dims_Q[k]), hermitian=True) for k in range(N_steps-1)]
    lambda0 = cp.Variable()
    
    # construct the non-diagonal block of matrix A
    block_N_phis = cp.hstack(N_phis)
    block_dN_phis = cp.hstack(dN_phis)
    block = cp.conj(block_dN_phis - 1j * block_N_phis @ h) # block @ block.H: peformance operator from an equivalent ensemble decomposition of N_phis
    
    # choose an orthornormal basis of tensor product of H_2N and construct n_i,j 
    dim_2N = dims[0]
    dim_without_2N = np.prod(dims[1:])
    ns = []
    for i in range(dim_h):
        for j in range(dim_2N):
            ket_j = np.kron(basis_state(dim_2N, j), np.eye(dim_without_2N)) 
            ns.append(np.conj(ket_j).T @ cp.reshape(block[:,i], (block.shape[0],1)))
    block_ns = cp.hstack(ns)

    # objective
    obj = cp.Minimize(lambda0)
    
    # constraints
    if N_steps == 1:
        constraints = [cp.bmat([[1/4*lambda0*np.eye(dim_h*dim_2N), block_ns.H], [block_ns, np.eye(dims[1])]]) >> 0]
    else:
        A_matrix = cp.bmat([[1/4*lambda0*np.eye(dim_h*dim_2N), block_ns.H], [block_ns, cp.kron(np.eye(dims[1]), Q_list[0])]])
        if N_steps == 2:
            constraints = [A_matrix >> 0, partial_trace(Q_list[-1], dims[-2:], axis=0) - np.eye(dims[-1]) == 0] 
        else:
            constraints = ([A_matrix >> 0, partial_trace(Q_list[-1], dims[-2:], axis=0) - np.eye(dims[-1]) == 0]
                           + [partial_trace(Q_list[k], dims[2*(k+1):], axis=0) 
               - cp.kron(np.eye(dims[2*(k+1)+1]), Q_list[k+1]) == 0 for k in range(N_steps-2)])
    
    
    # problem
    prob = cp.Problem(obj, constraints)
    return prob