# construct the problem of QFI for causal superposition strategies

from comb import *
from itertools import permutations

# N_phis_re/N_phis_im: the real/imaginary part of N_phis, where N_phis is a list of vectors in the ensemble decomposition  
# N_phis_re/N_phis_im: the real/imaginary part of dN_phis, where dN_phis is the derivative of N_phis
# dims: a list of integers, containing the dimension of each subsystem, from d_2N to d_1
# the number of steps N_steps >= 2
def Prob_CombQFI_sup(N_phis_re, N_phis_im, dN_phis_re, dN_phis_im, dims, N_steps):
    N_phis = [N_phis_re[i] + 1j * N_phis_im[i] for i in range(len(N_phis_re))]
    dN_phis = [dN_phis_re[i] + 1j * dN_phis_im[i] for i in range(len(dN_phis_re))]   
    dim_h = len(N_phis)
    
    # variables to be optimized
    dim_h = len(N_phis)
    h = cp.Variable((dim_h, dim_h), hermitian=True)
    lambda0 = cp.Variable()
    
    # construct the non-diagonal block of matrix A
    block_N_phis = cp.hstack(N_phis)
    block_dN_phis = cp.hstack(dN_phis)
    block = cp.conj(block_dN_phis - 1j * block_N_phis @ h) # block @ block.H: peformance operator from an equivalent ensemble decomposition of N_phis
    
    # objective
    obj = cp.Minimize(lambda0)
    
    # constraints
    constraints = []
    
    # for permutation pi
    # choose an orthornormal basis of tensor product of H_2pi(N) and construct n_i,j 
    perm = permutations(range(N_steps))
    for pi in list(perm):
        dim_2piN = dims[2*pi[0]]
        dim_without_2piN = np.prod(dims)//dim_2piN
        ns = []
        for i in range(dim_h):
            for j in range(dim_2piN):
                ket_j = np_move_subsystem(np.kron(basis_state(dim_2piN, j), np.eye(dim_without_2piN)),
                                          dims, [1]+dims[:2*pi[0]]+dims[2*pi[0]+1:], [0], [2*pi[0]]) 
                ns.append(np.conj(ket_j).T @ cp.reshape(block[:,i], (block.shape[0],1)))
        block_ns = cp.hstack(ns)
        
        # dual comb Q
        # lists of dimensions of subsystems for Q matrices
        dims_Q_list = []
        # dimenisons of Q matrices
        dim_Q_list = []
        for k in range(1, N_steps):
            remove_indices = list(np.array([[2*pi[i], 2*pi[i]+1] for i in range(k)]).flatten())
            dims_Q_k = [i for j, i in enumerate(dims) if j not in remove_indices]
            dims_Q_list.append(dims_Q_k)
            dim_Q_list.append(np.prod(dims_Q_k))
        Q_list = [cp.Variable((dim_Q_list[k], dim_Q_list[k]), hermitian=True) for k in range(N_steps-1)]
        
        # matrix A
        A_matrix = cp.bmat([[1/4*lambda0*np.eye(dim_h*dim_2piN), block_ns.H], 
                            [block_ns, move_subsystem(cp.kron(np.eye(dims[2*pi[0]+1]), Q_list[0]), 
                                                      [dims[2*pi[0]+1]]+dims[:2*pi[0]]+dims[2*pi[0]+2:], 
                                                      [dims[2*pi[0]+1]]+dims[:2*pi[0]]+dims[2*pi[0]+2:], [0], [2*pi[0]])]])
        
        # constraints
        if N_steps == 2:
            constraints_pi = [A_matrix >> 0, partial_trace(Q_list[-1], dims_Q_list[-1], axis=0) - np.eye(dims[2*pi[-1]+1]) == 0] 
        else:
            
            # find the index for the subsystem traced over in the relations between Q matrices
            indices_identity = []
            for k in range(1, N_steps-1):
                remove_indices_identity = [pi[i] for i in range(k)]
                indices_identity.append(2*([i for i in range(N_steps) if i not in remove_indices_identity].index(pi[k+1])))

            constraints_pi = ([A_matrix >> 0, partial_trace(Q_list[-1], dims_Q_list[-1], axis=0) - np.eye(dims[2*pi[-1]+1]) == 0] 
                           + [partial_trace(Q_list[k], dims_Q_list[k], axis=indices_identity[k]) 
               - move_subsystem(cp.kron(np.eye(dims[2*pi[k+1]+1]), Q_list[k+1]), [dims[2*pi[k+1]+1]]+dims_Q_list[k+1],
                                [dims[2*pi[k+1]+1]]+dims_Q_list[k+1], [0], [indices_identity[k]]) == 0 for k in range(N_steps-2)])
        constraints += constraints_pi
        
    # problem
    prob = cp.Problem(obj, constraints)
    return prob