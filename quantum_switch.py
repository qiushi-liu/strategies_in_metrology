# construct the problem of QFI for quantum switch strategies

from comb import *
from itertools import permutations

# N_phis_re/N_phis_im: the real/imaginary part of N_phis, where N_phis is a list of vectors in the ensemble decomposition  
# N_phis_re/N_phis_im: the real/imaginary part of dN_phis, where dN_phis is the derivative of N_phis
# all subsystems have the same dimension d
# the number of steps N_steps >= 2
def Prob_CombQFI_swi(N_phis_re, N_phis_im, dN_phis_re, dN_phis_im, d, N_steps):  
    dims = [d for i in range(2*N_steps)]
    N_phis = [N_phis_re[i] + 1j * N_phis_im[i] for i in range(len(N_phis_re))]
    dN_phis = [dN_phis_re[i] + 1j * dN_phis_im[i] for i in range(len(dN_phis_re))]   
    
    # variables to be optimized
    dim_h = len(N_phis)
    h = cp.Variable((dim_h, dim_h), hermitian=True)
    lambda0 = cp.Variable()
    
    # construct the non-diagonal block of matrix A
    block_N_phis = cp.hstack(N_phis)
    block_dN_phis = cp.hstack(dN_phis)
    block = cp.conj(block_dN_phis - 1j * block_N_phis @ h) # block @ block.H: peformance operator from an equivalent ensemble decomposition of N_phis
    
    # Choi state of the identity channel
    identity_channel = 0
    for i in range(d):
        identity_channel += np.kron(basis_state(d, i), basis_state(d, i))
    
    # (N_steps-1)-fold product of the identity channel
    multi_identity_channel = 1
    for i in range(N_steps-1):
        multi_identity_channel = np.kron(multi_identity_channel, identity_channel)
    
    # objective
    obj = cp.Minimize(lambda0)
    
    # constraints
    constraints = []
    
    # for permutation pi
    # choose an orthornormal basis of tensor product of H_2pi(N) and construct n_i,j 
    perm = permutations(range(N_steps))
    for pi in list(perm):
        ns = []
        for i in range(dim_h):
            for j in range(d):
                ket_j = np_move_subsystem(np.kron(np.kron(basis_state(d, j), multi_identity_channel), np.eye(d)),
                                          dims, [1]*(2*N_steps-1)+[d], [i for i in range(2*N_steps)], 
                                          list(np.array([[2*pi[i], 2*pi[i]+1] for i in range(N_steps)]).flatten())) 
                ns.append(np.conj(ket_j).T @ cp.reshape(block[:,i], (block.shape[0],1)))
        block_ns = cp.hstack(ns)
        
        # matrix A
        A_matrix = cp.bmat([[1/4*lambda0*np.eye(dim_h*d), block_ns.H], [block_ns, np.eye(d)]])
        
        # constraints
        constraints_pi = [A_matrix >> 0]
        constraints += constraints_pi
    
    # problem
    prob = cp.Problem(obj, constraints)
    return prob