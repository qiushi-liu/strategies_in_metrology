# construct the problem of QFI for parallel strategies

from comb import *

# N_phis_re/N_phis_im: the real/imaginary part of N_phis, where N_phis is a list of vectors in the ensemble decomposition  
# N_phis_re/N_phis_im: the real/imaginary part of dN_phis, where dN_phis is the derivative of N_phis
# dims: a list of integers, containing the dimension of each subsystem, from d_2N to d_1
# N_steps: the number of steps
def Prob_CombQFI_par(N_phis_re, N_phis_im, dN_phis_re, dN_phis_im, dims, N_steps):  
    N_phis = [N_phis_re[i] + 1j * N_phis_im[i] for i in range(len(N_phis_re))]
    dN_phis = [dN_phis_re[i] + 1j * dN_phis_im[i] for i in range(len(dN_phis_re))]
    
    # variables to be optimized
    dim_h = len(N_phis)
    h = cp.Variable((dim_h, dim_h), hermitian=True)
    lambda0 = cp.Variable()
    
    #construct the non-diagonal block of matrix A
    block_N_phis = cp.hstack(N_phis)
    block_dN_phis = cp.hstack(dN_phis)
    block = cp.conj(block_dN_phis - 1j * block_N_phis @ h) # block @ block.H: peformance operator from an equivalent ensemble decomposition of N_phis
    
    # choose an orthornormal basis of tensor product of (H_2N,...,H4,H_2) and construct n_i,j 
    dim_out = np.prod(dims[::2])
    dim_in = np.prod(dims[1::2])
    ns = []
    for i in range(dim_h):
        for j in range(dim_out):
            item = [0]*(N_steps - len(numberToBase(j, dims[0]))) + numberToBase(j, dims[0])
            ket_j = np.kron(basis_state(dims[0], item[0]), np.eye(2))
            if N_steps == 1:
                ns.append(np.conj(ket_j).T @ cp.reshape(block[:,i], (block.shape[0],1)))
            else:
                for k in range(1, N_steps):
                    ket_j = np.kron(ket_j, np.kron(basis_state(dims[2*k], item[k]), np.eye(2)))   
                ns.append(np.conj(ket_j).T @ cp.reshape(block[:,i], (block.shape[0],1)))
    block_ns = cp.hstack(ns)

    # objective
    obj = cp.Minimize(lambda0)
    
    # constraints
    constraints = [cp.bmat([[1/4*lambda0*np.eye(dim_h*dim_out), block_ns.H], [block_ns, np.eye(dim_in)]]) >> 0]
    
    # problem
    prob = cp.Problem(obj, constraints)
    return prob