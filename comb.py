# functions for comb construction and matrix operation

import cvxpy as cp
from cvxpy.expressions.expression import Expression
import numpy as np
import mosek

# cvxpy expression as a numpy array
def expr_as_np_array(cvx_expr):
    if cvx_expr.is_scalar(): # cvxpy scalar as a 0d array
        return np.array(cvx_expr)
    elif len(cvx_expr.shape) == 1: # cvx_expr is a 1d array 
        return np.array([v for v in cvx_expr])
    else:
        # then cvx_expr is a 2d array
        rows = []
        for i in range(cvx_expr.shape[0]):
            row = [cvx_expr[i,j] for j in range(cvx_expr.shape[1])]
            rows.append(row)
        arr = np.array(rows)
        return arr

# numpy array as a cvxpy expression
def np_array_as_expr(np_arr):
    aslist = np_arr.tolist()
    expr = cp.bmat(aslist)
    return expr

# partial trace over the subsystem at axis for numpy 2d array (assume that each subsystem is square)
# rho: a matrix (numppy 2d array))
# dims: a list containing the dimension of each subsystem
# axis: the index of the subsytem to be traced out

def np_partial_trace(rho, dims, axis):
    dims_ = np.array(dims)
    
    # return full trace of the matrix consisting of single subsystem
    if dims_.size == 1:
        return np.trace(rho)
    
    # reshape the matrix into a tensor with the following shape:
    # [dim_0, dim_1, ..., dim_n, dim_0, dim_1, ..., dim_n]
    # each subsystem gets one index for its row and another one for its column
    reshaped_rho = np.reshape(rho, np.concatenate((dims_, dims_), axis=None))
    
    # if axis is an integer, trace over single subsystem
    if isinstance(axis, int):
        traced_out_rho = np.trace(reshaped_rho, axis1=axis, axis2=dims_.size+axis)

        # traced_out_rho is still in the shape of a tensor
        # reshape back to a matrix
        dims_untraced = np.delete(dims_, axis)
        rho_dim = np.prod(dims_untraced)
        return traced_out_rho.reshape([rho_dim, rho_dim])
    
    # if axis is a list, trace over multiple subsystems
    elif isinstance(axis, list):
        axis = sorted(axis)
        dims_untraced = dims_
        traced_out_rho = reshaped_rho
        for i in range(len(axis)):
            traced_out_rho = np.trace(traced_out_rho, axis1=axis[i]-i, axis2=dims_untraced.size+axis[i]-i)
            dims_untraced = np.delete(dims_untraced, axis[i]-i)
        rho_dim = np.prod(dims_untraced)
        return traced_out_rho.reshape([rho_dim, rho_dim])
    
# partiral trace over the subsystem at axis for 2d expression
def partial_trace(rho, dims, axis):
    if not isinstance(rho, Expression):
        rho = cp.Constant(shape=rho.shape, value=rho)
    rho_np = expr_as_np_array(rho)
    traced_rho = np_partial_trace(rho_np, dims, axis)
    traced_rho = np_array_as_expr(traced_rho)
    return traced_rho
        
# move the subsystem at axis_old to axis_new for 2d numpy array of shape (np.prod(dims_in), np.prod(dims_out))
# dims_in: [dim_in_1, ..., dim_in_n]
# dims_out: [dim_out_1, ..., dim_out_n]
# axis_old: list of indices of the old position of the subsystem
# axis_new: list of indices of the new position of the subsystem
def np_move_subsystem(rho, dims_in, dims_out, axis_old, axis_new):
    dims_in_ = np.array(dims_in)
    dims_out_ = np.array(dims_out)
    axis_old_ = np.array(axis_old)
    axis_new_ = np.array(axis_new)
    reshaped_rho = np.reshape(rho, np.concatenate((dims_in_, dims_out_), axis=None))

    # swap the subsystem i and j
    reshaped_rho = np.moveaxis(reshaped_rho, np.concatenate((axis_old_, len(dims_in)+axis_old_), axis=None),
                               np.concatenate((axis_new_, len(dims_in)+axis_new_), axis=None))
    return reshaped_rho.reshape((np.prod(dims_in),np.prod(dims_out)))

# move the subsystem at axis_old to axis_new for 2d expression of shape (np.prod(dims_in), np.prod(dims_out))
def move_subsystem(rho, dims_in, dims_out, axis_old, axis_new):
    if not isinstance(rho, Expression):
        rho = cp.Constant(shape=rho.shape, value=rho)
    rho_np = expr_as_np_array(rho)
    reshaped_rho = np_move_subsystem(rho_np, dims_in, dims_out, axis_old, axis_new)
    return np_array_as_expr(reshaped_rho)

#convert decimal number n to arbitrary base b
def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

# the i-th basis vector of n-dim Hilbert space
def basis_state(n, i):
    return np.array([0]*i + [1] + [0]*(n-1-i)).reshape(n,1)

# K_phis_AD is a list of Kraus operators for a unitary evolution U(phi,t) followed by amplitude damping noise
# phi: the parameter to be estimated
# t: the unitary evolution time
# p: the decay parameter
def K_phis_AD(phi, t, p):
    # U is the unitary evolution encoding phi
    U = np.array([[np.exp(-1j * (phi * t) / 2), 0],
                  [0, np.exp(1j * (phi * t) / 2)]])
    return [np.array([[1, 0], [0, np.sqrt(1 - p)]]) @ U, np.array([[0, np.sqrt(p)], [0, 0]]) @ U]


# dK_phis_AD is the derivative of K_phis_AD
def dK_phis_AD(phi, t, p):
    # dU is the derivative of U
    dU = np.array([[(-1j * t / 2) * np.exp(-1j * (phi * t) / 2), 0],
                   [0, (1j * t / 2) * np.exp(1j * (phi * t) / 2)]])
    return [np.array([[1, 0], [0, np.sqrt(1 - p)]]) @ dU, np.array([[0, np.sqrt(p)], [0, 0]]) @ dU]

# K_phis_SWAP is a list of Kraus operators for a unitary evolution U(phi,t) followed by SWAP-type noise
# phi: the parameter to be estimated
# t: the unitary evolution time
# tau: the interaction time
# g: the interaction strength
def K_phis_SWAP(phi, t, tau, g):
    U = np.array([[np.exp(-1j * (phi*t) / 2), 0],
                  [0, np.exp(1j * (phi*t) / 2)]])
    return [np.array([[np.exp(-1j*g*tau), 0], [0, np.cos(g*tau)]]) @ U, np.array([[0, -1j*np.sin(g*tau)], [0, 0]]) @ U]

# dK_phis_SWAP is the derivative of K_phis_SWAP
def dK_phis_SWAP(phi, t, tau, g):
    dU = np.array([[(-1j * t / 2) * np.exp(-1j * (phi*t) / 2), 0],
                 [0, (1j * t / 2) * np.exp(1j * (phi*t) / 2)]])
    return [np.array([[np.exp(-1j*g*tau), 0], [0, np.cos(g*tau)]]) @ dU, np.array([[0, -1j*np.sin(g*tau)], [0, 0]]) @ dU]

# one-step comb of a channel, ensemble decomposition 
# K_phis: a list of Kraus operators
# dK_phis: the derivative of K_phis
# d_out/d_in: integer, the dimension of the output/input space
# E_phis: a list of vectors in the ensemble decomposition  
# dE_phis: the derivative of E_phis
def Comb_step(K_phis, dK_phis, d_out, d_in):
    E_phis = [K_phis[i].flatten().reshape(d_out*d_in, 1) for i in range(len(K_phis))]
    dE_phis = [dK_phis[i].flatten().reshape(d_out*d_in, 1) for i in range(len(dK_phis))]
    return E_phis, dE_phis

# (N_steps)-step comb of N_steps channels, ensemble decomposition
# dims: a list of integers, containing the dimension of each subsystem, from d_2N to d_1
# N_steps: the number of steps
# N_phis: a list of vectors in the ensemble decomposition  
# dN_phis: the derivative of N_phis
def Comb(K_phis, dK_phis, dims, N_steps):
    d_out = dims[0]
    d_in = dims[1]
    E_phis = Comb_step(K_phis, dK_phis, d_out, d_in)[0]
    dE_phis = Comb_step(K_phis, dK_phis, d_out, d_in)[1]
    def N_phi(items):
        if len(items) > 1:
            return np.kron(E_phis[items[0]], N_phi(items[1:]))
        else:
            return E_phis[items[0]]
    def dN_phi(items):
        if len(items) > 1:
            return np.kron(E_phis[items[0]], dN_phi(items[1:])) + np.kron(dE_phis[items[0]], N_phi(items[1:]))
        else:
            return dE_phis[items[0]]
    N_phis = []
    dN_phis = []
    r = len(E_phis)**N_steps
    for i in range(r):
        item = [0]*(N_steps - len(numberToBase(i, len(E_phis)))) + numberToBase(i, len(E_phis))
        N_phis.append(N_phi(item))
        dN_phis.append(dN_phi(item))
    return N_phis, dN_phis