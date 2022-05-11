# Quantum channel estimation within constraints on strategies
This repository contains the Python code and data accompanying the article ["Strict Hierarchy of Strategies for Non-asymptotic Quantum Metrology"](https://arxiv.org/abs/2203.09758). Give finite queries to an unknown quantum channel within specified constraints, the quantum Fisher information (QFI) is numerically computed via semidefinite programming (SDP). Different constraints define different families of strategies, including parallel, sequential and indefinite-causal-order strategies. A strict hierarchy of QFI for these families of strategies is shown by the numerical results. The commonly used upper bound on QFI (see [Fujiwara et al.](https://iopscience.iop.org/article/10.1088/1751-8113/41/25/255304) or [Demkowicz-Dobrzański et al.](https://www.nature.com/articles/ncomms2067)) for parallel strategies is also evaluated, for comparison with the exact QFI. 
## Requirements
The code for SDP requires the open source Python package [CVXPY](https://www.cvxpy.org) with the optimizer [MOSEK](https://www.mosek.com). The code for generating random quantum channels requires the open source Python package [QuTiP](https://qutip.org).
## Description
* `comb.py` contains useful functions for the comb construction and matrix operation.
* `parallel.py`, `sequential.py`, `quantum_switch.py`, `causal_superposition.py` and `general_indefinite_causal_order.py` construct the problem of QFI evaluation for parallel, sequential, quantum switch, causal superposition and general indefinite_causal_order strategies respectively.
* `solve.py` solves the problems of QFI evaluation with amplitude damping noise for 5 families of strategies.
* `random_channel_estimation.py` solves the problems of QFI evaluation with random rank-2 quantum channels as noise for 5 families of strategies.
* `bound.py` computes the parallel QFI upper bound for the amplitude damping noise and the SWAP-type noise, and compare with exact values of parallel and parallel and sequential QFI.
* `plot.py` draws the figures showing the hierarchy of strategies and the gaps between QFI upper bounds and exact values.
## How to use
* To solve the problems of QFI evaluation for the amplitude damping noise, feel free to adjust the setup in `solve.py`
```python
# setup
phi = 1.0 # the parameter to be estimated
t = 1.0 # the evolution time
N_steps = 2 # number of steps: we consider N_steps = 2 or 3 in the paper 
d = 2 # Here we suppose all subsystems have the same dimension d = 2. This requirement is only necessary for the quantum switch strategy. 
dims_s = [d for i in range(2 * N_steps)] # list of dimensions [d_2N,...,d_1]
eps = 1.0e-12 # relative tolerance of the gap between primal and dual SDP
```
and run `solve.py`. The other noise models can also be used for QFI evaluation, by defining the Kraus operators with the derivative in `comb.py`, similar to
```python
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
```
* Similarly the setup can be adjusted in `random_channel_estimation.py` to solve the problems of QFI evaluation for the random channels as noise, and in `bound.py` to compute the parallel QFI upper bound.
* `1000_rand_channels.npy` in `data` contains the Kraus operators of 1000 randomly sampled channels and can be directly used for random channel estimation. Alternatively, to generate new random channels, uncomment the block of code in `random_channel_estimation.py`
```python
# generate random rank-2 channels
import qutip as qt
channels_range = [] # list of random channels in the Kraus representation
for i in range(num_channels):
    rand_choi = np_move_subsystem(qt.to_choi(qt.rand_super_bcsz(N=2, enforce_tp=True, rank=2, dims=None)).full(), [2,2], [2,2], [0], [1]) # Choi operator of a random channel
    rand_channel = [np.reshape(np.sqrt(np.linalg.eig(rand_choi)[0][i]) * np.linalg.eig(rand_choi)[1][:,i], (2,2)) for i in range(rand_choi.shape[1])] # list of Kraus operators
    channels_range.append(rand_channel)
# save random channels
np.save('./data/my_{0}_rand_channels.npy'.format(num_channels), channels_range, allow_pickle=True)   
```
and do not load `1000_rand_channels.npy`
```python
# load the data used in the paper, or alternatively use the new data generated
channels_range = np.load('my_{0}_rand_channels.npy'.format(num_channels), allow_pickle=True)
# channels_range = np.load('./data/1000_rand_channels.npy', allow_pickle=True)
```
