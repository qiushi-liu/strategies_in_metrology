# plot the figures

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
# hierarchy of strategies for the amplitude damping noise
p_range = np.linspace(0, 1.0, 201)
N_steps = 2 # N_steps: 2 or 3
QFI_par = np.loadtxt('./data/QFI_par_rangep_0_1_N_{0}.txt'.format(N_steps), delimiter=',')
QFI_seq = np.loadtxt('./data/QFI_seq_rangep_0_1_N_{0}.txt'.format(N_steps), delimiter=',')
QFI_swi = np.loadtxt('./data/QFI_swi_rangep_0_1_N_{0}.txt'.format(N_steps), delimiter=',')
QFI_sup = np.loadtxt('./data/QFI_sup_rangep_0_1_N_{0}.txt'.format(N_steps), delimiter=',')
QFI_ico = np.loadtxt('./data/QFI_ico_rangep_0_1_N_{0}.txt'.format(N_steps), delimiter=',')
fig, ax1 = plt.subplots() 
ax1.plot(p_range, QFI_par, label='Par')
ax1.plot(p_range, QFI_seq, label='Seq')
ax1.plot(p_range, QFI_swi, label='SWI')
ax1.plot(p_range, QFI_sup, label='Sup')
ax1.plot(p_range, QFI_ico, label='ICO')
ax1.set_xlabel('$p$')  
ax1.set_ylabel('QFI')  
ax1.legend()  
ax2 = ax1.inset_axes([0.1, 0.1, 0.4, 0.4])
ax2.plot(p_range, QFI_par, label='Par')
ax2.plot(p_range, QFI_seq, label='Seq')
ax2.plot(p_range, QFI_swi, label='SWI')
ax2.plot(p_range, QFI_sup, label='Sup')
ax2.plot(p_range, QFI_ico, label='ICO')
ax2.set_xlim(0.35, 0.45)
ax2.set_ylim(2.0, 3.0)

# use the following code if N_steps = 3
# ax2.set_xlim(0.1, 0.2)
# ax2.set_ylim(6.0, 9.0)
plt.savefig('./figures/N_{0}_AD_hierarchy_strategies.pdf'.format(N_steps))
plt.show()

# compare the parallel upper bound on QFI and the exact parallel/sequential QFI
# the amplitude damping noise
times=np.linspace(0, 2*np.pi, 201)
QFI_AD_upper_bound = np.loadtxt('./data/QFI_AD_upper_bound_times_0_2pi_N_{0}.txt'.format(N_steps), delimiter=',')
QFI_AD_seq = np.loadtxt('./data/QFI_AD_seq_times_0_2pi_N_{0}.txt'.format(N_steps), delimiter=',')
QFI_AD_par = np.loadtxt('./data/QFI_AD_par_times_0_2pi_N_{0}.txt'.format(N_steps), delimiter=',')
fig, ax = plt.subplots()  
ax.plot(times, QFI_AD_upper_bound, label='parallel upper bound', color='k', linestyle='dashed')
ax.plot(times, np.array(QFI_AD_seq), label='sequential (exact value)')
ax.plot(times, np.array(QFI_AD_par), label='parallel (exact value)')
ax.set_xlabel('$t$')  
ax.set_ylabel('QFI')  
ax.legend() 
plt.savefig('./figures/N_{0}_AD.pdf'.format(N_steps))
plt.show()

# the SWAP-type noise
QFI_SWAP_upper_bound = np.loadtxt('./data/QFI_SWAP_upper_bound_times_0_2pi_N_{0}.txt'.format(N_steps), delimiter=',')
QFI_SWAP_seq = np.loadtxt('./data/QFI_SWAP_seq_times_0_2pi_N_{0}.txt'.format(N_steps), delimiter=',')
QFI_SWAP_par = np.loadtxt('./data/QFI_SWAP_par_times_0_2pi_N_{0}.txt'.format(N_steps), delimiter=',')
fig, ax = plt.subplots()  
ax.plot(times, QFI_SWAP_upper_bound, label='parallel upper bound', color='k', linestyle='dashed')
ax.plot(times, np.array(QFI_SWAP_seq), label='sequential (exact value)')
ax.plot(times, np.array(QFI_SWAP_par), label='parallel (exact value)')
ax.set_xlabel('$t$')  
ax.set_ylabel('QFI')  
ax.legend() 
plt.savefig('./figures/N_{0}_SWAP.pdf'.format(N_steps))
plt.show()
