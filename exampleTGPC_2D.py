# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:55:22 2024

@author: dpqb1
"""

import numpy as np
import matplotlib.pyplot as plt
import matrixOpsTGPC as mat 
import gaussianDictionary as gd
import peakCoding as pc

# Generate example data
T = 10
N1 = 31
N2 = 101
K1 = 6;
K2 = 12;
B, B_poiss, awmv_true = gd.generateExampleData2((N1,N2,T))
plt.imshow(B[:,:,1])


# Define parameters
P = {}
P['N1'] = B.shape[0]  # height
P['N2'] = B.shape[1]  # width
P['K1'] = K1          # number of atoms in vertical direction (e.g. y)
P['K2'] = K2          # number of atoms in horizontal direction (e.g. x)
P['sigmas1'] = np.linspace(0.5, 5, P['K1'])  # vertical widths
P['sigmas2'] = np.linspace(0.5, 16, P['K2'])  # horizontal widths
params = {
    'adaptRho': 1,
    'mu': 2,
    'tau': 1.05,
    'alpha': 1.8,
    'isNonnegative': 1,
    'normData': 1,
    'stoppingCriterion': 'OBJECTIVE_VALUE',
    'maxIter': 10,
    'conjGradIter': 100,
    'tolerance': 1e-8,
    'cgEpsilon': 1e-6,
    'plotProgress': 0,
    'verbose': 1
}
P['params'] = params

# Construct dictionary
A0ft = gd.peakDictionaryFFT(P)
A0 = gd.peakDictionary(P)

# Setup and solve
params['rho1'] = 1
params['lambda'] = 4e-4 * np.ones(T)
params['rho2'] = 0
params['gamma'] = 0
admmOutIndep = pc.convADMM_TGPC(A0ft, B, np.zeros((N1,N2,P['K1']*P['K2'],T)), params)
X_hat_indep = np.array(admmOutIndep[0].get())
B_hat_indep = mat.Ax_ft_Time(A0ft, X_hat_indep)

# params['gamma'] = 5e-3
# admmOut = pc.convADMM_LASSO_CG_TVphi_1D(A0ft, B_poiss, np.zeros((N, P['K'], T)), params)
# X_hat = admmOut[0]
# B_hat = mat.Ax_ft_1D_Time(A0ft, X_hat)

# Plot AWMV recovery
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

t = 2
# Plot truth
axs[0].imshow(B[:, :, t].T, aspect='auto', origin='lower', cmap='viridis')
axs[0].set_title(f'Truth (t={t})')

# Plot independent reconstruction
axs[1].imshow(B_hat_indep[:, :, t].T, aspect='auto', origin='lower', cmap='viridis')
err1a = np.linalg.norm(B_hat_indep[:, :, t] - B[:, :, t]) / np.linalg.norm(B[:, :, t])
err1b = np.linalg.norm(B_hat_indep[:, :, t] - B_poiss[:, :, t]) / np.linalg.norm(B_poiss[:, :, t])
axs[1].set_title(f'Recon (Indep), err: {err1a:.3f}, {err1b:.3f}')

# Plot Poisson data
axs[2].imshow(B_poiss[:, :, t].T, aspect='auto', origin='lower', cmap='viridis')
axs[2].set_title(f'Poisson Data (t={t})')

#%% AWMV plot
fig4, ax4 = plt.subplots(figsize=(8, 6))
awmv = gd.computeAWMV(X_hat_indep, P['sigmas1'], P['sigmas2'])

