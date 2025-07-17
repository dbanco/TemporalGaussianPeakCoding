# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:55:22 2024

@author: dpqb1
"""

import numpy as np
import matplotlib.pyplot as plt
import matrixOpsTGPC as mat 
import gaussianDictionary as gd
import peakCodingGPU as pc

# Generate example data
T = 10
N = 101
K = 20;
B, B_poiss, awmv_true = gd.generateExampleData(N, T)


# Define parameters
P = {}
P['N'] = B.shape[0]
P['K'] = K
P['sigmas'] = np.linspace(1/2, 16, P['K'])
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
admmOutIndep = pc.convADMM_TGPC(A0ft, B, np.zeros((N, P['K'], T)), params)
X_hat_indep = np.array(admmOutIndep[0].get())
B_hat_indep = mat.Ax_ft_Time(A0ft, X_hat_indep)

# Plot awmv recovery
fig, axs = plt.subplots(1, 4, figsize=(15, 4))

# Plot truth
axs[0].imshow(B.T, aspect='auto', origin='lower', cmap='viridis')
axs[0].set_title('Truth')

# Plot independent reconstruction
axs[1].imshow(B_hat_indep.T, aspect='auto', origin='lower', cmap='viridis')
err1a = np.linalg.norm(B_hat_indep - B) / np.linalg.norm(B)
err1b = np.linalg.norm(B_hat_indep - B_poiss) / np.linalg.norm(B_poiss)
axs[1].set_title(f'Recon (Indep), err: {err1a:.3f}, {err1b:.3f}')


