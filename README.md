# TemporalGaussianPeakCoding

This repository implements a temporal convolutional sparse coding framework using multiscale Gaussian dictionaries. It supports both 1D and 2D data, with evolving Gaussian peaks over time, and total variation regularization.

## Features

- Simulated data generation with time-varying Gaussian peaks
- Multiscale separable Gaussian dictionary construction (1D & 2D)
- Convolutional ADMM solver for sparse coding
- Poisson noise modeling
- AWMV (Amplitude Weighted Mean Variance) analysis
- Modular architecture for flexibility and extensibility

## Structure

- `matrixOpsTGPC.py` — FFT-based forward/adjoint convolution operators
- `gaussianDictionary.py` — Dictionary construction and synthetic data generation
- `peakCoding.py` — ADMM solvers for sparse coding (TV and independent)
- `test_script.py` — Example script demonstrating 2D time series reconstruction

## Example Usage

```python
# Generate synthetic data
B, B_poiss, awmv_true = generateExampleData2((N1, N2, T))

# Construct dictionary
P = {
  'N1': N1,
  'N2': N2,
  'K1': 6,
  'K2': 12,
  'sigmas1': np.linspace(0.5, 5, 6),
  'sigmas2': np.linspace(0.5, 16, 12),
  'params': {
      'rho1': 1,
      'lambda': 4e-4 * np.ones(T),
      'gamma': 0,
      'rho2': 0,
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
}

A0ft = peakDictionaryFFT(P)
X_init = np.zeros((N1, N2, P['K1']*P['K2'], T))

# Solve
X_hat, _ = convADMM_TGPC(A0ft, B_poiss, X_init, P['params'])

# Evaluate
B_hat = Ax_ft_Time(A0ft, X_hat)
awmv = computeAWMV(X_hat, P['sigmas1'], P['sigmas2'])
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- (Optional) CuPy for GPU acceleration (not used in this CPU version)

## Author

Daniel Banco

## License

MIT License
