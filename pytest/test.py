# %%
import numpy as np
import matplotlib.pyplot as plt
import rim

# %%
xs = [1, 1, 1]
xr = [[1, 2, 3],
      [3, 4, 5],
      [5.1, 4.7, 5.0],
      [2.1, 3.3, 3.3],
      [1.2, 4.0, 0.1],
      [5.5, 5.5, 5.5]
      ]
Nt = 44100
L = [6, 6, 6]
beta = [0.9, 0.9, 0.9, 0.99, 0.9, 0.99]
Fs = 44100
N = [10, 10, 10]
xr_dirs = [[1, 0, 0],
           [1, 0, 0],
           [0, 1, 0],
           [1, 0, 0],
           [1, 0, 0],
           [0, 1, 0]
           ]
xr_types = ['o', 'o', 'o', 'o', 'o', 'o']

h, seed = rim.rim(xs, xr, Nt, L, beta, Fs, Rd=0.0, N=N, Tw=20,
                  Fc=0.9, MicDirs=xr_dirs, MicTypes=xr_types)

# %%
fig, axes = plt.subplots(h.shape[1], 1, figsize=(8, 10))
h /= np.max(h)
for i, row in enumerate(axes):
    row.plot(h[:, i])
    row.set_ylabel("Amplitude [1]")
    row.set_xlabel("Samples [1]")
    row.spines['right'].set_visible(False)
    row.spines['top'].set_visible(False)
    row.set_ylim((-1, 1))
plt.tight_layout()
plt.show()

# %%
