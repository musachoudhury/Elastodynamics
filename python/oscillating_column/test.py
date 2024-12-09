import numpy as np

Nsteps = 350
Nsave = 100
times = np.linspace(0, 2, Nsteps + 1)
save_freq = Nsteps // Nsave

print(save_freq)