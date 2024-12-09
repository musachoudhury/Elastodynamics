import numpy as np

# Nsteps = 200
# times = np.linspace(0, 1, Nsteps + 1)
def custom_spacing(n, end):
    x = np.linspace(0, end, n)
    return np.tanh(x) # Logarithmic compression

times = custom_spacing(400, 3)
print(times)

#print(np.diff(times))