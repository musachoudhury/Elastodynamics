import numpy as np
import matplotlib.pyplot as plt

# Define different spacing methods
def linear_spacing(n):
    return np.linspace(0, 1, n)

def quadratic_spacing(n):
    return np.linspace(0, 1, n)**2

def geometric_spacing(n):
    return np.geomspace(1e-2, 1, n)

def logarithmic_spacing(n):
    return np.logspace(-2, 0, n)

def custom_spacing(n):
    linear = np.linspace(0, 1, n)
    return np.tanh(5*linear) # Logarithmic compression

# Number of points
n_points = 100

# Generate data for each spacing method
x_linear = linear_spacing(n_points)
y_quadratic = quadratic_spacing(n_points)
y_geometric = geometric_spacing(n_points)
y_logarithmic = logarithmic_spacing(n_points)
y_custom = custom_spacing(n_points)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(x_linear, x_linear, label="Linear", linestyle='--', linewidth=2)
plt.plot(x_linear, y_quadratic, label="Quadratic")
plt.plot(np.linspace(0, 1, n_points), y_geometric, label="Geometric")
plt.plot(np.linspace(0, 1, n_points), y_logarithmic, label="Logarithmic")
plt.plot(np.linspace(0, 1, n_points), y_custom, label="Custom (log1p)")

# Add labels and legend
plt.title("Comparison of Different Spacing Methods", fontsize=16)
plt.xlabel("Index (normalized)", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
