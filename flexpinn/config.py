"""
Configuration and constants for PINN simulation
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.interpolate import CubicSpline

# Device configuration
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Set random seeds for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

# Print device info
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of CUDA devices: {torch.cuda.device_count()}")
print(f"Using device: {device}")

# Matplotlib configuration
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "lines.linewidth": 2,
    "axes.linewidth": 1,
})

# Geometry constants
L0 = 0.9 / 0.3
L = 2.7 / 0.3
d = 0.3 / 0.3
l = 0.3 / 0.3

# Control points
p0 = np.array([0, 0])
p4 = np.array([0.5, 0])

p1_max = np.array([0.125, 0.5])
p2_max = np.array([0.25, 0.5])
p3_max = np.array([0.375, 0.5])

p1_min = np.array([0.125, -0.5])
p2_min = np.array([0.25, -0.5])
p3_min = np.array([0.375, -0.5])

# Parameter bounds
p1_mi, p2_mi, p3_mi, Re_mi, Sc_mi = p1_min[1], p2_min[1], p3_min[1], 0, 1
p1_ma, p2_ma, p3_ma, Re_ma, Sc_ma = p1_max[1], p2_max[1], p3_max[1], 0, 1

# Domain dimensions
lf = p4[0] - p0[0]
hf = p2_max[1]
h = d
ndf = 2

x_min = 0.0
x_max = L
y_min = -d/2 - hf
y_max = d/2 + hf

# Collocation points counts
N_c = 80000
N_r = 20000
N_f = 5000
N_i = 1000
N_w = 1000

# Upper and lower bounds
ub = np.array([x_max, y_max, p1_ma, p2_ma, p3_ma, Re_ma, Sc_ma])
lb = np.array([x_min, y_min, p1_mi, p2_mi, p1_mi, Re_mi, Sc_mi])