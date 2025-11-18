"""
Physical constants and parameters for PINN simulation
"""

# Concentration bounds
cmax = 1
cmin = 0

# Velocity bounds
umax = 1

# Training parameters
AD_EP = 1
LB_EP = 1

# Reynolds and Schmidt number bounds
re_bnd = [5, 40]
sc_bnd = [1, 100]