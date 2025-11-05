"""
Boundary condition point generation for PINN training
"""

import numpy as np
from pyDOE import lhs
from scipy.interpolate import CubicSpline
from config import *

def generate_wall_boundaries():
    """Generate wall boundary points"""
    # Generate initial wall points
    wall_upo = [d, lf, p1_mi, p2_mi, p1_mi, Re_mi, Sc_mi] + \
               [L-d, 0, p1_ma - p1_mi, p2_ma - p1_mi, p3_ma - p1_mi, Re_ma - Re_mi, Sc_ma - Sc_mi] * lhs(7, N_w)
    wall_dno = [d, -lf, p1_mi, p2_mi, p1_mi, Re_mi, Sc_mi] + \
               [L-d, 0, p1_ma - p1_mi, p2_ma - p1_mi, p3_ma - p1_mi, Re_ma - Re_mi, Sc_ma - Sc_mi] * lhs(7, N_w)
    wall_left = [0.0, -lf, p1_mi, p2_mi, p1_mi, Re_mi, Sc_mi] + \
                [0.0, d, p1_ma - p1_mi, p2_ma - p1_mi, p3_ma - p1_mi, Re_ma - Re_mi, Sc_ma - Sc_mi] * lhs(7, N_w)

    # Filter wall points to exclude fin regions
    wall_up = []
    wall_dn = []
    for x in wall_upo:
        if L0 < x[0] < L0+lf or L0+2*l < x[0] < L0+2*l+lf:
            pass
        else:
            wall_up.append(x)
    for x in wall_dno:
        if L0+l < x[0] < L0+l+lf or L0+3*l < x[0] < L0+3*l+lf:
            pass
        else:
            wall_dn.append(x)

    return np.array(wall_up), np.array(wall_dn), np.array(wall_left)

def generate_inlet_outlet():
    """Generate inlet and outlet boundary points"""
    inlet_1 = [x_min, -lf, p1_mi, p2_mi, p1_mi, Re_mi, Sc_mi] + \
              [d, 0.0, p1_ma - p1_mi, p2_ma - p1_mi, p3_ma - p1_mi, Re_ma - Re_mi, Sc_ma - Sc_mi] * lhs(7, N_i)
    inlet_2 = [x_min, lf, p1_mi, p2_mi, p1_mi, Re_mi, Sc_mi] + \
              [d, 0.0, p1_ma - p1_mi, p2_ma - p1_mi, p3_ma - p1_mi, Re_ma - Re_mi, Sc_ma - Sc_mi] * lhs(7, N_i)
    outlet = [L, -lf, p1_mi, p2_mi, p1_mi, Re_mi, Sc_mi] + \
             [0.0, d, p1_ma - p1_mi, p2_ma - p1_mi, p3_ma - p1_mi, Re_ma - Re_mi, Sc_ma - Sc_mi] * lhs(7, N_i)
    
    inout = np.stack((inlet_1, inlet_2, outlet), axis=2)
    return inout

def generate_sections():
    """Generate section points for penalty terms"""
    fsloc = (L0 + L0 + l + p1_ma)/2
    secs = []
    for i in range(-1, ndf):
        sec = [fsloc + l*i, -lf, p1_mi, p2_mi, p1_mi, Re_mi, Sc_mi] + \
              [0.0, d, p1_ma - p1_mi, p2_ma - p1_mi, p3_ma - p1_mi, Re_ma - Re_mi, Sc_ma - Sc_mi] * lhs(7, N_f)
        secs.append(sec)
    return np.array(secs)