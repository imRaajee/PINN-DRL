"""
Generation of collocation points for PINN training
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.interpolate import CubicSpline
from config import *

def interpolating_curve(t, p0, p1, p2, p3, p4):
    """Create interpolating curve using cubic spline"""
    xs = np.array([p0[0], p1[0], p2[0], p3[0], p4[0]])
    ys = np.array([p0[1], p1[1], p2[1], p3[1], p4[1]])
    cs = CubicSpline(xs, ys)
    return np.array([t, cs(t)])

def generate_collocation_points():
    """Generate collocation points within the domain"""
    # Generate initial collocation points
    colo = lb + (ub - lb) * lhs(7, N_c)
    
    # Add refined regions around the curves
    colo_1 = [L0, 0.0, p1_mi, p2_mi, p1_mi, Re_mi, Sc_mi] + \
             [lf, y_max, p1_ma - p1_mi, p2_ma - p1_mi, p3_ma - p1_mi, Re_ma - Re_mi, Sc_ma - Sc_mi] * lhs(7, N_r)
    colo_2 = [L0 + 2*l, 0.0, p1_mi, p2_mi, p1_mi, Re_mi, Sc_mi] + \
             [lf, y_max, p1_ma - p1_mi, p2_ma - p1_mi, p3_ma - p1_mi, Re_ma - Re_mi, Sc_ma - Sc_mi] * lhs(7, N_r)
    colo_3 = [L0 + l, 0.0, p1_mi, p2_mi, p1_mi, Re_mi, Sc_mi] + \
             [lf, -y_max, p1_ma - p1_mi, p2_ma - p1_mi, p3_ma - p1_mi, Re_ma - Re_mi, Sc_ma - Sc_mi] * lhs(7, N_r)
    colo_4 = [L0 + 3*l, 0.0, p1_mi, p2_mi, p1_mi, Re_mi, Sc_mi] + \
             [lf, -y_max, p1_ma - p1_mi, p2_ma - p1_mi, p3_ma - p1_mi, Re_ma - Re_mi, Sc_ma - Sc_mi] * lhs(7, N_r)
    
    colo = np.concatenate((colo, colo_1, colo_2, colo_3, colo_4), axis=0)
    
    # Filter points to be within the domain
    col = []
    for x in colo:
        p1 = np.array([0.125, x[2]])
        p2 = np.array([0.25, x[3]])
        p3 = np.array([0.375, x[4]])
        
        # Upper curves (positive y)
        if (L0 < x[0] < L0 + lf) and (x[1] > 0):
            t = (x[0] - L0)
            curve_y = interpolating_curve(t, p0, p1, p2, p3, p4)[1]
            if x[1] < curve_y + hf:
                col.append(x)
        elif (L0 + 2*l < x[0] < L0 + 2*l + lf) and (x[1] > 0):
            t = (x[0] - L0 - 2*l)
            curve_y = interpolating_curve(t, p0, p1, p2, p3, p4)[1]
            if x[1] < curve_y + hf:
                col.append(x)
        
        # Lower curves (negative y)  
        elif (L0 + l < x[0] < L0 + l + lf) and (x[1] < 0):
            t = (x[0] - L0 - l)
            curve_y = interpolating_curve(t, p0, p1, p2, p3, p4)[1]
            if x[1] > -curve_y - hf:
                col.append(x)
        elif (L0 + 3*l < x[0] < L0 + 3*l + lf) and (x[1] < 0):
            t = (x[0] - L0 - 3*l)
            curve_y = interpolating_curve(t, p0, p1, p2, p3, p4)[1]
            if x[1] > -curve_y - hf:
                col.append(x)
        
        # Central channel region
        elif (-hf < x[1] < hf):
            col.append(x)
    
    return np.array(col)

def plot_collocation_points(col):
    """Plot the generated collocation points"""
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.scatter(col[:, 0], col[:, 1], marker='.', alpha=0.1, color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Collocation Points')
    plt.show()

if __name__ == "__main__":
    # Generate and plot collocation points
    collocation_points = generate_collocation_points()
    print(f"Generated {np.shape(collocation_points)} collocation points")
    plot_collocation_points(collocation_points)