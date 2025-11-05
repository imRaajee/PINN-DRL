"""
Baffle (Fin) geometry and normal vector generation
"""

import numpy as np
from pyDOE import lhs
from scipy.interpolate import CubicSpline
from config import *

def generate_baffles_and_normals():
    """Generate baffle points and their normal vectors"""
    # Generate fin parameters
    Fin_params = [p1_mi, p2_mi, p1_mi, Re_mi, Sc_mi] + \
                 [p1_ma - p1_mi, p2_ma - p1_mi, p3_ma - p1_mi, Re_ma - Re_mi, Sc_ma - Sc_mi] * lhs(5, N_f)
    t = lhs(1, N_f)

    Fin1 = []
    Fin2 = []
    Normals = []

    for i in range(N_f):
        p1, p2, p3 = Fin_params[i, :3]
        Re, Sc = Fin_params[i, 3:]
        
        # Create cubic spline for fin curve
        xs = np.array([p0[0], 0.125, 0.25, 0.375, p4[0]])
        ys = np.array([p0[1], p1, p2, p3, p4[1]])
        
        cs = CubicSpline(xs, ys)
        
        # Calculate point on curve
        x = (t[i] * (p4[0] - p0[0]) + p0[0])[0]
        y = cs(x)
        
        # Calculate tangent and normal vectors
        dy_dx = cs.derivative()(x)
        tangent = np.array([1.0, dy_dx])
        tangent /= np.linalg.norm(tangent)
        
        normal = np.array([-tangent[1], tangent[0]])
        
        # Create fin points at different locations
        normal = np.array([-tangent[1], tangent[0]])
        if -0.5 < y < 0.5:
            Fin1.append([x + L0, y + lf, p1, p2, p3, Re, Sc])
            Fin2.append([x + L0 + l, y - lf, p1, p2, p3, Re, Sc])
            Normals.append(normal)

    Fin1 = np.array(Fin1)
    Fin2 = np.array(Fin2)
    
    # Create additional fins by translation
    Fin3 = Fin1 + [2*l, 0, 0, 0, 0, 0, 0]
    Fin4 = Fin2 + [2*l, 0, 0, 0, 0, 0, 0]
    Normals = np.array(Normals)

    # Combine all fins and normals
    Fins = np.concatenate((Fin1, Fin2, Fin3, Fin4), axis=0)
    All_Normals = np.concatenate((Normals, -Normals, Normals, -Normals), axis=0)
    
    return Fins, All_Normals

def plot_baffles(Fins):
    """Plot baffle points for visualization"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    
    # Plot different fins with different colors
    N_per_fin = len(Fins) // 4
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['Fin1 (Upper Left)', 'Fin2 (Lower Left)', 'Fin3 (Upper Right)', 'Fin4 (Lower Right)']
    
    for i in range(4):
        start_idx = i * N_per_fin
        end_idx = (i + 1) * N_per_fin
        plt.scatter(Fins[start_idx:end_idx, 0], Fins[start_idx:end_idx, 1], 
                   marker='.', alpha=0.5, color=colors[i], label=labels[i])
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Baffle Points')
    plt.legend()
    plt.show()