"""
Utility functions for PINN training
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plotLoss(losses_dict, path, info=["B.C.", "inout", "P.D.E.", "Help"]):
    """Plot training losses"""
    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(10, 6))
    axes[0].set_yscale("log")
    for i, j in zip(range(4), info):
        axes[i].plot(losses_dict[j.lower()])
        axes[i].set_title(j)
    plt.show()
    fig.savefig(path)

def save_losses(losses, prefix=''):
    """Save loss data to Excel files"""
    filepaths = {
        'bc': f'{prefix}bc.xlsx',
        'pde': f'{prefix}pde.xlsx', 
        'inout': f'{prefix}inout.xlsx',
        'help': f'{prefix}help.xlsx'
    }
    
    for key, path in filepaths.items():
        pd.DataFrame(np.array(losses[key])).to_excel(path, index=True)