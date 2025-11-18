"""
FlexPINN - Main PINN training class
"""

import time
import torch
from networks import CombinedNetwork
from pde_losses import LossFunctions
from utils import plotLoss, save_losses
from constants import *
from config import device, ndf

class PINN:
    """FlexPINN - Physics-Informed Neural Network with flexible architecture"""
    
    def __init__(self):
        self.net = CombinedNetwork().to(device)
        self.loss_fn = LossFunctions(self.net)
        
        # Optimizers
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * torch.finfo(torch.float32).eps,
            history_size=100,
            line_search_fn="strong_wolfe",
        )
        
        self.adam = torch.optim.Adam(self.net.parameters(), lr=5e-4)
        
        # Training tracking
        self.losses = {"bc": [], "inout": [], "pde": [], "help": []}
        self.iter = 0

    def predict(self, xyp):
        """Forward pass through the network"""
        return self.loss_fn.predict(xyp)

    def closure(self, bc, wall_Jx, wall_Jy, Fins, Normals, inout, col, secs):
        """LBFGS closure function"""
        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        # Compute individual losses
        mse_bc = self.loss_fn.bc_loss(bc, wall_Jx, wall_Jy, Fins, Normals)
        mse_inout = self.loss_fn.inout_loss(inout)
        mse_pde = self.loss_fn.pde_loss(col)
        mse_help = self.loss_fn.help_loss(secs, ndf)

        # Adaptive weighting
        total = mse_bc + mse_inout + mse_pde + mse_help
        wbc = mse_bc / total
        wio = mse_inout / total
        wpde = mse_pde / total
        wh = mse_help / total

        # Weighted total loss
        loss = ((1+wbc)**4)*mse_bc + ((1+wio)**4)*mse_inout + ((1+wpde)**4)*mse_pde + ((1+wh)**4)*mse_help
        loss.backward()

        # Store losses
        self.losses["bc"].append(mse_bc.detach().cpu().item())
        self.losses["inout"].append(mse_inout.detach().cpu().item())
        self.losses["pde"].append(mse_pde.detach().cpu().item())
        self.losses["help"].append(mse_help.detach().cpu().item())

        self.iter += 1

        print(
            f"\r It: {self.iter} Loss: {loss.item():.5e} BC: {mse_bc.item():.3e} "
            f"inout: {mse_inout.item():.3e} pde: {mse_pde.item():.3e} help: {mse_help.item():.3e}",
            end="",
        )

        if self.iter % 500 == 0:
            torch.save(self.net.state_dict(), "Param.pt")
            print("")

        return loss

    def train(self, bc, wall_Jx, wall_Jy, Fins, Normals, inout, col, secs):
        """Train the PINN"""
        start_time = time.time()

        # Adam training
        for i in range(AD_EP):
            self.closure(bc, wall_Jx, wall_Jy, Fins, Normals, inout, col, secs)
            self.adam.step()
        
        # LBFGS training
        self.lbfgs.step(lambda: self.closure(bc, wall_Jx, wall_Jy, Fins, Normals, inout, col, secs))

        print("--- %s seconds ---" % (time.time() - start_time))
        print(f'-- {(time.time() - start_time)/60} mins --')
        
        # Save final model and results
        torch.save(self.net.state_dict(), "Param.pt")
        plotLoss(self.losses, "LossCurve.png", ["BC", "InOut", "PDE", "Help"])
        save_losses(self.losses)