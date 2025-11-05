"""
PDE, boundary condition, and physics loss functions
"""

import torch
from torch.autograd import grad
from constants import *

class LossFunctions:
    """Collection of loss functions for PINN training"""
    
    def __init__(self, network):
        self.network = network

    def predict(self, xyp):
        """Get all output variables from the network"""
        out = self.network(xyp)
        u = out[0]
        v = out[1]
        p = out[2]
        sig_xx = out[3]
        sig_xy = out[4]
        sig_yy = out[5]
        c = out[6]
        J_x = out[7]
        J_y = out[8]
        return u, v, p, sig_xx, sig_xy, sig_yy, c, J_x, J_y

    def bc_loss(self, bc_uv, wall_Jx, wall_Jy, Fins, Normals):
        """Boundary condition loss"""
        u, v = self.predict(bc_uv)[0:2]
        Jx = self.predict(wall_Jx)[7]
        Jy = self.predict(wall_Jy)[8]
        Jxf = self.predict(Fins)[7]
        Jyf = self.predict(Fins)[8]

        mse_uv = torch.mean(torch.square(u - bc_uv[:, 0:1])) + torch.mean(
            torch.square(v - bc_uv[:, 1:2]))
        mse_J = torch.mean(torch.square(Jx)) + torch.mean(torch.square(Jy)) + torch.mean(
            torch.square(Jxf[:, 0] * Normals[:, 0] + Jyf * Normals[:, 1]))

        return mse_uv + mse_J

    def inout_loss(self, inout):
        """Inlet/outlet boundary loss"""
        in1 = self.network(inout[:, :, 0])
        in2 = self.network(inout[:, :, 1])
        out = self.network(inout[:, :, 2])

        p = out[2]
        Jx = out[7]
        c1 = in1[6]
        c2 = in2[6]

        mse_in1 = torch.mean(torch.square(c1))
        mse_in2 = torch.mean(torch.square(c2 - cmax))
        mse_out = torch.mean(torch.square(p)) + torch.mean(torch.square(Jx))

        return mse_in1 + mse_in2 + mse_out

    def help_loss(self, secs, ndf):
        """Auxiliary loss for sections"""
        mse_help = 0.0
        for i in range(ndf + 1):
            u = self.predict(secs[i, :, :])[0]
            c = self.predict(secs[i, :, :])[6]
            mse_help += torch.mean(torch.square(torch.mean(u, dim=0) - umax)) + torch.mean(
                torch.square(torch.mean(c, dim=0) - cmax/2))
        return mse_help

    def pde_loss(self, xyp):
        """Physics loss (PDE residuals)"""
        xyp = xyp.clone()
        xyp.requires_grad = True

        u, v, p, sig_xx, sig_xy, sig_yy, c, J_x, J_y = self.predict(xyp)

        # Denormalize Re and Sc
        Re = xyp[:, 5:6] * (re_bnd[1] - re_bnd[0]) + re_bnd[0]
        Sc = xyp[:, 6:7] * (sc_bnd[1] - sc_bnd[0]) + sc_bnd[0]

        # Compute gradients
        u_out = grad(u.sum(), xyp, create_graph=True)[0]
        v_out = grad(v.sum(), xyp, create_graph=True)[0]
        sig_xx_out = grad(sig_xx.sum(), xyp, create_graph=True)[0]
        sig_xy_out = grad(sig_xy.sum(), xyp, create_graph=True)[0]
        sig_yy_out = grad(sig_yy.sum(), xyp, create_graph=True)[0]
        c_out = grad(c.sum(), xyp, create_graph=True)[0]
        J_x_out = grad(J_x.sum(), xyp, create_graph=True)[0]
        J_y_out = grad(J_y.sum(), xyp, create_graph=True)[0]

        # Extract partial derivatives
        u_x, u_y = u_out[:, 0:1], u_out[:, 1:2]
        v_x, v_y = v_out[:, 0:1], v_out[:, 1:2]
        sig_xx_x, sig_xy_x = sig_xx_out[:, 0:1], sig_xy_out[:, 0:1]
        sig_xy_y, sig_yy_y = sig_xy_out[:, 1:2], sig_yy_out[:, 1:2]
        c_x, c_y = c_out[:, 0:1], c_out[:, 1:2]
        J_x_x, J_y_y = J_x_out[:, 0:1], J_y_out[:, 1:2]

        # Continuity equation
        f0 = u_x + v_y

        # Momentum equations
        f1 = (u * u_x + v * u_y) - sig_xx_x - sig_xy_y
        f2 = (u * v_x + v * v_y) - sig_xy_x - sig_yy_y

        # Stress relations
        f3 = -p + (2 / Re) * u_x - sig_xx
        f4 = -p + (2 / Re) * v_y - sig_yy
        f5 = (1 / Re) * (u_y + v_x) - sig_xy

        # Concentration equations
        f6 = (u * c_x + v * c_y) - (1 / (Re * Sc)) * (J_x_x + J_y_y)
        f7 = J_x - c_x
        f8 = J_y - c_y

        # Combine all residuals
        mse_f0 = torch.mean(torch.square(f0))
        mse_f1 = torch.mean(torch.square(f1))
        mse_f2 = torch.mean(torch.square(f2))
        mse_f3 = torch.mean(torch.square(f3))
        mse_f4 = torch.mean(torch.square(f4))
        mse_f5 = torch.mean(torch.square(f5))
        mse_f6 = torch.mean(torch.square(f6))
        mse_f7 = torch.mean(torch.square(f7))
        mse_f8 = torch.mean(torch.square(f8))

        return (mse_f0 + mse_f1 + mse_f2 + mse_f3 + mse_f4 + mse_f5) + (mse_f6 + mse_f7 + mse_f8)