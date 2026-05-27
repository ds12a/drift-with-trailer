import torch
from scipy.signal import savgol_filter
from scipy.stats import norm
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal



class SMPPI_Torch:
    """
    Torch SMPPI
    """

    def __init__(
        self,
        x_d: int,
        u_d: int,
        dynamics_func,
        term_cost_func,
        cost_func,
        bound_control_func,
        cv,
        omega,
        inverse_temp=1,
        alpha=0.01,
        gamma=0.01,
        K=20000,
        step=0.02,
        T=70,
        device="mps"
    ):
        """
        Args:
            x_d (int): State dimension
            u_d (int): Control dimension
            dynamics_func (Callable): Dynamics function
            term_cost_func (Callable): Terminal cost function
            cost_func (Callable): Cost function
            bound_control_func (Callable): Function that bounds controls
            inverse_temp (int, optional): Actually the temperature. Defaults to 1.
            alpha (float, optional): Proportion of samples set to just noise. Defaults to 0.01.
            gamma (float, optional): Cost weight for pertubations. Defaults to 0.1.
            K (int, optional): Samples. Defaults to 5000.
            step (float, optional): Time step. Defaults to 0.02.
            T (int, optional): Time horizon in steps. Defaults to 50.
        """          
        self.last_trajectory = None
        self.dynamics = dynamics_func
        self.term_cost = term_cost_func
        self.cost = cost_func
        self.bound_control = bound_control_func
        self.alpha = alpha
        self.inverse_temp = inverse_temp
        self.gamma = gamma
        self.omega = omega
        self.K = K
        
        self.device = torch.device(device)
        
        self.x_d = x_d
        self.u_d = u_d
        self.T = T

        self.step = step
        self.cv = cv
        
        self.inv_cv = torch.inverse(self.cv)

        self.dist = MultivariateNormal(torch.zeros(self.u_d, device=device), self.cv)


    def _forward_sim(self, x: torch.Tensor, u: torch.Tensor, a: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Uses Euler's method to integrate the dynamics

        Args:
            x (torch.Tensor): State (T, K, x_d)
            u (torch.Tensor): Control (T, K, u_d)
            a (torch.Tensor): Previous action
            noise (torch.Tensor): Noise (T, K, u_d)

        Returns:
            torch.Tensor: Cost per sample (K)
        """        

        v = u + noise
        new_a = a + v
        # a = a0.repeat(self.K, 1)

        prev = round(self.K * (1 - self.alpha))
        v[:, prev:] = noise[:, prev:]
        new_a = self.bound_control(new_a)
        noise.copy_(new_a - a - u)

        S = torch.zeros(self.K, device=self.device)

        for i in range(self.T):
            x += self.dynamics(x, new_a[i, :]) * self.step
            
            # print(u.device, self.inv_cv.device, noise.device)
            S += (
                self.cost(x, new_a[i, :], i)
                + self.gamma
                * (u[i, :].unsqueeze(1) @ self.inv_cv @ noise[i, :].unsqueeze(2)).squeeze(-1).squeeze(-1)
            )

        if self.term_cost:
            S += self.term_cost(x, u[-1, :])

        S += self._smoothing_cost(new_a)

        return S

    def _smoothing_cost(self, a: torch.Tensor) -> torch.Tensor:
        diff = a - torch.roll(a, 1, dims=0)
 
        return torch.einsum("tbn,nn,tbn->b", diff, self.omega, diff)

    def _weights(self, costs: torch.Tensor) -> torch.Tensor:
        """
        Computes weights

        Args:
            costs (torch.Tensor): Costs (K)

        Returns:
            torch.Tensor: Weights (K)
        """        
        weights = torch.exp(-(costs - costs.min()) / self.inverse_temp)
        return weights / weights.sum()
    

    def run_mpc(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs a single MPC solve.

        Args:
            x (torch.Tensor): State (x_d)

        Returns:
            torch.Tensor: Control output
        """        

        u = torch.zeros(self.T, self.u_d, device=self.device)
        a = torch.zeros(self.T, self.u_d, device=self.device)

        if self.last_trajectory is not None:
            # u[:-1] = self.last_trajectory[1:]
            u_last, a_last = self.last_trajectory
            u[:-1] = u_last[1:]
            a[:-1] = a_last[1:]

        x = torch.from_numpy(x).to(self.device)

        x_batch = x.unsqueeze(0).repeat(self.K, 1)
        u_batch = u.unsqueeze(0).repeat(self.K, 1, 1).permute(1, 0, 2)
        a_batch = a.unsqueeze(0).repeat(self.K, 1, 1).permute(1, 0, 2)

        noise = self.dist.sample(u_batch.shape[:-1])

        costs = self._forward_sim(x_batch, u_batch, a_batch, noise)

        weights = self._weights(costs)
        weighted_noise = torch.sum(weights.view(1, -1, 1) * noise, dim=1)
        u += weighted_noise
        a += u
        self.last_trajectory = (u, a)
        return a[0].cpu().numpy()
