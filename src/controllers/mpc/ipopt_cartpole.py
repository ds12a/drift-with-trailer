import casadi as ca

class MPC:
    """
    Generalized MPC class that uses direct shooting with Euler Method
    """

    def __init__(self, x_d, u_d, dynamics_func, term_constraint_func, constraint_func, term_cost_func, cost_func, ipopt_settings, step=0.02*2, n=20):
        """
        Args:
            x_d (int): State vector dimensions
            u_d (int): Control vector dimensions
            dynamics_func (Callable): Dynamics function that returns x_dot
            term_constraint_func (Callable[[ca.Opti, SX, SX, SX], None]): Terminal constraint function t(opti, x(t_f), u(0), u(t_f))
            constraint_func (Callable[[ca.Opti, SX, SX], None]): Constraint function c(opti, x(t), u(t))
            term_cost_func (Callable[[SX, SX, SX], SX]): Terminal cost function t(x(t_f), u(0), u(t_f))
            cost_func (Callable[[SX, SX], SX]): Constraint function c(x(t), u(t))
            ipopt_settings (dict):
            step (float, optional): Timestep for controller. Defaults to 0.02*2.
            n (int, optional): Number of timesteps (lookahead) for MPC solve. Defaults to 20.
        """        
        self.last_trajectory = None
        self.lt_lam_g = None

        self.dynamics = dynamics_func
        self.constraint = constraint_func
        self.term_constraint = term_constraint_func
        self.term_cost = term_cost_func
        self.cost = cost_func

        self.opti = ca.Opti()
        
        self.x0 = self.opti.parameter(x_d)  # Params for running MPC opt
        self.x_d = x_d
        self.u_d = u_d
        self.n = n

        self.step = step
        self.dynamics = dynamics_func()  # ca.Function for efficiency
        self.u_sym = self.setup_mpc()



        self.ipopt_settings = ipopt_settings

        ws = {
            'ipopt.warm_start_init_point': 'yes',
            # 'ipopt.mu_init': 1e-1,
            'ipopt.warm_start_bound_push': 3e-2,
            'ipopt.warm_start_slack_bound_frac': 3e-2,
            'ipopt.warm_start_slack_bound_push': 3e-2,
            'ipopt.warm_start_mult_bound_push': 3e-2,
        }
        self.ipopt_ws_settings = ipopt_settings | ws
        # self.opti.solver("ipopt", ipopt_settings)


    def _euler_step(self, x, u):
        """
        Uses Euler's method to integrate the dynamics

        Args:
            x (ca.SX): States
            u (ca.SX): Controls

        Returns:
            ca.SX: Next state
        """        
        return x + self.dynamics(x, u) * self.step

    def setup_mpc(self):
        """
        Formulates problem, sets constraints

        Returns:
            ca.SX: Controls (u)
        """        
        x = self.x0
        u = self.opti.variable(self.n, self.u_d)
        J = 0

        for i in range(self.n):
            u_i = u[i, :].T
            x = self._euler_step(x, u_i)

            J += self.cost(x, u_i)
            self.constraint(self.opti, x, u_i)

        if self.term_cost:
            J += self.term_cost(x, u[0, :].T, u[-1, :].T)

        self.opti.minimize(J)

        if self.term_constraint:
            self.term_constraint(self.opti, x, u[0, :].T, u[-1, :].T)

        return u

    def run_mpc(self, x, verbose=True, warm_start=True):
        """
        Runs a single MPC solve.
        Args:
            x (np.ndarray): Measured stat
            verbose (bool, optional): _description_. Defaults to True.
            warm_start (bool, optional): _description_. Defaults to True.

        Returns:
            ca.SX: u
        """
        dual_warm_start = warm_start and self.last_trajectory is not None and self.u_d == 1
        self.opti.solver(
            "ipopt",
            self.ipopt_ws_settings if dual_warm_start else self.ipopt_settings,
        )

        if self.last_trajectory is not None and warm_start:
            self.opti.set_initial(self.u_sym,  ca.vertcat(self.last_trajectory[1:], self.last_trajectory[-1:]))
            if dual_warm_start:
                self.opti.set_initial(self.opti.lam_g, ca.vertcat(self.lt_lam_g[1:], self.lt_lam_g[-1]))
        
        self.opti.set_value(self.x0, x)

        sol = self.opti.solve()
        
        if verbose:
            stats = sol.stats()
            print(f"Solve iteration succeeded in {stats['iter_count']} iterations in {stats['t_wall_total']} s")

        u = sol.value(self.u_sym)[0]

        if warm_start:
            self.last_trajectory = sol.value(self.u_sym)
            self.lt_lam_g = sol.value(self.opti.lam_g)
            
        
        return u
