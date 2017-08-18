import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
# from scipy.linalg import solve, lstsq
# from scipy.sparse.linalg import spsolve, lsqr
# from scipy.sparse import csr_matrix

from qutip import *

from datetime import datetime

__author__ = "Longfei Fan"
__version__ = "1.0"


class MasterEq(object):
    """
    To solve the ode equation given by
        \dot{p_m} = - G_m * (m+1) * p_m + G_{m-1} * m * p_{m-1}
                    - L_m * m * p_m + L_{m+1} * (m+1) * p_{m+1}
        G_m = A - B * (m+1)
        L_m = C
    which is a general form for master equastion for lasers and
    condensation N Boson (CNB) system.
    """
    
    def __init__(self, A, B, C, N_max=None):
        """
        Parameters
        ----------
        A: float
            gain coefficient
        B: float
            saturation coefficient
        C: float
            loss coefficient
        N_max: int
            Fock number used for numerical calculate will in [0, N_max - 1]
        """ 
        self.A = A
        self.B = B
        self.C = C
        self.N_max = N_max  
        
        # inital state
        self.init_state = None
        self.t_list = []
        self.rho_vs_t = []
        self.pn_vs_t = []
        self.nbar_vs_t = []
        self.entr_vs_t = []

        # self.steady_pn = None
        # self.steady_nbar = None
        # self.steady_entr = None

    def set_N_max(self, N_max):
        """
        set the truncated photon numbers for numerical calcualtions
        """
        self.N_max = N_max


    def set_t_list(self, t_list):
        """
        set the time points where data will be saved
        """
        self.t_list = t_list

    def set_init_state(self, init_state):
        """
        set the initial state
        """
        self.init_state = init_state

    def get_abc(self):
        """
        return A, B, C
        """
        return {'A': self.A, 'B': self.B, 'C': self.C}

    def get_init_state(self):
        """
        return the initial state
        """
        return self.init_state

    def get_t_list(self):
        """
        return t_list of ode
        """
        return self.t_list

    def get_pns(self):
        """
        return diagonal terms vs. time
        """
        return self.pn_vs_t
    
    def get_nbars(self):
        """
        return average photon numbers vs. time
        """
        return self.nbar_vs_t
    
    def get_entrs(self):
        """
        return entropy vs. time
        """
        return self.entr_vs_t

    # solve the ode for pn
    def pn_evolve(self, t_list):
        """ **ode solver for pn**
            init_pn: an array, initial diagonal terms of rho
            N_max: truncted photon numbers
            t_list: a list of time points to be calculated on
            diag: if rho only has diagonal terms, reconstruct rho
        """
        print(str(datetime.now()) + " START")

        self.t_list = t_list
        n_list = np.arange(self.N_max)
        
        # parameters
        g = np.array([self._gm(m) for m in n_list])
        l = np.array([self._lm(m) for m in n_list])
        
        # find diagonal terms
        if self.init_state.type is 'ket':
            init_state = ket2dm(self.init_state)
        init_pn = np.real(np.diag(init_state.data.toarray()))
        
        # solve the ode for pn
        self.pn_vs_t = odeint(self._pn_dot, init_pn, t_list, args=(g, l))

        # reconstruct rho from pn if only the main diagonal terms exist
        # self.rho_vs_t = np.array([Qobj(np.diag(pn)) for pn in self.pn_vs_t])
        
        # find average photon numbers
        self.nbar_vs_t = np.array([sum(pn * n_list) for pn in self.pn_vs_t])
        
        # find von Neumann entropy
        pn_vs_t = np.array([pn[pn > 0] for pn in self.pn_vs_t])
        self.entr_vs_t = np.array([- sum(pn * np.log(pn)) for pn in pn_vs_t])

        # check if the probability sums to 1
        # self.norm_vs_t = np.array([sum(pn) for pn in self.pn_vs_t])

        print(str(datetime.now()) + " FINISH")
        
    # G_m
    def _gm(self, m):
        return self.A - self.B * (m + 1)
    
    # L_m
    def _lm(self, m):
        return self.C
    
    # ordinary differential equation for diagonal terms only
    def _pn_dot(self, pn, t, g, l):
        """ ode update rule for pn
        """
        pn_new = np.zeros(self.N_max)
        
        for n in range(self.N_max):
            pn_new[n] -= (g[n] * (n + 1) + l[n] * n) * pn[n]
            if n > 0:
                pn_new[n] += g[n - 1] * n * pn[n - 1]
            if n < self.N_max - 1:
                pn_new[n] += l[n + 1] * (n + 1) * pn[n + 1] 
                
        return pn_new

#     def solve_steady_state(self):
#         """ If the state is always diagonal during evolution,
#             get the diagonal terms of the steady state.
#             Solver: `scipy.sparse.lina`
#         """
#         eq = np.zeros([self.N_max, self.N_max])
#         y = np.repeat(np.finfo(float).eps, self.N_max)
#         # cannot be done numerically if all set to be zero
#         # y = np.repeat(0, self.N_max)

#         for k in range(self.N_max):
#             eq[k, k] = self._fnm(k, k)
#             if k < self.N_max - 1:
#                 eq[k, k + 1] = self._hnm(k, k)
#             if k > 0:
#                 eq[k, k - 1] = self._gnm(k, k)        
#         pn = spsolve(csr_matrix(eq), y)
        
#         pn = pn/sum(pn)
#         n = sum(pn * range(self.N_max))
#         entr = sum([- p * np.log2(p) for p in pn if p > 0])
        
#         return pn, n, entr

    def plot_n_vs_time(self, time_factor=1, plot_only=True):
        """ Plot average Fock numbers with respect to time
        """
        if len(self.nbar_vs_t) == 0:
            print("Solve the master equation first before plotting!")
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.t_list * time_factor, self.nbar_vs_t)
        ax.set_xlabel("$time$", fontsize=14)
        ax.set_ylabel("average Fock number", fontsize=14)
        ax.set_title("Average Fock Numbers vs. Time", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        if not plot_only:
            return fig, ax

    def plot_entropy_vs_time(self, time_factor=1, plot_only=True):
        """ Plot von Neumann entropy with respect to time
        """
        if len(self.nbar_vs_t) == 0:
            print("Solve the master equation first before plotting!")
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.t_list * time_factor, self.entr_vs_t)
        ax.set_xlabel("$time$", fontsize=14)
        ax.set_ylabel("von Neumann entropy", fontsize=14)
        ax.set_title("von Neumann Entropy vs. Time", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        if not plot_only:
            return fig, ax


class LaserABC(MasterEq):
    """
    One mode lasers, defined by coefficients of A, B, and C
    g_m is now changed to the full expression without approximations.
    """
    def __init__(self, A, B, C, N_max=None):
        super(LaserABC, self).__init__(A, B, C, N_max)

    def nbar_above_threshold(self):
        """ 
        Analytic approximaiton of the average photon number for a steady
        laser operated **above the threshold**
        """
        return self.A * (self.A - self.kappa) / self.kappa / self.B

    # G_m, full expression without approximation, overriding the original one
    def _gm(self, m):
        """
        G_m defined specifically for one mode lasers
        """
        return self.A ** 2 / (self.A + self.B * (m + 1))


class Laser(MasterEq):
    """ 
    Quantum model of one mode laser discussed in Chapter 11 of Qunatum Optics
    by Scully and Zubairy
    """
    def __init__(self, g, ra, gamma, kappa, N_max=None):
        """
        One mode laser
        
        Parameters
        ----------
        g: float
            atom-cavity interation strength
        ra: float
            pumping rate
        gamma: float
            atom damping rate
        kappa: float
            cavity damping rate
        """
        self.g = g
        self.ra = ra
        self.gamma = gamma
        super(Laser, self).__init__(2 * ra * g**2 / gamma**2, 
            8 * ra * g**4 / gamma**4, kappa, N_max)


    def get_atom_cavity_args(self):
        """
        return the setup parameters for the atom and cavity
        """
        return {'g': self.g, 'ra': self.ra, 'gamma': self.gamma, 'kappa': self.C}

    def nbar_above_threshold(self):
        """ 
        Analytic approximaiton of the average photon number for a steady
        laser operated **above the threshold**
        """
        return self.A * (self.A - self.kappa) / self.kappa / self.B

    # G_m, overriding the original one
    def _gm(self, m):
        """
        G_m defined specifically for lasers
        """
        return self.A ** 2 / (self.A + self.B * (m + 1))


class CNBoson(MasterEq):
    """
    Condensation N Bosons (CNB) in a 3D harmonic trap
    """
    def __init__(self, N, TTc, rate, eta):
        """
        Condensation N Bosons in a 3D harmonic trap

        Parameters
        ----------
        """
        self.N = N
        self.TTc = TTc
        self.rate = rate
        self.eta = eta
        super(CNBoson, self).__init__(rate * (1 + eta) * (N + 1), 
            rate * (1 + eta), rate * N * TTc ** 3, N + 1)


    def get_cnb_args(self):
        """
        return the parameters for the CNB
        """
        return {'N': self.N, 'T/Tc': self.TTc, 'rate constant':self.rate,
                'cross excitation': self.eta}


    def _lm(self, m):
        """
        L_m: loss coefficient sepecifically for CNB
        """
        return self.C + self.rate * (self.N - m) * self.eta
