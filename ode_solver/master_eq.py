import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

from scipy.integrate import odeint, complex_ode
# from scipy.linalg import solve, lstsq
from scipy.sparse.linalg import spsolve, lsqr
# from scipy.stats import poisson
from scipy.sparse import csr_matrix


from qutip import *

__author__ = "Longfei Fan"
__version__ = "1.0"


class MasterEq(object):
    
    def __init__(self, A, B, C, N_max=None):
        """
        Master equation solver with gain, saturation, and loss parameters

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
        # self.A = kappa * (1 + eta) * (N + 1)
        # self.B = kappa * (1 + eta)
        # self.C = kappa * N * TTc**3
        
        # inital state
        self.init_state = None
        self.t_list = []
        self.rho_vs_t = []
        self.pn_vs_t = []
        self.nbar_vs_t = []
        self.entr_vs_t = []
        
        self.steady_pn = None
        self.steady_nbar = None
        self.steady_entr = None

    def set_n_max(self, N_max):
        """
        set the truncated photon numbers for numerical calcualtions
        """
        self.N_max = N_max

    # def get_cnb_args(self):
    #     """
    #     return the setup parameters for the atom and cavity
    #     """
    #     return {'N': self.N, 'rate constant': self.kappa, 
    #             'T/T_c': self.TTc, 'cross excitation': self.eta}

    def get_abc(self):
        """
        return A, B, C
        """
        return {'A': self.A, 'B': self.B, 'C': self.C}

    def get_tlist(self):
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
    def pn_evolve(self, init_state, t_list):
        """ **ode solver for pn**
            init_pn: an array, initial diagonal terms of rho
            N_max: truncted photon numbers
            t_list: a list of time points to be calculated on
            diag: if rho only has diagonal terms, reconstruct rho
        """
        self.t_list = t_list
        n_list = np.arange(self.N_max)
        
        # parameters
        g = np.array([self._gm(m) for m in n_list])
        l = np.array([self._lm(m) for m in n_list])
        
        # find diagonal terms
        if init_state.type is 'ket':
            init_state = ket2dm(init_state)
        init_pn = np.real(np.diag(init_state.data.toarray()))
        
        # solve the ode for pn
        self.pn_vs_t = odeint(self._pn_dot, init_pn, t_list, args=(g, l))

        # reconstruct rho from pn if only the main diagonal terms exist
        # self.rho_vs_t = np.array([Qobj(np.diag(pn)) for pn in self.pn_vs_t])
        
        # find average photon numbers
        self.n_vs_t = np.array([sum(pn * n_list) for pn in self.pn_vs_t])
        
        # find von Neumann entropy
        pn_vs_t = np.array([pn[pn > 0] for pn in self.pn_vs_t])
        self.entr_vs_t = np.array([- sum(pn * np.log(pn)) for pn in pn_vs_t])
        
        
    # G_m
    def _gm(self, m):
        return self.A - self.B * (m + 1)
    
    # L_m
    def _lm(self, m):
        return self.C + self.kappa * (self.N - m) * self.eta
    
        
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

    def plot_n_vs_time(self):
        """ Plot average photon numbers with respect to time
        """
        if len(self.n_vs_t) == 0:
            print("Solve the evolution equation first to obtain average photon numbers!")
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.t_list * self.kappa, self.n_vs_t)
        ax.set_xlabel("$\kappa t~(\kappa * time)$", fontsize=14)
        ax.set_ylabel("average photon number", fontsize=14)
        ax.set_title("Average Photon Number vs. Time", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        return fig, ax

    def plot_entropy_vs_time(self):
        """ Plot von Neumann entropy of the cavity field with respect to time
        """
        # if len(self.entr_vs_t) != len(self.t_list):
        #     self._calc_entropy()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.t_list * self.kappa, self.entr_vs_t)
        ax.set_xlabel("$\kappa t~(\kappa * time)$", fontsize=14)
        ax.set_ylabel("von Neumann entropy", fontsize=14)
        ax.set_title("von Neumann Entropy vs. Time", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        return fig, ax