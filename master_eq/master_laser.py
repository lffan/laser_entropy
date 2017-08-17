import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint, complex_ode
# from scipy.linalg import solve, lstsq
from scipy.sparse.linalg import spsolve, lsqr
from scipy.sparse import csr_matrix


from qutip import *
from master_eq import *

__author__ = "Longfei Fan"
__version__ = "1.0"


class Laser(MasterEq):
    """ 
    Numerical simulation of laser given on the equation of motion for the 
    density matrix of the cavity field in Chapter 11 of Qunatum Optics
    by Scully and Zubairy 
    """
    
    def __init__(self, g, ra, gamma, kappa, N_max=None):
        """

        w_c: cavity frequency
            w_a: atom frequency
            g: atom-cavity interation strength
            ra: pumping rate
            gamma: atom damping rate
            kappa: cavity damping rate
            init_state: initial cavity state (an qutip.Qobj object)
            the atom is assumed in the ground state at the beginning
        """
        self.g = g
        self.ra = ra
        self.gamma = gamma
        self.kappa = kappa
        super(Laser, self).__init__(2 * ra * g**2 / gamma**2, 
            8 * ra * g**4 / gamma**4, kappa, N_max)


    def get_atom_cavity_args(self):
        """ return the setup parameters for the atom and cavity
        """
        return {'g': self.g, 'ra': self.ra, 'gamma': self.gamma, 'kapa': self.kappa}


    def nbar_above_threshold(self):
        """ Calculate the average photon number in steady state
            for a laser operated **above threshold** (analytic approximaiton)
        """
        return self.A * (self.A - self.kappa) / self.kappa / self.B

    # G_m, override 
    def _gm(self, m):
        return 1 / (self.A + self.B * (m + 1))


    # solve the ode for rho
    # def rho_evolve(self, init_rho, N_max, t_list):
    #     """ **ode solver for the density matrix rho**
    #         init_rho: initial density matrix given as a quitp.Qobj
    #         N_max: truncted photon numbers
    #         t_list: a list of time points to be calculated on
    #     """
    #     self.N_max = N_max
    #     self.t_list = t_list
    #     self.init_state = init_rho
    #     n_list = np.arange(N_max)
        
    #     # if ket state, convert to density matrix, then to a 1-d array
    #     if init_rho.type == 'ket':
    #         init_rho = ket2dm(init_rho)
    #     init_array = np.real(init_rho.data.toarray().reshape(-1))          
        
    #      # parameters
    #     f = np.array([self._fnm(i, j) for i in n_list          
    #                   for j in n_list]).reshape(N_max, N_max)
    #     g = np.array([self._gnm(i, j) for i in n_list 
    #                   for j in n_list]).reshape(N_max, N_max)
    #     h = np.array([self._hnm(i, j) for i in n_list 
    #                   for j in n_list]).reshape(N_max, N_max)

    #     # sovle the ode
    #     print('ode solver started')
    #     self.arrays = odeint(self._rho_nm_dot, init_array, t_list, args=(f, g, h, ))
    #     print('ode solver finished')

    #     # convert arrays back to density matrices (rhos)
    #     rho_vs_t = np.array([Qobj(a.reshape(self.N_max, self.N_max)) 
    #                           for a in self.arrays])
        
    #     # find von Neumann entropy
    #     self.entr_vs_t = np.array([entropy_vn(rho) for rho in rho_vs_t])
        
    #     # find diagonal terms
    #     self.pn_vs_t = np.array([np.real(np.diag(rho.data.toarray())) for rho in rho_vs_t])
        
    #     del rho_vs_t
    #     print('rho deleted')
        
    #     # find average photon numbers
    #     self.n_vs_t = np.array([sum(pn * n_list) for pn in self.pn_vs_t])


    # coefficients which define the differential equation of motion for rho
    # refer to Eq. (11.1.14) Quantum Optics by Scully and Zubairy for details
    
    # def _M(self, n, m):
    #     return 0.5 * (n + m + 2) + (n - m)**2 * self.BdA / 8

    # def _N(self, n, m):
    #     return 0.5 * (n + m + 2) + (n - m)**2 * self.BdA / 16

    # def _fnm(self, n, m):
    #     f1 = self._M(n, m) * self.A / (1 + self._N(n, m) * self.BdA)
    #     f2 = 0.5 * self.kappa * (n + m)
    #     return - f1 - f2

    # def _gnm(self, n, m):
    #     return np.sqrt(n * m) * self.A / (1 + self._N(n - 1, m - 1) * self.BdA)

    # def _hnm(self, n, m):
    #     return self.kappa * np.sqrt((n + 1) * (m + 1)) 

    # differential equation for the whole density matrix
    # def _rho_nm_dot(self, rho_nm, t, f, g, h):
    #     """ ode update rule for rho_nm
    #     """
    #     # method 1 (memory economy) (final choice)
    #     rho_nm.shape = (self.N_max, self.N_max)
    #     rho_new = np.zeros([self.N_max, self.N_max])
    #     ij = range(self.N_max)
    #     for i in ij:
    #         for j in ij:
    #             rho_new[i, j] += f[i, j] * rho_nm[i, j]
    #             if i > 0 and j > 0:
    #                 rho_new[i, j] += g[i, j] * rho_nm[i - 1, j - 1]
    #             if i < self.N_max - 1 and j < self.N_max - 1:
    #                 rho_new[i, j] += h[i, j] * rho_nm[i + 1, j + 1]
    #     del rho_nm
    #     return rho_new.reshape(-1)

        # method 2 (original method):
        # rho = rho_nm.reshape(self.N_max, self.N_max)
        # rho_new = np.zeros([self.N_max, self.N_max])
        # ij = range(self.N_max)
        # for i in ij:
        #     for j in ij:
        #         rho_new[i, j] += f[i, j] * rho[i, j]
        #         if i > 0 and j > 0:
        #             rho_new[i, j] += g[i, j] * rho[i - 1, j - 1]
        #         if i < self.N_max - 1 and j < self.N_max - 1:
        #             rho_new[i, j] += h[i, j] * rho[i + 1, j + 1]
        # return rho_new.reshape(-1)
        
        # def helper(ij):
        #     i, j = ij
        #     result = f[i, j] * rho[i, j]
        #     if i > 0 and j > 0:
        #         result += g[i, j] * rho[i - 1, j - 1]
        #     if i < self.N_max - 1 and j < self.N_max - 1:
        #         result += h[i, j] * rho[i + 1, j + 1]
        #     return result
        
        # method 3:
        # ijpairs = [(i, j) for j in range(self.N_max) for i in range(self.N_max)]
        # # rho_new = np.array([helper(ij) for ij in ijpairs])
        # rho_new = list(map(helper, ijpairs))
        # return rho_new
        
        # method 4:
        # pool = ThreadPool(4)
        # # pool = Pool(4)
        # ijpairs = [(i, j) for j in range(self.N_max)  for i in range(self.N_max)]
        # rho_new = pool.map(helper, ijpairs)
        # pool.close()
        # pool.join()
        # return rho_new

#     def solve_steady_state_lst(self):
#         """ If the state is always diagonal during evolution,
#             get the diagonal terms of the steady state.
#             Solver: `scipy.sparse.linalg.lsqr()`.
#             Since none-zeor solutions are not existed, 
#             use `lsqr` to find the solution with the least squared error.
#         """
#         eq = np.zeros([self.N_max, self.N_max])
#         # y = np.repeat(np.finfo(float).eps, self.N_max)
#         y = np.repeat(0, self.N_max)

#         for k in range(self.N_max):
#             eq[k, k] = self._fnm(k, k)
#             if k < self.N_max - 1:
#                 eq[k, k + 1] = self._hnm(k, k)
#             if k > 0:
#                 eq[k, k - 1] = self._gnm(k, k)        
#         pn = lsqr(eq, y)[0]
        
#         pn = pn/sum(pn)
#         n = sum(pn * range(self.N_max))
#         entr = sum([- p * np.log2(p) for p in pn if p > 0])
        
#         return pn, n, entr
    
    
#     def solve_steady_state_three(self):
#         """ if the state is always diagonal during evolution
#             get the diagonal terms of the steady state
#         """
#         eq = np.zeros([self.N_max, self.N_max])
#         eq = np.vstack((eq, np.repeat(1, self.N_max)))
#         y = np.repeat(np.finfo(float).eps, self.N_max)
#         y = np.append(y, 1)

#         for k in range(self.N_max):
#             eq[k, k] = self._fnm(k, k)
#             if k < self.N_max - 1:
#                 eq[k, k + 1] = self._hnm(k, k)
#             if k > 0:
#                 eq[k, k - 1] = self._gnm(k, k)
        
#         eq = np.matrix(eq)
#         y = np.matrix(y).T
#         pn = (eq.T * eq).getI() * eq.T * y
#         pn = np.asarray(pn).reshape(-1)
        
#         # pn = pn/sum(pn)
#         n = sum(pn * range(self.N_max))
#         entr = sum([- p * np.log2(p) for p in pn if p > 0])
        
#         return pn, n, entr

# def boltzmann(ratio, N_max):
#     """ return an array of pn according to the boltzmann distribution
#     """
#     return np.array([(1 - ratio) * ratio ** n for n in np.arange(N_max)])