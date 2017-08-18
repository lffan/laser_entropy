#!/usr/bin/env python

""""""

from datatime import datatime
import numpy as np
import pandas as pd


__author__ = 'Longfei Fan'
__version__ = '1.0'
__status__ = 'Development'
__date__ = '08/18/2017'


def create_evolve_laser(nbar, alpha, g, kappa, N_max, init_psi, t_list):
    """
    create a laser given on the given paramenters, and then evolve
    return the evolved laser
    """
    ra = 2 * kappa * nbar * alpha / (alpha - 1)
    gamma = np.sqrt(nbar / (alpha - 1)) * 2 * g
    
    laser = master_eq.Laesr(g, ra, gamma, kappa, N_max)
    laser.set_init_state(init_psi)
    laser.pn_evolve(t_list)

    return '{:.2f}'.format(alpha), laser