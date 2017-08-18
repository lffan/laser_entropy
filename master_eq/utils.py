from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import erfc

# from multiprocessing import Pool
# from multiprocessing.dummy import Pool as ThreadPool

from qutip import *
import master_eq


def save_cnb_to_csv(cnb1, cnb2, N, t_list, kappa):
    """
    save nbar, entropy, and final pn of cnb1 and cnb2 into csv
    """
    n_dict = {'$\kappa t$': t_list * kappa}
    entr_dict = {'$\kappa t$': t_list * kappa}
    pn_dict = {'n': np.arange(N + 1)}

    n_dict['CNB I'] = cnb1.get_nbars()
    n_dict['CNB II'] = cnb2.get_nbars()
    entr_dict['CNB I'] = cnb1.get_entrs()
    entr_dict['CNB II'] = cnb2.get_entrs()
    pn_dict['CNB I'] = cnb1.get_pns()[-1]
    pn_dict['CNB II'] = cnb2.get_pns()[-1]

    entr_dict['I Approx'] = get_entr_approx(cnb1)
    entr_dict['II Approx'] = get_entr_approx(cnb2)
    entr_dict['I Diff'] = entr_dict['CNB I'] - entr_dict['I Approx']
    entr_dict['II Diff'] = entr_dict['CNB II'] - entr_dict['II Approx']

    n_df = pd.DataFrame(n_dict, columns=n_dict.keys())
    entr_df = pd.DataFrame(entr_dict, columns=entr_dict.keys())
    pn_df = pd.DataFrame(pn_dict, columns=pn_dict.keys())

    n_df.to_csv('./data/cnb_n_df.csv', index=False)
    entr_df.to_csv('./data/cnb_entr_df.csv', index=False)
    pn_df.to_csv('./data/cnb_pn_df.csv', index=False)


def read_cnb_from_csv(path, state_name):
    """
    read nbar, entropy, and final pn of cnb1 and cnb2 from csv
    """
    n_df = pd.read_csv(path + state_name + '_n_df.csv')
    entr_df = pd.read_csv(path + state_name + '_entr_df.csv')
    pn_df = pd.read_csv(path + state_name + '_pn_df.csv')
    return n_df, entr_df, pn_df


def df_plot(df, xaxis, columns, xlim, xlabel, ylabel, title, style, loc=0,\
            entr_cohe=False, entr_thml=False):
    """
    plot dataframe with multiple columns
    """
    df.loc[:, columns].plot(x=xaxis, xlim=xlim, style=style, 
        figsize=(6, 4), fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=14, loc=loc)

    if entr_thml:
        plt.axhline(y=entr_thml, color='black', linewidth=0.5, \
                    linestyle='--', label='thermal')
    if entr_cohe:
        plt.axhline(y=entr_cohe, color='red', linewidth=0.5, \
                    linestyle='-', label='coherent')


def plot_pn_vs_time(state, indics, title, x1, x2, y1, y2, factor):
    """ plot photon statistics with respect to time
        state: MasterEq object
        indics: time point to be plotted
        title: figure title
        x1, x2: xlim
        y1, y2: ylim
    """
    N_max = state.N_max
    t_list = state.t_list
    pns_vs_t = state.get_pns()

    lstyle = ['-', '--', ':', '-.', '-', '--']

    fig, ax = plt.subplots(sharex=True, figsize=(10, 4))
    for i, index in enumerate(indics):
        ax.plot(np.arange(N_max), pns_vs_t[index], linestyle=lstyle[i], 
            linewidth=2, label='{:.4f}'.format(t_list[index] *factor))
        
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    ax.set_xlabel(r'$n$', fontsize=14)
    ax.set_ylabel(r'$p_n$', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    plt.title(title, fontsize=14);


def calc_entr_vec(mean, varn):
    """ Calculate the vector of entropy 
        given on the vector of the mean and the variance
    """
    result = np.log(np.sqrt(2.0 * np.pi * varn)) + 0.5
    result +=  np.log(0.5 * erfc(- mean / np.sqrt(2.0 * varn)))
    nn = - mean * np.exp(- mean**2 / 2.0 / varn) 
    dd = np.sqrt(2.0 * np.pi * varn) * erfc(- mean / np.sqrt(2.0 * varn))
    result += nn / dd
    result = np.insert(result, 0, 0)
    return result


def get_entr_approx(state):
    pns_all = state.get_pns()[1:]
    mean_all = np.array(state.get_nbars())[1:]
    N_max = state.N_max
    
    var_n_all = []
    for pns in pns_all:
        aver_n = sum([pns[i] * i for i in range(N_max)])
        aver_n2 = sum([pns[i] * i**2 for i in range(N_max)])
        var_n_all.append(aver_n2 - aver_n**2)
       
    result = calc_entr_vec(mean_all, np.array(var_n_all))
    # result[result < 0] = 0
    return result


def plot_varn_entr(df, col1, col2, x1, x2):
    fig, ax = plt.subplots()
    t_list = df['$\kappa t$'][x1:x2]
    entr = df[col1][x1:x2]
    entr_approx = df[col2][x1:x2]
    
    ax.plot(t_list, entr_approx, linestyle='--', label='approx')
    ax.plot(t_list, entr, linestyle='-', label ='exact')
    ax.plot(t_list, entr - entr_approx, linestyle='-.', label ='difference')
    ax.set_xlabel('$time$', fontsize=14)
    ax.set_ylabel('$S$', fontsize=14)
    ax.legend(fontsize=14, loc=0)
    ax.set_title("Entropy for " + col1)



# def entropy_vs_ratio(ratios, t_list, g, kappa, nbar, N_max, init_psi, solver='pn'):
#     """ simulate lasers with different A/C ratios
#     """
#     def get_para(alpha, nbar, kappa, g):
#         """ calculate parameters given on ratio, nbar, kappa, and g
#         """
#         gamma = np.sqrt(nbar / (alpha - 1)) * 2 * g
#         ra = 2 * kappa * nbar * alpha / (alpha - 1)
#         return {'g': g, 'gamma': gamma, 'C': kappa, 'ra': ra,
#                 'A': 2 * ra * g**2 / gamma**2, 'B': 8 * ra * g**4 / gamma**4}
    
#     # step = round(len(t_list) / 100)
#     n_dict = {'gt': t_list * g}
#     entr_dict = {'gt': t_list * g}
#     l_dict = {}

#     for alpha in ratios:
#         paras = get_para(alpha, nbar, kappa, g)
#         g, ra, gamma, kappa = paras['g'], paras['ra'], paras['gamma'], paras['C']

#         print(str(datetime.now()))
#         print('ratio: {:>5.2f}, ra: {:3.4f}, A: {:.3e}, C: {:.3e}, B: {:.3e}\n'. \
#               format(alpha, ra, paras['A'], kappa, paras['B']))
#         l = laser.LaserOneMode(g, ra, gamma, kappa)
#         if solver == 'pn':
#             l.pn_evolve(init_psi, N_max, t_list)
#         elif solver == 'rho':
#             l.rho_evolve(init_psi, N_max, t_list)
        
#         key = '{:.2f}'.format(alpha)
#         l_dict[key] = l
#         n_dict[key] = l.get_ns()
#         entr_dict[key] = l.get_entrs()

#     print(str(datetime.now()))

#     return l_dict, n_dict, entr_dict


# def get_para(alpha, nbar, kappa, g):
#     """ calculate parameters given on ratio, nbar, kappa, and g
#     """
#     gamma = np.sqrt(nbar / (alpha - 1)) * 2 * g
#     ra = 2 * kappa * nbar * alpha / (alpha - 1)
#     return {'g': g, 'gamma': gamma, 'C': kappa, 'ra': ra,
#             'A': 2 * ra * g**2 / gamma**2, 'B': 8 * ra * g**4 / gamma**4}


# def evolution(alpha, t_list, g, kappa, nbar, N_max, init_psi, solver):
#     """
#     """
#     paras = get_para(alpha, nbar, kappa, g)
#     g, ra, gamma, kappa = paras['g'], paras['ra'], paras['gamma'], paras['C']
#     print('ratio: {:>5.2f}, ra: {:3.4f}, A: {:.3e}, C: {:.3e}, B: {:.3e}'. \
#           format(alpha, ra, paras['A'], kappa, paras['B']))
#     l = laser.LaserOneMode(g, ra, gamma, kappa)
#     if solver == 'pn':
#         l.pn_evolve(init_psi, N_max, t_list)
#     elif solver == 'rho':
#         l.rho_evolve(init_psi, N_max, t_list)
#     key = '{:.2f}'.format(alpha)
    
#     return key, l


# def entropy_vs_ratio(ratios, t_list, g, kappa, nbar, N_max, init_psi, solver='pn'):
#     """ simulate lasers with different A/C ratios
#     """
#     n_dict = {'gt': t_list * g}
#     entr_dict = {'gt': t_list * g}
#     l_array = []

#     for alpha in ratios:
#         key, l = evolution(alpha, t_list, g, kappa, nbar, N_max, init_psi, solver)
        
#         l_array.append([key, l])
#         n_dict[key] = l.get_ns()
#         entr_dict[key] = l.get_entrs()

#     return l_array, n_dict, entr_dict


# def entropy_vs_ratio(ratios, t_list, g, kappa, nbar, N_max, init_psi, solver='pn'):
#     """ simulate lasers with different A/C ratios
#     """
#     l_array = [evolution(alpha, t_list, g, kappa, nbar, N_max, init_psi, solver) for alpha in ratios]
    
#     n_dict = {'gt': t_list * g}
#     entr_dict = {'gt': t_list * g}
#     for key, l in l_array:
#         n_dict[key] = l.get_ns()
#         entr_dict[key] = l.get_entrs()

#     return l_array, n_dict, entr_dict

# test for multi-processing
# def entropy_vs_ratio(ratios, t_list, g, kappa, nbar, N_max, init_psi, solver='pn'):
#     """ simulate lasers with different A/C ratios
#     """
#     def evolution_wrapper(alpha):
#         return evolution(alpha, t_list, g, kappa, nbar, N_max, init_psi, solver)
    
#     l_array = list(map(evolution_wrapper, ratios))
    
#     # pool = ThreadPool(4)
#     # l_array = list(pool.map(evolution_wrapper, ratios))
#     # pool.close()
#     # pool.join()
    
#     n_dict = {'gt': t_list * g}
#     entr_dict = {'gt': t_list * g}
#     for key, l in l_array:
#         n_dict[key] = l.get_ns()
#         entr_dict[key] = l.get_entrs()

#     return l_array, n_dict, entr_dict
   

