import numpy as np
import sympy
import string
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
import pandas as pd
import csv
from os.path import join as pjoin
from tqdm import tqdm
import scipy.special

from utils import OperonArgs

def split_by_punctuation(s):
    """
    Convert a string into a list, where the string is split by punctuation,
    excluding underscores or full stops.
    
    For example, the string 'he_ll*o.w0%rl^d' becomes
    ['he_ll', '*', 'o.w0', '%', 'rl', '^', 'd']
    
    Args:
        :s (str): The string to split up
        
    Returns
        :split_str (list[str]): The string split by punctuation
    
    """
    pun = string.punctuation.replace('_', '') # allow underscores in variable names
    pun = string.punctuation.replace('.', '') # allow full stops
    pun = pun + ' '
    where_pun = [i for i in range(len(s)) if s[i] in pun]
    if len(where_pun) > 0:
        split_str = [s[:where_pun[0]]]
        for i in range(len(where_pun)-1):
            split_str += [s[where_pun[i]]]
            split_str += [s[where_pun[i]+1:where_pun[i+1]]]
        split_str += [s[where_pun[-1]]]
        if where_pun[-1] != len(s) - 1:
            split_str += [s[where_pun[-1]+1:]]
    else:
        split_str = [s]
    return split_str

def is_float(s):
    """
    Function to determine whether a string has a numeric value
    
    Args:
        :s (str): The string of interest
        
    Returns:
        :bool: True if s has a numeric value, False otherwise
        
    """
    try:
        float(eval(s))
        return True
    except:
        return False

def replace_floats(s):
    """
    Replace the floats in a string by parameters named b0, b1, ...
    where each float (even if they have the same value) is assigned a
    different b.
    
    Args:
        :s (str): The string to consider
        
    Returns:
        :replaced (str): The same string, but with floats replaced by parameter names
        :values (list[float]): The values of the parameters in order [b0, b1, ...]
        
    """
    split_str = split_by_punctuation(s)
    values = []
    for i in range(len(split_str)):
        if is_float(split_str[i]) and "." in split_str[i]:
            values.append(float(split_str[i]))
            split_str[i] = f'b{len(values)-1}'
        elif len(split_str[i]) > 1 and split_str[i][-1] == 'e' and is_float(split_str[i][:-1]):
            if split_str[i+1] in ['+', '-']:
                values.append(float(''.join(split_str[i:i+3])))
                split_str[i] = f'b{len(values)-1}'
                split_str[i+1] = ''
                split_str[i+2] = ''
            else:
                assert split_str[i+1].is_digit()
                values.append(float(''.join(split_str[i:i+2])))
                split_str[i] = f'b{len(values)-1}'
                split_str[i+1] = ''
    replaced = ''.join(split_str)
    return replaced, values


def convert_operon_fun(eq, names, do_replace_floats=True):
    """
    Given the function outputted by operon, express this so that
    the variables are now appropriately names and the floats are
    replaced by parameters.
    
    Args:
        :eq (str): The equation outputted by operon
        :names (list[str]): The names of the parameters in order passed to operon
    
    Returns:
        :new_eq (str): The equation with the replaced symbols and floats
        :values (list[float]): The values of the parameters in order [b0, b1, ...]
    
    """
    
    new_eq = split_by_punctuation(eq)
    for i, n in enumerate(names):
        new_eq = [n if b == f'X{i+1}' else b for b in new_eq]
    new_eq = ''.join(new_eq)
    new_eq = sympy.sympify(new_eq)
    if do_replace_floats:
        new_eq, values = replace_floats(str(new_eq))
    else:
        values = []
    
    return new_eq, values


def plot_pareto(ini_file, ilen=None, loss_max=None, print_par_table=False, yvar='RMSE', yscale='log'):
    """
    Make the Pareto front plot
    
    Args:
        :ini_file (str): The path to the ini file containing the run information
        :ilen (int, default=None): The length of the equation to highlight. If None,
            then this is taken to be the final equation
        :loss_max (float, default=None): Maximum value y axis can take
        :print_par_table (bool, default=False): Whether to print each parameter out individually
        :yvar (str, default='RMSE'): The variable to plot on the y axis
        :yscale (str, default='log'): The scale to use for the y axis. Can be 'log' or 'linear'.
            
    Returns:
        :fig (matplotlib.figure.Figure): Figure containing Pareto front
        :ax (matplotlib.pyplot.axis): Axis of fig containing the Pareto front
    """
    
    args = OperonArgs(ini_file)

    run_name = f'{args.target_name}_{str(args.version_num)}'
    out_dir = pjoin(args.fit_dir, run_name)
    fname = f'{out_dir}/{run_name}_fun.csv'
    df = pd.read_csv(fname, delimiter=';')

    if ilen is None:
        eq_idx = -1
    else:
        eq_idx = list(df['Length']).index(ilen)
    best_eq = list(df['Equation'])[eq_idx]
    print('\nAll model lengths:')
    print(np.sort(list(df['Length'])))
    print('\nEquation requested:')
    print(best_eq)
    with open(f'{out_dir}/{run_name}_names.txt', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        names = reader.__next__()

    if yvar + '_train' not in df.columns or yvar + '_test' not in df.columns:
        if yvar == 'RMSE' and 'MSE_train' in df.columns and 'MSE_val' in df.columns:
            df['RMSE_train'] = np.sqrt(df['MSE_train'])
            df['RMSE_val'] = np.sqrt(df['MSE_val'])
        else:
            length = list(df['Length'])[eq_idx]
            yvar_vals = {'train': [], 'val': []}
            for j in tqdm(range(len(df))):
                length = list(df['Length'])[j]
                for name in ['train', 'val']:
                    fname = pjoin(out_dir, f'{args.target_name}_{name}_{length}.csv')
                    ytrue, ypred = np.loadtxt(fname, usecols=(-2, -1), unpack=True)
                    if yvar == 'RMSE':
                        yvar_vals[name].append(np.sqrt(np.mean((ytrue - ypred) ** 2)))
                    elif yvar == 'MAE':
                        yvar_vals[name].append(np.mean(np.abs(ytrue - ypred)))
                    elif yvar == 'MedAE':
                        yvar_vals[name].append(np.median(np.abs(ytrue - ypred)))
                    elif yvar == 'R2':
                        yvar_vals[name].append(np.sum((ytrue - ypred) ** 2) / np.sum((ytrue - np.mean(ytrue)) ** 2))
                    elif yvar == 'MSE':
                        yvar_vals[name].append(np.mean((ytrue - ypred) ** 2))
                    else:
                        raise ValueError(f'Unknown yvar: {yvar}')
            df[yvar + '_train'] = yvar_vals['train']
            df[yvar + '_val'] = yvar_vals['val']
        
    # Make a nicer visual version
    param_dict = {'sigma8':'sigma_8', 'Omm':'Omega_m', 'Omb':'Omega_b', 'ns':'n_s', 'h':'h',
                  'lnksigma':'log(k_sigma)', 'neff':'n_e', 'C':'C', 'lnsigma':'log(sigma)', 
                  'hyper':'F', 'comoving':'chi'}
    param_dict = {
        'logMstar[logMsun]': 'log(M_star)', 
        'logMgas[logMsun]': 
        'log(M_gas)', 
        'SFR[Msun/yr]': 
        'SFR',
        'SFR_compact[Msun/yr]': 'SFR_compact', 
        'Zgas[mass_fraction]': 'Z_gas', 
        'rstar[kpc]': 'r_star',
       'rgas[kpc]': 'r_gas', 
       'Mdust_RKC[Msun]': 'M_dust', 
       'rgas_SF[kpc]': 'r_gasSF', 
       'Age[yr]': 'Age',
       'SFR_100Myr[Msun]': 'SFR_100Myr', 
       'SFR_10Myr[Msun]': 'SFR_10Myr', 
       'rstar_old[kpc]': 'r_starold',
       'rstar_young[kpc]': 'r_staryoung', 
       'Mmetals(<rstar_young)[Msun]': 'M_metalsstaryoung',
       'Mmetals(<rstar_old)[Msun]': 'M_metalsstarold', 
       'Mmetals(<rgas_SF)[Msun]': 'M_metalsgasSF', 
       'Mstar[Msun]': 'M_star',
       'Mgas[Msun]': 'M_gas', 
       'Sigma_SFR_10_young[Msun/yr/kpc^2]': 'Sigma_SFR10young',
       'Sigma_SFR_100_old[Msun/yr/kpc^2]': 'Sigma_SFR100old', 
       'Sigma_SFR_all[Msun/yr/kpc^2]': 'Sigma_SFRall',
       'Sigma_gas_rgas[Msun/kpc^2]': 'Sigma_gasrgas', 
       'Sigma_gas_rSF[Msun/kpc^2]': 'Sigma_gasrSF',
       'burstiness_ks_young': 'B_ksyoung', 
       'burstiness_ks_old': 'B_ksold', 
       'burstiness_ks_all': 'B_ksall',
       'rstar_old_over_rgas': 'r_StarOldOverRgas', 
       'rstar_old_over_rSF': 'r_StarOldOverRgasSF', 
       'rstar_young_over_rgas': 'r_StarYoungOverRgas',
       'rstar_young_over_rSF': 'r_StarYoungOverRgasSF', 
       'rstar_old_over_rstar_young': 'r_StarOldOverRStaryoung',
       'Sigma_dust_rgas[Msun/kpc^2]': 'Sigma_DustRgas',
       'Av': 'A_V',
       'B_0': 'B_0',
       'B_1': 'B_1',
       'B_2': 'B_2',
       'B_3': 'B_3'}


    for i, n in enumerate(names):
        if n in param_dict.keys():
            names[i] = param_dict[n]
        else:
            print(f'Warning: {n} not in param_dict, using original name')

    eq, pars = convert_operon_fun(best_eq, names)
    eq = sympy.sympify(eq)
    display(eq)
    print('\nLatex version:')
    sympy.print_latex(eq)

    print('\nNumber of parameters:', len(pars))
    print('Values:', pars)
    if print_par_table:
        for i in range(len(pars)):
            print(f'b{i} = {pars[i]}')
    
    rcParams['font.size'] = 16
    rcParams["text.usetex"] = False
    
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('Set1')
    ax.axvline(df['Length'].to_numpy()[eq_idx], ls=':', color='k', label='Chosen')
    m = np.isfinite(df[f'{yvar}_train'])
    x = np.array(df['Length'][m])
    y = np.array(df[f'{yvar}_train'][m])
    i = np.argsort(x)
    ax.plot(x[i], y[i], marker='.', color=cmap(0), label='Training')
    m = np.isfinite(df[f'{yvar}_val'])
    x = np.array(df['Length'][m])
    y = np.array(df[f'{yvar}_val'][m])
    i = np.argsort(x)
    ax.plot(x[i], y[i], marker='.', ls='--', color=cmap(1), label='Validation')
    if ax.get_xlim()[1] > args.max_length:
        ax.set_xlim(None, args.max_length)
    ax.set_yscale(yscale)
    if loss_max is not None:
        ylim = list(ax.get_ylim())
        ylim[1] = min(loss_max, ylim[1])
        ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Model Length')
    if yvar == 'R2':
        ylab = '1 - R2'
    else:
        ylab = yvar
    ax.set_ylabel(ylab)
    if yvar == 'R2' and yscale == 'linear':
        # Ensure ylim doesn't extend outside [0, 1]
        ylim = list(ax.get_ylim())
        ylim[0] = max(0, ylim[0])
        ylim[1] = min(1, ylim[1])
        ax.set_ylim(*ylim)
    ax.legend(loc='upper right')
    
    fig.align_labels()
    fig.tight_layout()

    # Print result for train and validation
    print(f'\n{ylab} for chosen model:')
    print(f'Training: {df[f"{yvar}_train"].values[eq_idx]:.4f}')
    print(f'Validation: {df[f"{yvar}_val"].values[eq_idx]:.4f}')

    return fig, ax


# def prediction_plots(ini_file, ilen=None, plot_abs_error=False, frac_error=True):
#     """
#     Show the difference between the truth and predicted
    
#     Args:
#         :ini_file (str): The path to the ini file containing the run information
#         :ilen (int, default=None): The length of the equation to highlight. If None,
#             then this is taken to be the final equation
#         :plot_abs_error (bool, default=False): Whether to plot the distirbution of the absolute
#             error in logspace
#         :frac_error (bool, default=True): Whether to use the fractional error (True) in the plot
#             or absolute error (False).
            
#     Returns:
#         :fig (matplotlib.figure.Figure): Figure containing plot
#         :axs (np.ndarray[matplotlib.pyplot.axis]): Axes of fig containing the plot
#     """
    
#     args = OperonArgs(ini_file)
    
#     run_name = f'{args.param}_{str(args.version_num)}'
#     out_dir = pjoin(args.fit_dir, run_name)
#     fname = f'{out_dir}/{run_name}_fun.csv'
#     df = pd.read_csv(fname, delimiter=';')
    
#     if ilen is None:
#         eq_idx = -1
#     else:
#         eq_idx = list(df['Length']).index(ilen)
#     length = list(df['Length'])[eq_idx]
        
#     outname_pred_train = f'{out_dir}/{run_name}_train_{length}.csv'
#     outname_pred_test = f'{out_dir}/{run_name}_test_{length}.csv'
    
#     cmap = plt.get_cmap('Set1')
#     ms = 8
#     rcParams['font.size'] = 16
#     rcParams["text.usetex"] = False
    
#     if args.param in ['ksigma', 'neff', 'C', 'sigma', 'hyper', 'As', 'comoving', 'Dz']:
        
#         fig, axs = plt.subplots(1, 3, figsize=(20,4))
#         bins = None
        
#         for i, name in enumerate(['train', 'test']):

#             fname= f'{out_dir}/{run_name}_{name}_{length}.csv'
#             data = np.loadtxt(fname)
#             ytrue = data[:,-2]
#             ypred = data[:,-1]

#             if args.param == 'ksigma':
#                 ytrue = np.exp(ytrue)
#                 ypred = np.exp(ypred)
#             elif args.param == 'As':
#                 if name == 'train':
#                     fname = pjoin(args.data_dir, f'{args.param}_data', f'{args.param}_{args.boltzmann}_data_n{args.n}_s{args.train_seed}.txt')
#                 elif name == 'test':
#                     fname = pjoin(args.data_dir, f'{args.param}_data', f'{args.param}_{args.boltzmann}_data_n{args.n}_s{args.val_seed}.txt')
#                 data = np.loadtxt(fname, skiprows=1)
#                 with open(fname, 'r') as f:
#                     names = f.readline().strip().split()
#                 idx = names.index('sigma8')
#                 sigma8 = data[:,idx]
#                 ytrue = (np.exp(ytrue) * sigma8) ** 2
#                 ypred = (np.exp(ypred) * sigma8) ** 2
#             elif args.param == 'sigma':
#                 if name == 'train':
#                     fname = pjoin(args.data_dir, f'{args.param}_data', f'{args.param}_{args.boltzmann}_data_n{args.n}_s{args.train_seed}.txt')
#                 elif name == 'test':
#                     fname = pjoin(args.data_dir, f'{args.param}_data', f'{args.param}_{args.boltzmann}_data_n{args.n}_s{args.val_seed}.txt')
#                 data = np.loadtxt(fname, skiprows=1)
#                 with open(fname, 'r') as f:
#                     names = f.readline().strip().split()
#                 idx = names.index('As')
#                 As = data[:,idx]
#                 ytrue = np.exp(ytrue) * np.sqrt(1e9 * As)
#                 ypred = np.exp(ypred) * np.sqrt(1e9 * As)

            
#             label = ['Training', 'Validation']
#             label = label[i]
            
#             axs[0].plot(ytrue, ypred, '.', ms=ms, label=label, color=cmap(i))
#             if frac_error:
#                 error = ypred / ytrue - 1
#             else:
#                 error = ypred - ytrue
#             if plot_abs_error:
#                 axs[1].semilogy(ytrue, np.abs(error), '.', ms=ms, label=label, color=cmap(i))
#             else:
#                 axs[1].plot(ytrue, error, '.', ms=ms, label=label, color=cmap(i))
                
#             if bins is None:
#                 xmin = np.percentile(error, 1)
#                 xmax = np.percentile(error, 99)
#                 bins = np.linspace(xmin, xmax, 30)
#             axs[2].hist(error, bins=bins, density=True, histtype='step', label=label, color=cmap(i))

#             if args.param in ['ksigma', 'sigma', 'As']:
#                 axs[0].set_xscale('log')
#                 axs[0].set_yscale('log')
#                 axs[1].set_xscale('log')
            
#             all_frac_res = ypred / ytrue
#             rmse = np.sqrt(np.mean((all_frac_res - 1) ** 2))
#             print(f"\n{name}")
#             print("\tRMSE:", rmse)
#             rmae = np.mean(np.abs(all_frac_res - 1))
#             print("\tMAE:", rmae)
#             error = np.abs(ypred / ytrue - 1)
#             m = error > 0.01
#             print(f'\tNumber of points greater than 1% off: {m.sum()} of {len(m)} = {round(m.sum()/len(m)*100, 2)}%')
#             m = error > 0.02
#             print(f'\tNumber of points greater than 2% off: {m.sum()} of {len(m)} = {round(m.sum()/len(m)*100, 2)}%')
        
#         axs[0].set_xlabel('True')
#         axs[1].set_xlabel('True')
#         axs[0].set_ylabel('Predicted')
#         axs[1].set_ylabel('Predicted')
#         if frac_error:
#             axs[1].set_ylabel('Fractional error')
#             axs[2].set_xlabel('Fractional error')
#         else:
#             axs[1].set_ylabel('Error')
#             axs[2].set_xlabel('Error')
#         if plot_abs_error and frac_error:
#             axs[1].axhline(0.01, color='k', ls='--')
#         else:
#             axs[1].axhline(0, color='k')
#         axs[2].axvline(0, color='k')
#         if frac_error:
#             axs[2].axvline(0.01, color='k', ls='--')
#             axs[2].axvline(-0.01, color='k', ls='--')
#         axs[0].legend()
        
#         xlim = axs[0].get_xlim()
#         ylim = axs[0].get_ylim()
#         xlim = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
#         axs[0].plot(xlim, xlim, color='k')
#         axs[0].set_ylim(xlim)
#         axs[0].set_xlim(xlim)
#         axs[1].set_xlim(xlim)
        
#     elif args.param in ['pk', 'logn']:
        
#         fig, axs = plt.subplots(1, 2, figsize=(15,4), sharex=True, sharey=True)

#         for i, name in enumerate(['train', 'test']):
#         # for i, name in enumerate(['train']):

#             fname= f'{out_dir}/{run_name}_{name}_{length}.csv'
#             data = np.loadtxt(fname)
#             ytrue = data[:,-2]
#             ypred = data[:,-1]

#             if args.remove_fnw:
#                 logfnw = data[:,-3]
#                 ytrue += logfnw
#                 ypred += logfnw

#             rmse = np.sqrt(np.mean((ytrue - ypred) ** 2))
#             print(f'\nRMSE {name}: %.3e'%rmse)
            
#             ypred = np.exp(ypred)
#             ytrue = np.exp(ytrue)

#             fname = pjoin(args.data_dir, f'pk_data', f'pk_{args.boltzmann}_data_n{args.n}_k{args.nk}_s{args.train_seed}')
#             if args.nk_bao == 0:
#                 fname += '_unifk'
#             fname += '.txt'
#             with open(fname, 'r') as f:
#                 header = f.readline().split()
#             k = np.unique(np.loadtxt(fname, skiprows=1)[:,header.index('k')])

#             # if name == 'train':
#             #     fname = pjoin(args.data_dir, f'{args.param}_data', f'{args.param}_{args.boltzmann}_data_n{args.n}_s{args.train_seed}.txt')
#             # elif name == 'test':
#             #     fname = pjoin(args.data_dir, f'{args.param}_data', f'{args.param}_{args.boltzmann}_data_n{args.n}_s{args.val_seed}.txt')
#             # with open(fname, 'r') as f:
#             #     names = f.readline().strip().split()
#             k_all = data[:,header.index('k')]
#             print('k_all shape:', k_all.shape, len(set(k_all)))
#             mask = (k_all >= args.kmin) & (k_all <= args.kmax)
#             print(k_all.shape, ypred.shape)
#             ytrue = ytrue[mask]
#             ypred = ypred[mask]
#             k = np.unique(k_all[mask])

#             if args.cut_bao:
#                 bao_mask = (k < args.kbao_start) | (k > args.kbao_end)
#                 print(f"Cutting BAO: keeping {bao_mask.sum()} out of {len(bao_mask)} k values")
#                 k = k[bao_mask]


#             n = int(len(ytrue) / len(k))
#             nk = len(k)
#             print(f"Number of different k values: {len(k)}")
#             print(f"Number of different other parameter combinations: {n}")
#             all_frac_res = [None] * n
#             for j in range(n):
#                 all_frac_res[j] = ypred[j*nk:(j+1)*nk] / ytrue[j*nk:(j+1)*nk]
#             all_frac_res = np.array(all_frac_res)
#             all_perc = [34+13.5+2.35, 34+13.5, 34]
#             all_perc = all_perc[1:]
#             for j, delta in enumerate(all_perc):
#                 low = np.percentile(all_frac_res, 50 - delta, axis=0) - 1
#                 high = np.percentile(all_frac_res, 50 + delta, axis=0) - 1
#                 print(f'\t\t{len(all_perc)-j} sigma:', np.amin(low), np.amax(high))
#                 if args.cut_bao:
#                     mlow = k < args.kbao_start
#                     mhigh = k > args.kbao_end
#                     axs[i].fill_between(k[mlow], low[mlow], high[mlow], color=cmap(j), 
#                                         label=str(len(all_perc)-j) + r'$\sigma$')
#                     axs[i].fill_between(k[mhigh], low[mhigh], high[mhigh], color=cmap(j))
#                 else:
#                     axs[i].fill_between(k, low, high, color=cmap(j), label=str(len(all_perc)-j) + r'$\sigma$')
#             if args.cut_bao:
#                 mlow = k < args.kbao_start
#                 mhigh = k > args.kbao_end
#                 axs[i].plot(k[mlow], np.median(all_frac_res, axis=0)[mlow] - 1, color='k')
#                 axs[i].plot(k[mhigh], np.median(all_frac_res, axis=0)[mhigh] - 1, color='k')
#             else:
#                 axs[i].plot(k, np.median(all_frac_res, axis=0) - 1, color='k')
#             rmse = np.sqrt(np.mean((all_frac_res - 1) ** 2))
#             print("\t\tRMSE:", rmse)
#             rmae = np.mean(np.abs(all_frac_res - 1))
#             print("\t\tMAE:", rmae)

#             axs[i].set_xscale('log')
#             axs[i].set_xlabel(r'$k \ / \ h {\rm Mpc^{-1}}$')
#             axs[i].legend()
#             axs[i].axhline(0, color='k', ls='--', lw=2)
#             axs[i].axhline(0.01, color='k', ls='--', lw=2)
#             axs[i].axhline(-0.01, color='k', ls='--', lw=2)

#         axs[0].set_title('Training')
#         axs[1].set_title('Validation')
#         if args.param == 'pk':
#             axs[0].set_ylabel(r'Fractional Error on $F(k)$')
#         elif args.param == 'logn':
#             axs[0].set_ylabel(r'Fractional Error on $N(k,z)$')
#         else:
#             axs[0].set_ylabel(r'Fractional Error on $%s$'%(args.param))
    
#     fig.align_labels()
#     fig.tight_layout()
    
#     return fig, axs


def print_to_latex(ini_file):
    """
    Convert operon output to latex and print to file
    
    Args:
        :ini_file (str): The path to the ini file containing the run information
    """
    
    args = OperonArgs(ini_file)
    
    run_name = f'{args.param}_{str(args.version_num)}'
    out_dir = pjoin(args.fit_dir, run_name)
    fname = f'{out_dir}/{run_name}_fun.csv'
    df = pd.read_csv(fname, delimiter=';')
    
    with open(f'{out_dir}/{run_name}_names.txt', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        names = reader.__next__()

    fname = f'{out_dir}/{run_name}_latex.txt'
    with open(fname, 'w') as f:
        for i in tqdm(range(len(df))):
            best_eq = list(df['Equation'])[i]
            length = list(df['Length'])[i]
            for i, n in enumerate(names):
                best_eq = best_eq.replace(f'X{i+1}',n)
            best_eq = sympy.sympify(best_eq)
            replaced, values = replace_floats(str(best_eq))
            expr = sympy.sympify(replaced)
            print(length, ' & $', sympy.latex(expr), '$ \\\\', file=f)
    
    return


# Here we define some useful sympy variables

basis_functions = [["x", "b"],  # type0
                ["square", "exp", "inv", "sqrt", "log", "cos"],  # type1
                ["+", "*", "-", "/", "pow"]]  # type2

x, y = sympy.symbols('x y', positive=True)
a, b = sympy.symbols('a b', real=True)
sympy.init_printing(use_unicode=True)
inv = sympy.Lambda(a, 1/a)
square = sympy.Lambda(a, a*a)
cube = sympy.Lambda(a, a*a*a)
sqrt = sympy.Lambda(a, sympy.sqrt(a))
log = sympy.Lambda(a, sympy.log(a))
power = sympy.Lambda((a,b), sympy.Pow(a, b))

sympy_locs = {"inv": inv,
            "square": square,
            "cube": cube,
            "pow": power,
            "Abs": sympy.Abs,
            "x":x,
            "sqrt":sqrt,
            "log":log,
            }