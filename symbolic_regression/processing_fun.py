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


def plot_pareto(ini_file, ilen=None, loss_max=None, print_par_table=False):
    """
    Make the Pareto front plot
    
    Args:
        :ini_file (str): The path to the ini file containing the run information
        :ilen (int, default=None): The length of the equation to highlight. If None,
            then this is taken to be the final equation
        :loss_max (float, default=None): Maximum value y axis can take
        :print_par_table (bool, default=False): Whether to print each parameter out individually
            
    Returns:
        :fig (matplotlib.figure.Figure): Figure containing Pareto front
        :ax (matplotlib.pyplot.axis): Axis of fig containing the Pareto front
    """
    
    args = OperonArgs(ini_file)
    
    run_name = f'{args.in_param}_{str(args.version_num)}'
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
    eq, _ = convert_operon_fun(best_eq, names, do_replace_floats=False)
    print('\nEquation with pars')
    print(eq)
    eq, pars = convert_operon_fun(best_eq, names)
    print('\nConverted equation:')
    print(eq)
    print('\nNumber of parameters:', len(pars))
    print('Values:', pars)
    if print_par_table:
        for i in range(len(pars)):
            print(f'b{i} = {pars[i]}')
        
    # Make a nicer visual version
    param_dict = {}
    for i in range(1,9):
        param_dict[f'IOB{i}'] = f'I_{i}'
        param_dict[f'PCA{i}'] = f'P_{i}'
    param_dict['lam'] = 'x'
    for i, n in enumerate(names):
        if n in param_dict.keys():
            names[i] = param_dict[n]
    eq, pars = convert_operon_fun(best_eq, names)
    eq = sympy.sympify(eq)
    display(eq)
    print('\nLatex version:')
    sympy.print_latex(eq)
    
    rcParams['font.size'] = 16
    rcParams["text.usetex"] = True
    
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('Set1')
    ax.axvline(df['Length'].to_numpy()[eq_idx], ls=':', color='k', label='Chosen')
    m = np.isfinite(np.sqrt(df['MSE_train']))
    x = np.array(df['Length'][m])
    y = np.array(np.sqrt(df['MSE_train'])[m])
    i = np.argsort(x)
    ax.plot(x[i], y[i], marker='.', color=cmap(0), label='Training')
    m = np.isfinite(np.sqrt(df['MSE_val']))
    x = np.array(df['Length'][m])
    y = np.array(np.sqrt(df['MSE_val'])[m])
    i = np.argsort(x)
    ax.plot(x[i], y[i], marker='.', ls='--', color=cmap(1), label='Validation')
    if ax.get_xlim()[1] > args.max_length:
        ax.set_xlim(None, args.max_length)
    ax.set_yscale('log')
    if loss_max is not None:
        ylim = list(ax.get_ylim())
        ylim[1] = min(loss_max, ylim[1])
        ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Model Length')
    ax.set_ylabel('Root Mean Squared Error')
    ax.legend(loc='upper right')
    
    fig.align_labels()
    fig.tight_layout()

    return fig, ax


def prediction_plots(ini_file, ilen=None):
    """
    Show the difference between the truth and predicted
    
    Args:
        :ini_file (str): The path to the ini file containing the run information
        :ilen (int, default=None): The length of the equation to highlight. If None,
            then this is taken to be the final equation
            
    Returns:
        :fig (matplotlib.figure.Figure): Figure containing plot
        :axs (np.ndarray[matplotlib.pyplot.axis]): Axes of fig containing the plot
    """
    
    args = OperonArgs(ini_file)
    
    run_name = f'{args.in_param}_{str(args.version_num)}'
    out_dir = pjoin(args.fit_dir, run_name)
    fname = f'{out_dir}/{run_name}_fun.csv'
    df = pd.read_csv(fname, delimiter=';')
    
    if ilen is None:
        eq_idx = -1
    else:
        eq_idx = list(df['Length']).index(ilen)
    length = list(df['Length'])[eq_idx]
    
    cmap = plt.get_cmap('Set1')
    rcParams['font.size'] = 16
    rcParams["text.usetex"] = True
        
    fig, axs = plt.subplots(1, 2, figsize=(15,4), sharex=True, sharey=True)

    for i, name in enumerate(['train', 'val']):

        fname= f'{out_dir}/{run_name}_{name}_{length}.csv'
        data = np.loadtxt(fname)
        ytrue = data[:,-2]
        ypred = data[:,-1]

        rmse = np.sqrt(np.mean((ytrue - ypred) ** 2))
        print(f'\nRMSE {name}: %.3e'%rmse)
        
        ypred = np.exp(ypred)
        ytrue = np.exp(ytrue)

        fname = pjoin(args.data_dir, f'{args.in_param}_data_{args.version_num}', f'{args.in_param}_train_data.txt')
        with open(fname, 'r') as f:
            header = f.readline().split()
        lam = np.unique(np.loadtxt(fname, skiprows=1)[:,header.index('lam')])

        all_frac_res = [None] * getattr(args, f'n{name}')
        for j in range(getattr(args, f'n{name}')):
            all_frac_res[j] = ypred[j*len(lam):(j+1)*len(lam)] / ytrue[j*len(lam):(j+1)*len(lam)]
        all_frac_res = np.array(all_frac_res)
        all_perc = [34+13.5+2.35, 34+13.5, 34]
        all_perc = all_perc[1:]
        for j, delta in enumerate(all_perc):
            low = np.percentile(all_frac_res, 50 - delta, axis=0) - 1
            high = np.percentile(all_frac_res, 50 + delta, axis=0) - 1
            print(f'\t\t{len(all_perc)-j} sigma:', np.amin(low), np.amax(high))
            axs[i].fill_between(lam, low, high, color=cmap(j), label=str(len(all_perc)-j) + r'$\sigma$')
        axs[i].plot(lam, np.median(all_frac_res, axis=0) - 1, color='k')
        rmse = np.sqrt(np.mean((all_frac_res - 1) ** 2))
        print("\t\tRMSE:", rmse)
        rmae = np.mean(np.abs(all_frac_res - 1))
        print("\t\tRMAE:", rmae)

        axs[i].set_xlabel(r'$\lambda \ / \ \lambda_{\rm V}$')
        axs[i].legend()
        axs[i].axhline(0, color='k', ls='--', lw=2)
        axs[i].axhline(0.01, color='k', ls='--', lw=2)
        axs[i].axhline(-0.01, color='k', ls='--', lw=2)

        axs[0].set_title('Training')
        axs[1].set_title('Validation')
        axs[0].set_ylabel(r'Fractional Error on $A$')
    
    fig.align_labels()
    fig.tight_layout()
    
    return fig, axs


def print_to_latex(ini_file):
    """
    Convert operon output to latex and print to file
    
    Args:
        :ini_file (str): The path to the ini file containing the run information
    """
    
    args = OperonArgs(ini_file)
    
    run_name = f'{args.in_param}_{str(args.version_num)}'
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