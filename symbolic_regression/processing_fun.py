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
from matplotlib.lines import Line2D
import re

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
    
    if args.fit_log:
        print('\nTarget: log10A')
    else:
        print('\nTarget: A')
    
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
    pretty_eq, pars = convert_operon_fun(best_eq, names)
    pretty_eq = sympy.sympify(pretty_eq)
    display(pretty_eq)
    print('\nLatex version:')
    sympy.print_latex(pretty_eq)

    # Make a reparameterised version
    global_vals = {f'b{i}': pars[i] for i in range(len(pars))}
    print(eq)
    reparameterise(eq, global_vals, xname='lam', old_local_prefix=args.in_param.upper(), global_prefix='a', local_prefix='B')
    
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
    if args.fit_log:
        ax.set_title('Fit to log10A')
    else:
        ax.set_title('Fit to A')
    
    fig.align_labels()
    fig.tight_layout()

    return fig, ax


def prediction_plots(ini_file, ilen=None, plot_frac_error=True):
    """
    Show the difference between the truth and predicted
    
    Args:
        :ini_file (str): The path to the ini file containing the run information
        :ilen (int, default=None): The length of the equation to highlight. If None,
            then this is taken to be the final equation
        :plot_frac_error (bool, default=True): Whether to plot the fractional error
            (True) or absolute error (False)
            
    Returns:
        :fig (matplotlib.figure.Figure): Figure containing plot
        :axs (np.ndarray[matplotlib.pyplot.axis]): Axes of fig containing the plot
    """
    
    args = OperonArgs(ini_file)
    
    run_name = f'{args.in_param}_{str(args.version_num)}'
    out_dir = pjoin(args.fit_dir, run_name)
    fname = f'{out_dir}/{run_name}_fun.csv'
    df = pd.read_csv(fname, delimiter=';')
    
    if args.fit_log:
        print('\nTarget: log10A')
    else:
        print('\nTarget: A')
    
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
        
        if args.fit_log:
            ypred = 10. ** ypred
            ytrue = 10. ** ytrue

        fname = pjoin(args.data_dir, f'{args.in_param}_data_{args.version_num}', f'{args.in_param}_train_data.txt')
        with open(fname, 'r') as f:
            header = f.readline().split()
        lam = np.unique(np.loadtxt(fname, skiprows=1)[:,header.index('lam')])

        all_frac_res = [None] * getattr(args, f'n{name}')
        for j in range(getattr(args, f'n{name}')):
            if plot_frac_error:
                all_frac_res[j] = ypred[j*len(lam):(j+1)*len(lam)] / ytrue[j*len(lam):(j+1)*len(lam)] - 1
            else:
                all_frac_res[j] = ypred[j*len(lam):(j+1)*len(lam)] - ytrue[j*len(lam):(j+1)*len(lam)]
        all_frac_res = np.array(all_frac_res)
        all_perc = [34+13.5+2.35, 34+13.5, 34]
        all_perc = all_perc[1:]
        for j, delta in enumerate(all_perc):
            low = np.percentile(all_frac_res, 50 - delta, axis=0) 
            high = np.percentile(all_frac_res, 50 + delta, axis=0)
            print(f'\t\t{len(all_perc)-j} sigma:', np.amin(low), np.amax(high))
            axs[i].fill_between(lam, low, high, color=cmap(j), label=str(len(all_perc)-j) + r'$\sigma$')
        axs[i].plot(lam, np.median(all_frac_res, axis=0), color='k')
        rmse = np.sqrt(np.mean((all_frac_res) ** 2))
        print("\t\tRMSE:", rmse)
        rmae = np.mean(np.abs(all_frac_res))
        print("\t\tRMAE:", rmae)

        axs[i].set_xlabel(r'$\lambda \ / \ \lambda_{\rm V}$')
        axs[i].legend()
        axs[i].axhline(0, color='k', ls='--', lw=2)
        axs[i].axhline(0.01, color='k', ls='--', lw=2)
        axs[i].axhline(-0.01, color='k', ls='--', lw=2)

        axs[0].set_title('Training')
        axs[1].set_title('Validation')
        if plot_frac_error:
            axs[0].set_ylabel(r'Fractional Error on $A$')
        else:
            axs[0].set_ylabel(r'Absolute Error on $A$')
    
    fig.align_labels()
    fig.tight_layout()
    
    return fig, axs


def plot_example(ini_file, ilen=None, nexamples=5):
    """
    Plot an example curve
    
    Args:
        :ini_file (str): The path to the ini file containing the run information
        :ilen (int, default=None): The length of the equation to highlight. If None,
            then this is taken to be the final equation
        :nexamples (int, default=5): Number of examples to plot
            
    Returns:
        :fig (matplotlib.figure.Figure): Figure containing plot
        :axs (np.ndarray[matplotlib.pyplot.axis]): Axes of fig containing the plot
    """
    
    args = OperonArgs(ini_file)
    
    run_name = f'{args.in_param}_{str(args.version_num)}'
    out_dir = pjoin(args.fit_dir, run_name)
    fname = f'{out_dir}/{run_name}_fun.csv'
    df = pd.read_csv(fname, delimiter=';')
    
    if args.fit_log:
        print('\nTarget: log10A')
    else:
        print('\nTarget: A')
    
    if ilen is None:
        eq_idx = -1
    else:
        eq_idx = list(df['Length']).index(ilen)
    length = list(df['Length'])[eq_idx]
    
    cmap = plt.get_cmap('Set1')
    rcParams['font.size'] = 16
    rcParams["text.usetex"] = True
        
    fig, axs = plt.subplots(1, 2, figsize=(15,6), sharex=True)

    for i, name in enumerate(['train', 'val']):

        fname= f'{out_dir}/{run_name}_{name}_{length}.csv'
        data = np.loadtxt(fname)
        ytrue = data[:,-2]
        ypred = data[:,-1]
        
        if args.fit_log:
            ytrue = 10. ** ytrue
            ypred = 10. ** ypred
        
        fname = pjoin(args.data_dir, f'{args.in_param}_data_{args.version_num}', f'{args.in_param}_train_data.txt')
        with open(fname, 'r') as f:
            header = f.readline().split()
        lam = np.unique(np.loadtxt(fname, skiprows=1)[:,header.index('lam')])
        
        for j in range(nexamples):
            c = f'C{j}'
            axs[i].plot(lam, ytrue[j*len(lam):(j+1)*len(lam)], color=c, ls='--')
            axs[i].plot(lam, ypred[j*len(lam):(j+1)*len(lam)], color=c)

        axs[i].set_xlabel(r'$\lambda \ / \ \lambda_{\rm V}$')
        axs[i].set_ylabel(r'$A$')
        axs[i].set_ylim(0, None)
                
    custom_lines = [Line2D([0], [0], color='k', lw=2, ls='--'),
                Line2D([0], [0], color='k', lw=2, ls='-')]
    for ax in axs:
        ax.legend(custom_lines, ['True', 'Predicted'])
    axs[0].set_title('Training')
    axs[1].set_title('Validation')
        
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


def reparameterise(expr_str, global_vals, xname='x', old_local_prefix='IOB', global_prefix='a', local_prefix='b'):
    """
    Reparameterise the expression to only keep as a function of a single variable, as well as local and global parameters.
    This function will replace all x-independent parts of the expression by parameters, and rename the parameters
    to the desired prefixes.

    Args:
        :expr_str (str): The expression to reparameterise, as a string
        :global_vals (dict): A dictionary containing the values of the global parameters, where the
            keys are the parameter names and the values are the parameter values.
        :xname (str, default='x'): The name of the variable in the expression
        :old_local_prefix (str, default='IOB'): The prefix for the local parameters in the input expression
        :global_prefix (str, default='a'): The prefix for the global parameters in the output expression
        :local_prefix (str, default='b'): The prefix for the local parameters in the output expression
    """

    print('\n' + '--'*100)
    print('Reparameterising expression:')

    # Store the dictionary of all variables values
    vals = global_vals.copy()
    
    # Parse the expression
    expr = sympy.sympify(expr_str)
    x = sympy.Symbol(xname)

    # Find an appropriate temporary prefix for the parameters
    symbols = list(expr.free_symbols)
    def get_alpha_prefix(s):
        match = re.match(r'[a-zA-Z]+', str(s))
        return match.group(0) if match else ''
    prefixes = {get_alpha_prefix(s) for s in symbols}
    symbols = [s for s in symbols if s != x] # Remove the variable x
    print(f'Original number of parameters: {len(symbols)}')
    temp_global_prefix = next(ell for ell in string.ascii_lowercase if ell not in prefixes and ell not in ['e', 'x', global_prefix, local_prefix])
    if global_prefix == local_prefix:
        temp_local_prefix = temp_global_prefix
    else:
        temp_local_prefix = next(ell for ell in string.ascii_lowercase if ell not in prefixes and ell not in ['e', 'x', global_prefix, local_prefix, temp_global_prefix])
    print(f'Temporary global prefix: {temp_global_prefix}, temporary local prefix: {temp_local_prefix}')

    # Get old local parameters
    local_pars = [str(s) for s in symbols if str(s).startswith(old_local_prefix)]
    local_pars.sort()
    print(f'Old local parameters: {local_pars}')
    
    # Collect all symbols except x
    symbols = expr.free_symbols - {x}
    
    # Replacement dictionary: subexpr → new param
    replacements = {}

    def has_local_pars(s):
        ep = s.free_symbols
        ep = [str(e) in local_pars for e in ep]
        return any(ep)

    def try_update(expr, param_counter):
        for node in sympy.preorder_traversal(expr):
            if isinstance(node, sympy.Mul):
                factors = node.args
                dependent = [f for f in factors if x in f.free_symbols and f.free_symbols]
                independent = [f for f in factors if x not in f.free_symbols and (f.free_symbols or f.is_number)]
                if len(dependent) + len(independent) != len(factors):
                    raise ValueError(f"Some factors are not accounted for: {independent}, {dependent}, {factors}")
                if len(independent) > 1:
                    subexpr = sympy.Mul(*independent)
                    if (subexpr not in replacements):
                        hl = has_local_pars(subexpr)
                        if hl:
                            replacements[subexpr] = sympy.Symbol(f'{temp_local_prefix}{param_counter}')
                        else:
                            replacements[subexpr] = sympy.Symbol(f'{temp_global_prefix}{param_counter}')
                        param_counter += 1
                        new_mul = sympy.Mul(replacements[subexpr], *dependent)
                        expr = expr.replace(node, new_mul)
                        if (not hl) and (x not in subexpr.free_symbols):
                            vals[str(replacements[subexpr])] = subexpr.subs(vals)
                        return expr, param_counter, False
            elif isinstance(node, sympy.Pow):
                base, exp = node.args
                if (
                    x not in base.free_symbols and base.free_symbols and
                    x not in exp.free_symbols and exp.free_symbols and
                    node not in replacements
                ):
                    hl = has_local_pars(base) or has_local_pars(exp) 
                    if hl:
                        replacements[node] = sympy.Symbol(f'{temp_local_prefix}{param_counter}')
                    else:
                        replacements[node] = sympy.Symbol(f'{temp_global_prefix}{param_counter}')
                    param_counter += 1
                    expr = expr.replace(node, replacements[node])
                    if (not hl) and (x not in new_mul.free_symbols):
                        vals[str(replacements[node])] = node.subs(vals)
                    return expr, param_counter, False
            elif isinstance(node, sympy.Add):
                terms = node.args
                dependent = [f for f in terms if x in f.free_symbols and f.free_symbols]
                independent = [t for t in terms if x not in t.free_symbols and (t.free_symbols or t.is_number)]
                if len(dependent) + len(independent) != len(terms):
                    raise ValueError(f"Some terms are not accounted for: {independent}, {dependent}, {terms}")
                if len(independent) > 1:
                    subexpr = sympy.Add(*independent, evaluate=False)
                    if subexpr not in replacements:
                        hl = has_local_pars(subexpr)
                        if hl:
                            replacements[subexpr] = sympy.Symbol(f'{temp_local_prefix}{param_counter}')
                        else:
                            replacements[subexpr] = sympy.Symbol(f'{temp_global_prefix}{param_counter}')
                        param_counter += 1
                        new_add = sympy.Add(replacements[subexpr], *dependent, evaluate=False)
                        expr = expr.replace(node, new_add)
                        if (not hl) and (x not in subexpr.free_symbols):
                            vals[str(replacements[subexpr])] = subexpr.subs(vals)
                        return expr, param_counter, False
        return expr, param_counter, True

    # Step 1: look for Mul, Pow and Add expressions and extract x-independent parts
    is_same = False
    param_counter = 0
    while not is_same:
        expr, param_counter, is_same = try_update(expr, param_counter)
        if param_counter > 100:
            print("Too many parameters, stopping to avoid infinite loop.")
            break

    # Step 2: rename all remaining symbols that are not x
    for s in symbols:
        if s not in replacements:
            if has_local_pars(s):
                replacements[s] = sympy.Symbol(f'{temp_local_prefix}{param_counter}')
            else:
                replacements[s] = sympy.Symbol(f'{temp_global_prefix}{param_counter}')
                vals[str(replacements[s])] = vals[str(s)]
            param_counter += 1
            expr = expr.replace(s, replacements[s])

    # Step 3: Rename the parameters to the desired prefix
    symbols = list(expr.free_symbols)
    symbols.remove(sympy.Symbol(xname)) 
    symbols.sort(key=lambda s: str(s))
    replacements = {}
    final_vals = []
    nlocal = 0
    nglobal = 0
    for s in symbols:
        if str(s).startswith(temp_local_prefix):
            replacements[s] = sympy.Symbol(f'{local_prefix}{nlocal}')
            nlocal += 1
        elif str(s).startswith(temp_global_prefix):
            replacements[s] = sympy.Symbol(f'{global_prefix}{nglobal}')
            final_vals.append(vals[str(s)])
            nglobal += 1
        else:
            raise ValueError(f"Unexpected symbol: {s}")
    expr = expr.subs(replacements)
    print('Final number of parameters:', len(symbols))
    print('Of which global:', nglobal, 'and local:', nlocal)
    print('Final values:', final_vals)

    expr = expr.replace(sympy.Symbol('lam'), sympy.Symbol('x'))
    display(expr)
    print('--'*100 + '\n')

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