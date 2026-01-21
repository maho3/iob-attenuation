import numpy as np
import pandas as pd
import re
from os.path import join as pjoin
from scipy.optimize import minimize
from tqdm import tqdm
from utils import OperonArgs

def compute_initial_C(IOB):
    c = [0.908373, 0.748169, 0.944124, 59.20784, 5.327414, 0.010842]
    B0 = c[0]*np.log(IOB[0]*c[3] + c[4])
    C0 = B0 * np.log(10)
    return C0

def compute_Av(x, C0):
    b = [0.9753444117120513, 0.944124]
    A_over_Av = b[0] * np.exp(C0*(np.tanh(b[1]) - np.tanh(b[1]*x)))
    return A_over_Av

def loss_fun_b(C0, x, A_true):
    A_pred = compute_Av(x, C0)
    loss = np.sqrt(np.mean((A_true - A_pred)**2))
    return loss


def load_data(args, run):
    
    dirname = pjoin(args.out_data_dir, f'{args.in_param}_data_{args.version_num}')
    fname = pjoin(dirname, f'{args.in_param}_{run}_data.txt')
    df = pd.read_csv(fname, sep=r'\s+')

    # Extract IOB values and variables we care about
    IOB_cols = sorted([col for col in df.columns if re.match(r'IOB\d+', col)])
    IOB = df[IOB_cols].values
    x = df['lam'].values

    # Reshape so each galaxy is its own row
    nx = len(np.unique(x))
    n_gal = IOB.shape[0] // nx
    IOB_reshaped = IOB.reshape((n_gal, nx, len(IOB_cols)))
    x_reshaped = x.reshape((n_gal, nx))
    A_reshaped = df['A'].values.reshape((n_gal, nx))

    # Every x value for a given gal should have the same IOB values
    assert np.allclose(IOB_reshaped[:, :, 0], IOB_reshaped[:, 0:1, 0])
    IOB_reshaped = IOB_reshaped[:, 0, :]

    # All the x rows should be the same
    assert np.allclose(x_reshaped, x_reshaped[0:1, :])
    x_reshaped = x_reshaped[0, :]

    print(f'Loaded {n_gal} galaxies with {nx} x values each for {run} data')

    # Get initial guess for C0 for each galaxy
    C0_init = compute_initial_C(IOB_reshaped.T)

    return x_reshaped, C0_init, A_reshaped


def optimise_gal(x, C0_init, A_true):

    loss_init = loss_fun_b(C0_init, x, A_true)
    res = minimize(loss_fun_b, C0_init, args=(x, A_true), method='Nelder-Mead')
    C0_opt = res.x[0]
    if not res.success:
        print('Warning: optimisation did not converge:', res.message)
    loss_opt = loss_fun_b(C0_opt, x, A_true)

    return C0_opt, loss_init, loss_opt


def run_all_gals(args, run):

    x, C0_init, A_true = load_data(args, run)
    n_gal = C0_init.shape[0]
    C0_opt = np.zeros_like(C0_init)
    loss_init = np.zeros(n_gal)
    loss_opt = np.zeros(n_gal)

    for i in tqdm(range(n_gal)):
        C0_opt[i], loss_init[i], loss_opt[i] = optimise_gal(x, C0_init[i], A_true[i])

    return C0_init, C0_opt, loss_init, loss_opt

def main():

    ini_file = 'conf/iob_35.ini'
    args = OperonArgs(ini_file)

    for run in ['train', 'val']:
        C0_init, C0_opt, loss_init, loss_opt = run_all_gals(args, run)
        print(args.out_data_dir, f'Run: {run}')

        # Combine results into a dataframe and save
        df_results = pd.DataFrame({
            'C0_init': C0_init,
            'C0_opt': C0_opt,
            'loss_init': loss_init,
            'loss_opt': loss_opt
        })
        df_results.to_csv(pjoin(args.out_data_dir, f'outer_fit_results_{run}.csv'), index=False)

    return

if __name__ == '__main__':
    main()