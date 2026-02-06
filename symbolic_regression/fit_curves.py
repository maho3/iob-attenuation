import numpy as np
import pandas as pd
import sys
import re
from tqdm import tqdm
import os

from utils import OperonArgs
import outer_term

sys.path.insert(0, '../literature_fits')
import attenuation_curves, fit_literature

def load_data(args, name):
    """
    Load the data for D parameters.

    Args:
        :args (OperonArgs): Parsed ini file arguments.
        :name (str): 'train' or 'val' to specify which dataset to load

    Returns:
        :x (np.ndarray): Wavelengths normalized by lambda_V, shape (nlam,).
        :ytrue (np.ndarray): True attenuation curves normalized by A_V, shape (ngal, nlam).
        :D (np.ndarray): Initial D parameters for each galaxy, shape (ngal, 4).
        :galaxy_ids (np.ndarray): Galaxy IDs, shape (ngal,).
        :los (np.ndarray): Lines of sight, shape (ngal,).
        :Av (np.ndarray): A_V values for each galaxy, shape (ngal,).
    """

    fname = args.selection.train_file if name == 'train' else args.selection.val_file
    run_name = f'{args.in_param}_{str(args.version_num)}'
    fname_outer = f'{args.fit_dir}/{run_name}/{run_name}_outer_{name}.csv'

    df_orig = pd.read_csv(fname)
    print(f'Loaded original data from {fname} with {df_orig.shape[0]} galaxies.')
    if df_orig.shape[0] == 0:
        print(f'No galaxies found in {fname}. Skipping.')
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # Load data with C0
    run_name = f'{args.in_param}_{str(args.version_num)}'
    if not os.path.isfile(fname_outer):
        _, C0, _, _ = outer_term.run_all_gals(args, name)
        nx = len(C0)
    else:
        df_outer = pd.read_csv(fname_outer)
        nx = len(set(df_outer['x'].values))
        assert int(df_outer.shape[0] / nx) == df_orig.shape[0]
        C0 = df_outer['C0'].values[::nx]

    # Get initial D values
    # {B0: c13*(IOB1*c14 + c15), B1: IOB2*c16, B2: IOB1*c3 - IOB3*c4, B3: (IOB2*c6 + exp(IOB1*c7))*exp(-IOB3*c10), B4: -IOB2*c9, B5: -IOB2*c11}
    c = [1.000648, 45.120201, 208.741272, 2.214609, 11.210193, 1.380509, 22.421267, 5.736165, 3.953059, 5.040799, 7.073277, 3.570695, 9.125808, 1.000648, 21.685377, 1.474194, 1.215716, 16.786709, 6.730947, 0.002466]
    a = [-0.00246600000000000, -9.12580800000000, 45.120201, 1.380509, -6.73094700000000, 16.786709, 1.000648, -208.741272000000, -3.95305900000000]
    B0 = c[13] * (df_orig['IOB1'].values * c[14] + c[15])
    B1 = df_orig['IOB2'].values * c[16]
    B2 = c[3] * df_orig['IOB1'].values - c[4] * df_orig['IOB3'].values
    B3 = (c[6] * df_orig['IOB2'].values + np.exp(c[7] * df_orig['IOB1'].values)) * np.exp(-c[10] * df_orig['IOB3'].values)
    B4 = -c[9] * df_orig['IOB2'].values
    B5 = -c[11] * df_orig['IOB2'].values
    D0 = B0 * np.exp(B1)
    D1 = np.exp(B5) * (B2 + B3 * np.exp(B4))
    D2 = (a[8] * B3 + a[3]) * np.exp(B5)

    # Stack the D parameters
    D = np.vstack((D0, D1, D2, C0)).T  # shape (ngal, 4)

    # Get the true data
    attenuation_cols = [col for col in df_orig.columns if re.match(r'A_\d+A', col)]
    lam_arr = np.array([int(re.search(r'_(\d+)A', col).group(1)) / 1e4 for col in attenuation_cols])
    sort_idx = np.argsort(lam_arr)
    lam_arr = lam_arr[sort_idx]
    mask = (lam_arr < args.lam_max) & (lam_arr > args.lam_min)
    lam_arr = lam_arr[mask]
    attenuation_cols = [attenuation_cols[i] for i in sort_idx if mask[i]]
    A_v_col = f'A_{int(args.lambda_V*1e4)}A'
    x = lam_arr / args.lambda_V
    curves_flat = df_orig[attenuation_cols].values
    Av = df_orig[A_v_col].values
    ytrue = curves_flat / Av[:,None]

    galaxy_ids = df_orig['galaxy_id'].values.astype(int)
    los = df_orig['los'].values.astype(int)

    return x, ytrue, D, galaxy_ids, los, Av


def new_model(x, D0, D1, D2, D3):
    """
    The model to be optimised

    Args:
        x: (N,) input array of lamda/lambda_V
        D0, D1, D2, D3: Scalars representing the D parameters for this galaxy

    Returns:
        y: (N,) output array of predicted A(lambda)/A_V
    """

    b = [1.015638351440429688e+00, -9.222948074340820312e+00, 4.451240158081054688e+01,
         -2.127728118896484375e+02, 4.002230465412139893e-01, 2.856298522949218750e+02]

    y = (
        D0 * (np.exp(-b[5] * (x - b[4])**2) - np.exp(-b[5] * (1. - b[4])**2))
        + (b[2] + b[3] * x) * (D1 + D2 * x) * (np.exp(b[1] * x) - np.exp(b[1]))
        + np.exp(D3 * (np.tanh(b[0]) - np.tanh(b[0] * x)))
    )

    return y


def new_model_reparam(x, A0, A1, A2, A3):
    """
    The model to be optimised

    Args:
        x: (N,) input array of lamda/lambda_V
        A0, A1, A2, A3: Scalars representing the A parameters for this galaxy

    Returns:
        y: (N,) output array of predicted A(lambda)/A_V
    """

    c = [0.4002, 285.6, 0.2092, 9.223, 1.016]

    y = (
        A0 * (np.exp(-c[1] * (x - c[0])**2) - np.exp(-c[1] * (1. - c[0])**2))
        + (A1 + A2 * (x - c[2])) * (x - c[2]) * (np.exp(-c[3] * x) - np.exp(-c[3]))
        + np.exp(A3 * (np.tanh(c[4]) - np.tanh(c[4] * x)))
    )

    return y


def optimise_all_fun(lam, Alam_Av, D_init, gal_id, los, lam_min, lam_max, lambda_v=0.5542):
    """
    Optimise all three functions for a given galaxy and line of sight.

    Args:
        :lam: (N,) array of wavelengths in microns
        :Alam_Av: (N,) array of A(lambda)/A_V values
        :D_init: (4,) array of initial D parameters for the new model
        :gal_id: (int) galaxy ID
        :los: (int) line of sight
        :lam_min: (float) minimum wavelength in microns for fitting
        :lam_max: (float) maximum wavelength in microns for fitting
    """

    m = (lam < lam_max) & (lam > lam_min)
    lam_cut = lam[m]
    x_cut = lam_cut / lambda_v
    Alam_Av_arr_cut = Alam_Av[m]

    # Initial guesses for 4-parameter fit
    all_p0_4par = [
        [44.9, 7.56, 61.2, 0.],  # 'Calzetti'
        [38.7, 3.83, 6.34, 0.], #'SMC'
        [14.4, 6.52, 2.04, 0.0519], # 'MW'
        [4.47, 2.39, -0.988, 0.0221], # 'LMC'
        [1.0, 1.0, 1.0, 1.0],  # Generic initial guess
    ]

    # 4 parameter fit
    try:
        popt_4par, success_4par = fit_literature.run_fit(
                        attenuation_curves.Li_08_fit_noratio, 
                        lam_cut, Alam_Av_arr_cut, 
                        bounds=([-np.inf,-np.inf,-np.inf,0],[np.inf,np.inf,np.inf,1.]),
                        all_p0=all_p0_4par)
        fit_nb = attenuation_curves.Li_08_fit_noratio(lam_cut,*popt_4par)
        rmse_4par = np.sqrt(np.mean((Alam_Av_arr_cut - fit_nb)**2))
    except RuntimeError as e:
        print(f"Galaxy ID: {gal_id}, LoS: {los}, 4par fit failed with error: {e}")
        popt_4par = [None] * 4
        success_4par = False
        rmse_4par = None
    res_4par = {'params': popt_4par, 'rmse': rmse_4par, 'success': success_4par}

    # 2 parameter fit
    try:
        popt_2par, success_2par = fit_literature.run_fit(
                            attenuation_curves.Att_Curve_2param, 
                            1e4*lam_cut, Alam_Av_arr_cut, 
                            bounds = ([-np.inf,-np.inf],[np.inf,np.inf]))
        fit_nb = attenuation_curves.Att_Curve_2param(1e4*lam_cut,B=popt_2par[0],delta=popt_2par[1])
        rmse_2par = np.sqrt(np.mean((Alam_Av_arr_cut - fit_nb)**2))
    except RuntimeError as e:
        print(f"Galaxy ID: {gal_id}, LoS: {los}, 2par fit failed with error: {e}")
        popt_2par = [None, None]   
        success_2par = False 
        rmse_2par = None
    res_2par = {'params': popt_2par, 'rmse': rmse_2par, 'success': success_2par}

    # My function
    try:
        popt_newpar, success_newpar = fit_literature.run_fit(
                            new_model,
                            x_cut, Alam_Av_arr_cut, 
                            bounds=([0,-np.inf,-np.inf,0],[np.inf,np.inf,np.inf,np.inf]),
                            all_p0 = [D_init.tolist()])
        fit_nb = new_model(x_cut, *popt_newpar)
        rmse_newpar = np.sqrt(np.mean((Alam_Av_arr_cut - fit_nb)**2))
    except RuntimeError as e:
        print(f"Galaxy ID: {gal_id}, LoS: {los}, My function fit failed with error: {e}")
        popt_newpar = [None, None, None, None]   
        success_newpar = False 
        rmse_newpar = None
    res_newpar = {'params': popt_newpar, 'rmse': rmse_newpar, 'success': success_newpar}

    # Reparameterised version of my function
    b = [1.015638351440429688e+00, -9.222948074340820312e+00, 4.451240158081054688e+01,
         -2.127728118896484375e+02, 4.002230465412139893e-01, 2.856298522949218750e+02]
    c = [0.4002, 285.6, 0.2092, -9.223, 1.016]
    A_init = np.empty(4)
    A_init[0] = D_init[0]
    A_init[1] = b[3] * (D_init[1] + c[2] * D_init[2])
    A_init[2] = b[3] * D_init[2]
    A_init[3] = D_init[3]
    try:
        popt_newpar, success_newpar = fit_literature.run_fit(
                            new_model_reparam,
                            x_cut, Alam_Av_arr_cut, 
                            bounds=([0,-np.inf,-np.inf,0],[np.inf,np.inf,np.inf,np.inf]),
                            all_p0 = [A_init.tolist()])
        fit_nb = new_model_reparam(x_cut, *popt_newpar)
        rmse_newpar = np.sqrt(np.mean((Alam_Av_arr_cut - fit_nb)**2))
    except RuntimeError as e:
        print(f"Galaxy ID: {gal_id}, LoS: {los}, My function reparam. fit failed with error: {e}")
        popt_newpar = [None, None, None, None]   
        success_newpar = False 
        rmse_newpar = None
    res_newpar_reparam = {'params': popt_newpar, 'rmse': rmse_newpar, 'success': success_newpar}

    return res_2par, res_4par, res_newpar, res_newpar_reparam


def run_all_gals(args, name):
    """
    Run optimisation for all galaxies in the dataset.

    Args:
        :args: (OperonArgs) Parsed ini file arguments.
        :name: (str) 'train' or 'val' to specify which dataset to
    """

    print(f'\nRunning optimisation for all galaxies in {name} set.')

    xtrue, ytrue, D, galaxy_ids, los, _ = load_data(args, name)
    lam_true = xtrue * args.lambda_V
    # lam_min = 0.12
    # lam_max = 1.0
    lam_min = args.lam_min
    lam_max = args.lam_max

    results = []

    for i in tqdm(range(len(galaxy_ids))):
        res_2par, res_4par, res_newpar, res_newpar_reparam = optimise_all_fun(
            lam_true, ytrue[i,:], D[i,:], galaxy_ids[i], los[i], lam_min, lam_max, args.lambda_V)
        
        results.append({
            'galaxy_id': galaxy_ids[i],
            'los': los[i],
            'params_2par': res_2par["params"],
            'rmse_2par': res_2par["rmse"],
            'success_2par': res_2par["success"],
            'params_4par': res_4par["params"],
            'rmse_4par': res_4par["rmse"],
            'success_4par': res_4par["success"],
            'params_newpar': res_newpar["params"],
            'rmse_newpar': res_newpar["rmse"],
            'success_newpar': res_newpar["success"],
            'params_newpar_reparam': res_newpar_reparam["params"],
            'rmse_newpar_reparam': res_newpar_reparam["rmse"],
            'success_newpar_reparam': res_newpar_reparam["success"]
        })

    if len(galaxy_ids) == 0:
        df_results = pd.DataFrame(columns=[
            'galaxy_id', 'los', 'params_2par', 'rmse_2par', 'success_2par',
            'params_4par', 'rmse_4par', 'success_4par',
            'params_newpar', 'rmse_newpar', 'success_newpar',
            'params_newpar_reparam', 'rmse_newpar_reparam', 'success_newpar_reparam'
        ])
    else:

        # Create dataframe from results
        df_results = pd.DataFrame(results)

        # Expand parameter lists into separate columns
        df_results[['2par_B', '2par_delta']] = pd.DataFrame(df_results['params_2par'].tolist(), index=df_results.index)
        df_results[['4par_p0', '4par_p1', '4par_p2', '4par_p3']] = pd.DataFrame(df_results['params_4par'].tolist(), index=df_results.index)
        df_results[['newpar_D0', 'newpar_D1', 'newpar_D2', 'newpar_D3']] = pd.DataFrame(df_results['params_newpar'].tolist(), index=df_results.index)
        df_results[['newpar_reparam_A0', 'newpar_reparam_A1', 'newpar_reparam_A2', 'newpar_reparam_A3']] = pd.DataFrame(df_results['params_newpar_reparam'].tolist(), index=df_results.index)

        # Drop the original parameter list columns
        df_results = df_results.drop(columns=['params_2par', 'params_4par', 'params_newpar', 'params_newpar_reparam'])

    return df_results


def main():

    # ini_file = 'conf/iob_44.ini'
    ini_file = 'conf/iob_47.ini'
    args = OperonArgs(ini_file)

    for name in ['train', 'val']:
        df_results = run_all_gals(args, name)
        run_name = f'{args.in_param}_{str(args.version_num)}'
        dirname = f'{args.fit_dir}/{run_name}'
        os.makedirs(dirname, exist_ok=True)
        fname_out = f'{dirname}/{run_name}_opt_results_{name}.csv'
        df_results.to_csv(fname_out, index=False)
        print(f'Saved results to {fname_out}')


if __name__ == '__main__':
    main()