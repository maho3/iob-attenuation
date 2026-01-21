import numpy as np
from sklearn.metrics import r2_score
import csv
import os
import multiprocessing
from pyoperon.sklearn import SymbolicRegressor
from pyoperon import MSE
from utils import OperonArgs
from os.path import join as pjoin
import argparse
import pandas as pd
import warnings
import outer_term

def run_operon(ini_file):
    """
    Run pyoperon and save results to file. Four files are made:
    1) *_fun.csv contains the list of functions and the model length, rmse and r2 values for
        the training validations sets.
    2) *_train_{length}.csv contains columns of the X values, true y values and predicted y
        values for the training set for the model of length 'length'.
    3) *_val_{length}.csv contains columns of the X values, true y values and predicted y
        values for the validation set for the model of length 'length'.
    4) *_names.txt contains the names of the variables used in the fit, in order X0, X1, ...
    
    Args:
        :ini_file (str): The path to the ini file containing the run information
    """
    
    args = OperonArgs(ini_file)
    target_name = 'A'

    if args.subtract_outer and args.fit_log:
        raise NotImplementedError("Cannot subtract outer term when fitting log(A)")
    
    # Load training and validation data
    dirname = pjoin(args.out_data_dir, f'{args.in_param}_data_{args.version_num}')
    fname_train = pjoin(dirname, f'{args.in_param}_train_data.txt')
    fname_val = pjoin(dirname, f'{args.in_param}_val_data.txt')
    df_train = pd.read_csv(fname_train, sep=r'\s+')
    df_val = pd.read_csv(fname_val, sep=r'\s+')

    in_cols = [col for col in df_train.columns if col.startswith(args.in_param.upper()) and col[len(args.in_param):].isdigit()]
    in_cols += ['lam']
    print('Input columns:', in_cols)

    X = df_train[in_cols].values
    y = df_train[target_name].values
    Xval = df_val[in_cols].values
    yval = df_val[target_name].values

    if args.subtract_outer:
        # Get outer term fit parameter C0
        df_outer_train = pd.read_csv(pjoin(args.out_data_dir, 'outer_fit_results_train.csv'))
        df_outer_val = pd.read_csv(pjoin(args.out_data_dir, 'outer_fit_results_val.csv'))
        C0_train = df_outer_train['C0_opt'].values
        C0_val = df_outer_val['C0_opt'].values
        nx = len(np.unique(df_train['lam'].values))
        C0_train = np.repeat(C0_train, nx)
        C0_val = np.repeat(C0_val, nx)
        
        # Subtract outer term from y values
        print('Subtracting outer term from target values')
        outer_train = outer_term.compute_Av(df_train['lam'].values, C0_train).astype(y.dtype)
        outer_val = outer_term.compute_Av(df_val['lam'].values, C0_val).astype(yval.dtype)
        y -= outer_train
        yval -= outer_val

    # Get the normalisation of the curves
    df_all_train = pd.read_csv(args.selection.train_file)
    df_all_val = pd.read_csv(args.selection.val_file)
    log_Av_col = f'logA_{int(args.lambda_V*1e4)}A'
    A_v_col = f'A_{int(args.lambda_V*1e4)}A'
    if log_Av_col in df_all_train.columns:
        lv_key = log_Av_col
    elif A_v_col in df_all_train.columns:
        lv_key = A_v_col
    else:
        raise ValueError("Column with lambda_V not found in input file")
    
    # Get A_v for training and validation sets so we can calculate dF/F after fitting
    train_id = df_train['galaxy_id'].values
    train_los = df_train['los'].values
    val_id = df_val['galaxy_id'].values
    val_los = df_val['los'].values
    m = df_all_train.set_index(['galaxy_id', 'los'])
    train_Av = m.loc[list(zip(train_id, train_los)), lv_key].values
    m = df_all_val.set_index(['galaxy_id', 'los'])
    val_Av = m.loc[list(zip(val_id, val_los)), lv_key].values

    if 'logA' in lv_key:
        train_Av = 10 ** train_Av
        val_Av = 10 ** val_Av
    if np.all(train_Av == 1) or np.all(val_Av == 1):
        warnings.warn('Av is 1 everywhere')
    assert train_Av.shape[0] == X.shape[0], "Mismatch in training Av and X shape"
    assert val_Av.shape[0] == Xval.shape[0], "Mismatch in validation Av and X shape"

    print('Target:', target_name)
    print('Fitting using parameters:', in_cols)

    print(X.shape, y.shape, y.min(), y.max())

    # If using log and the target is A, then convert to log10A
    # Filter out the non-positive values
    if args.fit_log and target_name == 'A':
        pos_train = y > 0
        pos_val = yval > 0
        X_train_use = X[pos_train,:].copy()
        y_train_use = np.log10(y[pos_train]).copy()
        X_val_use = Xval[pos_val,:].copy()
        y_val_use = np.log10(yval[pos_val]).copy()
        train_Av_use = train_Av[pos_train].copy()
        val_Av_use = val_Av[pos_val].copy()
    else:
        pos_train = np.ones(y.shape[0], dtype=bool)
        pos_val = np.ones(yval.shape[0], dtype=bool)
        X_train_use = X.copy()
        y_train_use = y.copy()
        X_val_use = Xval.copy()
        y_val_use = yval.copy()
        train_Av_use = train_Av.copy()
        val_Av_use = val_Av.copy()

    reg = SymbolicRegressor(
            allowed_symbols=args.allowed_symbols,
            offspring_generator='basic',
            optimizer_iterations=10,
            max_length=args.max_length,
            initialization_method='btc',
            n_threads=multiprocessing.cpu_count(),
            objectives = args.objectives,
            epsilon = args.epsilon,
            random_state=None,
            reinserter='keep-best',
            max_evaluations=args.max_evaluations,
            symbolic_mode=False,
            max_time=args.time_limit,
            generations=args.generations,
            )

    print('Fitting')
    reg.fit(X_train_use, y_train_use)
    print(reg.get_model_string(reg.model_, 2))
    print(reg.stats_)

    # if args.subtract_outer:
    #     print('Re-adding outer term to predictions when saving results')
    #     y += outer_train
    #     yval += outer_val
    #     y_train_use += outer_train[pos_train]
    #     y_val_use += outer_val[pos_val]

    mse = MSE()

    
    # Output directory
    run_name = f'{args.in_param}_{str(args.version_num)}'
    out_dir = pjoin(args.fit_dir, run_name)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # Make backup dir if it doesn't exist
    if not os.path.isdir(f'{out_dir}/backup'):
        os.mkdir(f'{out_dir}/backup')
    
    # Backup output train files
    outname_pred_train = f'{out_dir}/{run_name}_train'
    os.system(f'mv {outname_pred_train}*.csv {out_dir}/backup')
    
    # Backup output validation files
    outname_pred_val = f'{out_dir}/{run_name}_val'
    os.system(f'mv {outname_pred_val}*.csv {out_dir}/backup')

    # File name for functions
    outname = f'{out_dir}/{run_name}_fun.csv'
    os.system(f'mv {outname}*.csv {out_dir}/backup')
    
    # File for names of parameters
    with open(f'{out_dir}/{run_name}_names.txt', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(in_cols + [target_name])

    # Save the outer term results
    if args.subtract_outer:
        df_outer = pd.DataFrame({
        'C0': C0_train,
        'x': df_train['lam'],
        'A_outer': outer_train
        })
        df_outer.to_csv(f'{out_dir}/{run_name}_outer_train.csv', index=False)

        df_outer_val = pd.DataFrame({
        'C0': C0_val,
        'x': df_val['lam'],
        'A_outer': outer_val
        })
        df_outer_val.to_csv(f'{out_dir}/{run_name}_outer_val.csv', index=False)

    res = [(s['tree'],  s['model']) for s in reg.pareto_front_]

    if args.subtract_outer:
        print('Re-adding outer term to predictions when saving results')
        y += outer_train
        yval += outer_val
        y_train_use += outer_train[pos_train]
        y_val_use += outer_val[pos_val]

    with open(outname, "w") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["Equation", "Length", "R2_train", "MSE_train", "MedAE_train_F", "R2_val", "MSE_val", "MedAE_val_F"])
        for model, model_str in res:

            y_pred_train = reg.evaluate_model(model, np.asfortranarray(X))
            if args.subtract_outer:
                y_pred_train += outer_train

            # dF_F = 10. ** (0.4 * A_star * (t - p)) - 1.0
            if args.fit_log:
                A_pred = 10 ** y_pred_train
                A_true = 10 ** y_train_use
            else:
                A_pred = y_pred_train
                A_true = y_train_use


            dF_F_train = 10. ** (0.4 * train_Av_use * (A_true - A_pred[pos_train])) - 1.0

            try:
                mse_train = mse(y_train_use, y_pred_train[pos_train])
            except Exception as e:
                print('Error calculating train mse for model:', model.Length)
                print(e)
                print(y_train_use.dtype, y_pred_train[pos_train].dtype)
                mse_train = np.nan
            try:
                r2_train = r2_score(y_train_use, y_pred_train[pos_train])
            except Exception as e:
                print('Error calculating train r2 for model:', model.Length)
                print(e)
                r2_train = np.nan
            try:
                medae_train_F = float(np.median(np.abs(dF_F_train)))
            except:
                print('Error calculating train MedAE for model:', model.Length)
                medae_train_F = np.nan

            y_pred_val = reg.evaluate_model(model, np.asfortranarray(Xval))
            if args.subtract_outer:
                y_pred_val += outer_val

            # dF_F = 10. ** (0.4 * A_star * (t - p)) - 1.0
            if args.fit_log:
                A_pred = 10 ** y_pred_val
                A_true = 10 ** y_val_use
            else:
                A_pred = y_pred_val
                A_true = y_val_use

            dF_F_val = 10. ** (0.4 * val_Av_use * (A_true - A_pred[pos_val])) - 1.0

            try:
                mse_val = mse(y_val_use, y_pred_val[pos_val])
            except:
                print('Error calculating val mse for model:', model.Length)
                mse_val = np.nan
            try:
                r2_val = r2_score(y_val_use, y_pred_val[pos_val])
            except:
                print('Error calculating val r2 for model:', model.Length)
                r2_val = np.nan
            try:
                medae_val_F = float(np.median(np.abs(dF_F_val)))
            except:
                print('Error calculating val MedAE for model:', model.Length)
                medae_val_F = np.nan

            to_print = [model_str, model.Length, r2_train, mse_train, medae_train_F, r2_val, mse_val, medae_val_F]
            # print(f'\n{to_print[1]}\n{to_print[0]}\n{to_print[2:]}')
            print(f'\n{to_print[1]}\n{to_print[0]}')
            print('MSE train, val:', to_print[3], to_print[6])
            print('R2 train, val:', to_print[2], to_print[5])
            print('MedAE dF/F train, val:', to_print[4], to_print[7])
            writer.writerow(to_print)
        

            # Insert np.nan into missing positions in dF_F arrays
            dF_F_train_full = np.full(y.shape[0], np.nan)
            dF_F_train_full[pos_train] = dF_F_train
            dF_F_val_full = np.full(yval.shape[0], np.nan)
            dF_F_val_full[pos_val] = dF_F_val

            # Exponenitate if needed
            if args.fit_log:
                y_pred_train = 10. ** y_pred_train
                y_pred_val = 10. ** y_pred_val

            output = np.vstack([X.T, y, y_pred_train, dF_F_train_full]).T
            output_val = np.vstack([Xval.T, yval, y_pred_val, dF_F_val_full]).T
            np.savetxt(f'{outname_pred_train}_{model.Length}.csv', output)
            np.savetxt(f'{outname_pred_val}_{model.Length}.csv', output_val)

    print('\nRMSE train: ', np.sqrt(mse_train))
    print('RMSE val: ', np.sqrt(mse_val))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run operon with a specified config file.")
    parser.add_argument("config_path", help="Path to the configuration file.")
    args = parser.parse_args()
    run_operon(args.config_path)

