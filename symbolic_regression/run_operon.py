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
    
    # Load training and validation data
    dirname = pjoin(args.data_dir, f'{args.in_param}_data_{args.version_num}')
    fname_train = pjoin(dirname, f'{args.in_param}_train_data.txt')
    fname_val = pjoin(dirname, f'{args.in_param}_val_data.txt')
    fname_train_id = pjoin(args.data_dir, f'{args.in_param}_data_{args.version_num}', f'{args.in_param}_train_galaxy_ids_los.txt')
    fname_val_id = pjoin(args.data_dir, f'{args.in_param}_data_{args.version_num}', f'{args.in_param}_val_galaxy_ids_los.txt')

    with open(fname_train, 'r') as f:
        names = f.readline().strip().split()
    data = np.loadtxt(fname_train, skiprows=1)
    X = data[:,:-1]
    y = data[:,-1]
    train_id, train_los = np.loadtxt(fname_train_id, dtype=float, unpack=True, skiprows=1)

    with open(fname_val, 'r') as f:
        val_names = f.readline().strip().split()
    data = np.loadtxt(fname_val, skiprows=1)
    Xval = data[:,:-1]
    yval = data[:,-1]
    val_id, val_los = np.loadtxt(fname_val_id, dtype=float, unpack=True, skiprows=1)

    # Get the normalisation of the curves
    data = pd.read_csv(args.input_file, sep='\t',)
    if f'logA_{int(args.lambda_V*1e4)}A' in data.keys():
        lv_key = f'logA_{int(args.lambda_V*1e4)}A'
    elif f'A_{int(args.lambda_V*1e4)}A' in data.keys():
        lv_key = f'A_{int(args.lambda_V*1e4)}A'
    else:
        raise ValueError("Column with lambda_V not found in input file")
    
    # Get A_v for training and validation sets so we can calculate dF/F after fitting
    data['_key'] = list(zip(data['galaxy_id'], data['los']))
    data = data.drop_duplicates('_key', keep='first')
    target_pairs = list(zip(train_id, train_los))
    filtered = data[data['_key'].isin(target_pairs)]
    ordered_train = filtered.set_index('_key').loc[target_pairs].reset_index(drop=True)
    train_Av = ordered_train[lv_key].values
    target_pairs = list(zip(val_id, val_los))
    filtered = data[data['_key'].isin(target_pairs)]
    ordered_val = filtered.set_index('_key').loc[target_pairs].reset_index(drop=True)
    val_Av = ordered_val[lv_key].values
    if 'logA' in lv_key:
        train_Av = 10 ** train_Av
        val_Av = 10 ** val_Av
    if np.all(train_Av == 1) or np.all(val_Av == 1):
        print('Warning: Av is 1 everywhere. Will not calculate dF/F.')
        do_dF_F = False
    else:
        do_dF_F = True
    train_Av = np.repeat(train_Av, X.shape[0] // train_Av.shape[0], axis=0)
    val_Av = np.repeat(val_Av, Xval.shape[0] // val_Av.shape[0], axis=0)
    assert train_Av.shape[0] == X.shape[0], "Mismatch in training Av and X shape"
    assert val_Av.shape[0] == Xval.shape[0], "Mismatch in validation Av and X shape"

    use_names = names[:-1]
    print('Target:', names[-1])
    print('Fitting using parameters:', use_names)
    
    # Check arguments
    if args.fit_log:
        assert names[-1] == 'log10A', "Mismatch between config file target and that of file"
    else:
        assert names[-1] == 'A', "Mismatch between config file target and that of file"

    assert names == val_names, 'Training and validation data have different names'

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
    reg.fit(X, y)
    print(reg.get_model_string(reg.model_, 2))
    print(reg.stats_)

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
        writer.writerow(use_names)

    res = [(s['tree'],  s['model']) for s in reg.pareto_front_]

    with open(outname, "w") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["Equation", "Length", "R2_train", "MSE_train", "MedAE_train_F", "R2_val", "MSE_val", "MedAE_val_F"])
        for model, model_str in res:

            y_pred_train = reg.evaluate_model(model, np.asfortranarray(X))

            # dF_F = 10. ** (0.4 * A_star * (t - p)) - 1.0
            if do_dF_F:
                if args.fit_log:
                    A_pred = 10 ** y_pred_train
                    A_true = 10 ** y
                else:
                    A_pred = y_pred_train
                    A_true = y
                dF_F_train = 10. ** (0.4 * train_Av * (A_true - A_pred)) - 1.0
            else:
                dF_F_train = np.full_like(y_pred_train, np.nan)

            try:
                mse_train = mse(y, y_pred_train)
            except:
                mse_train = np.nan
            try:
                r2_train = r2_score(y, y_pred_train)
            except:
                r2_train = np.nan
            if do_dF_F:
                try:
                    medae_train_F = float(np.median(np.abs(dF_F_train)))
                except:
                    medae_train_F = np.nan
            else:
                medae_train_F = np.nan

            y_pred_val = reg.evaluate_model(model, np.asfortranarray(Xval))
            # dF_F = 10. ** (0.4 * A_star * (t - p)) - 1.0
            if do_dF_F:
                if args.fit_log:
                    A_pred = 10 ** y_pred_val
                    A_true = 10 ** yval
                else:
                    A_pred = y_pred_val
                    A_true = yval
                dF_F_val = 10. ** (0.4 * val_Av * (A_true - A_pred)) - 1.0
            else:
                dF_F_val = np.full_like(y_pred_val, np.nan)

            try:
                mse_val = mse(yval, y_pred_val)
            except:
                mse_val = np.nan
            try:
                r2_val = r2_score(yval, y_pred_val)
            except:
                r2_val = np.nan
            if do_dF_F:
                try:
                    medae_val_F = float(np.median(np.abs(dF_F_val)))
                except:
                    medae_val_F = np.nan
            else:
                medae_val_F = np.nan

            to_print = [model_str, model.Length, r2_train, mse_train, medae_train_F, r2_val, mse_val, medae_val_F]
            print(f'\n{to_print[1]}\n{to_print[0]}\n{to_print[2:]}')
            writer.writerow(to_print)
        
            output = np.vstack([X.T, y, y_pred_train, dF_F_train]).T
            output_val = np.vstack([Xval.T, yval, y_pred_val, dF_F_val]).T
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

