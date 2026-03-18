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

def run_operon(ini_file,):
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

    df_train = pd.read_csv(args.train_file, sep='\t')
    df_val = pd.read_csv(args.val_file, sep='\t')

    in_cols = df_train.columns.drop(['galaxy_id', 'los', 'dust_mixture', args.target_name])
    print('Input columns:', in_cols)

    X_train = df_train[in_cols].values
    y_train = df_train[args.target_name].values
    X_val = df_val[in_cols].values
    y_val = df_val[args.target_name].values

    print('Target:', args.target_name)
    print('Fitting using parameters:', in_cols)

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
    reg.fit(X_train, y_train)
    print(reg.get_model_string(reg.model_, 2))
    print(reg.stats_)

    mse = MSE()

    
    # Output directory
    run_name = f'{args.target_name}_{str(args.version_num)}'
    out_dir = pjoin(args.fit_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Make backup dir if it doesn't exist
    os.makedirs(f'{out_dir}/backup', exist_ok=True)
        
    
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
        writer.writerow(list(in_cols) + [args.target_name])

    res = [(s['tree'],  s['model']) for s in reg.pareto_front_]

    with open(outname, "w") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["Equation", "Length", "R2_train", "MSE_train", "R2_val", "MSE_val",])
        for model, model_str in res:

            y_pred_train = reg.evaluate_model(model, np.asfortranarray(X_train))

            try:
                mse_train = mse(y_train, y_pred_train)
            except Exception as e:
                print('Error calculating train mse for model:', model.Length)
                print(e)
                print(y_train.dtype, y_pred_train.dtype)
                mse_train = np.nan

            try:
                r2_train = r2_score(y_train, y_pred_train)
            except Exception as e:
                print('Error calculating train r2 for model:', model.Length)
                print(e)
                r2_train = np.nan

            y_pred_val = reg.evaluate_model(model, np.asfortranarray(X_val))

            try:
                mse_val = mse(y_val, y_pred_val)
            except:
                print('Error calculating val mse for model:', model.Length)
                mse_val = np.nan
            try:
                r2_val = r2_score(y_val, y_pred_val)
            except:
                print('Error calculating val r2 for model:', model.Length)
                r2_val = np.nan


            to_print = [model_str, model.Length, r2_train, mse_train, r2_val, mse_val]
            print(f'\n{to_print[1]}\n{to_print[0]}')
            print('MSE train, val:', to_print[3], to_print[5])
            print('R2 train, val:', to_print[2], to_print[4])
            writer.writerow(to_print)

            output = np.vstack([X_train.T, y_train, y_pred_train]).T
            output_val = np.vstack([X_val.T, y_val, y_pred_val]).T
            np.savetxt(pjoin(out_dir, f'{args.target_name}_train_{model.Length}.csv'), output)
            np.savetxt(pjoin(out_dir, f'{args.target_name}_val_{model.Length}.csv'), output_val)

    print('\nRMSE train: ', np.sqrt(mse_train))
    print('RMSE val: ', np.sqrt(mse_val))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run operon with a specified config file.")
    parser.add_argument("config_path", help="Path to the configuration file.")
    args = parser.parse_args()
    run_operon(args.config_path)


"""
TO DO

* Want to have log fit?
"""

