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
    
    with open(fname_train, 'r') as f:
        names = f.readline().strip().split()[2:]
    data = np.loadtxt(fname_train, skiprows=1)
    X = data[:,2:-1]
    y = data[:,-1]

    with open(fname_val, 'r') as f:
        val_names = f.readline().strip().split()[2:]
    data = np.loadtxt(fname_val, skiprows=1)
    Xval = data[:,2:-1]
    yval = data[:,-1]

    par_mask = np.ones(X.shape[1], dtype=bool)
    print('Target:', names[-1])
    print('Fitting using parameters:', [names[p] for p in range(len(names)-1) if par_mask[p]])

    assert names == val_names, 'Training and validation data have different names'

    reg = SymbolicRegressor(
            allowed_symbols=args.allowed_symbols,
            offspring_generator='basic',
            optimizer_iterations=1000,
            max_length=args.max_length,
            initialization_method='btc',
            n_threads=multiprocessing.cpu_count(),
            objectives = args.objectives,
            epsilon = args.epsilon,
            random_state=None,
            reinserter='keep-best',
            max_evaluations=args.max_evaluations,
            symbolic_mode=False,
            time_limit=args.time_limit,
            generations=args.generations,
            )

    print('Fitting')
    reg.fit(X[:,par_mask], y)
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
        writer.writerow(names)

    res = [(s['tree'],  s['model']) for s in reg.pareto_front_]

    with open(outname, "w") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["Equation", "Length", "R2_train", "MSE_train", "R2_val", "MSE_val"])
        for model, model_str in res:

            y_pred_train = reg.evaluate_model(model, X[:,par_mask])
            try:
                mse_train = mse(y, y_pred_train)
            except:
                mse_train = np.nan
            try:
                r2_train = r2_score(y, y_pred_train)
            except:
                r2_train = np.nan

            y_pred_val = reg.evaluate_model(model, Xval[:,par_mask])
            try:
                mse_val = mse(yval, y_pred_val)
            except:
                mse_val = np.nan
            try:
                r2_val = r2_score(yval, y_pred_val)
            except:
                r2_val = np.nan
        
            to_print = [model_str, model.Length, r2_train, mse_train, r2_val, mse_val]
            print(f'\n{to_print[1]}\n{to_print[0]}\n{to_print[2:]}')
            writer.writerow(to_print)
        
            output = np.vstack([X.T, y, y_pred_train]).T
            output_val = np.vstack([Xval.T, yval, y_pred_val]).T
            np.savetxt(f'{outname_pred_train}_{model.Length}.csv', output)
            np.savetxt(f'{outname_pred_val}_{model.Length}.csv', output_val)

    print('\nRMSE train: ', np.sqrt(mse_train))
    print('RMSE val: ', np.sqrt(mse_val))
    
    return


if __name__ == "__main__":
    # run_operon('conf/iob_0.ini')
    parser = argparse.ArgumentParser(description="Run operon with a specified config file.")
    parser.add_argument("config_path", help="Path to the configuration file.")
    args = parser.parse_args()
    run_operon(args.config_path)

