import numpy as np
import pandas as pd
import multiprocessing
from pyoperon.sklearn import SymbolicRegressor
from pyoperon import MSE
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from processing_fun import convert_operon_fun

target_par = 'IOB1'
seed = 1
ntrain = 5000
nval = 5000
max_length = 30
time_limit = 60 # seconds
ilen = 10

# Load the data
fname = '../data/gal_los_iobcomp_attcurve_galprop.dat'
att_df = pd.read_csv(fname, sep='\t')

# Get columns with physical parameters
cols = att_df.columns.tolist()
cols = [c for c in cols if not c.startswith('A_') and not c.startswith('logA_') and not c.startswith('galaxy_id')]
cols = [c for c in cols if not c.startswith('los') and not c.startswith('IOB') and not c.startswith('PCA')]
X = att_df[cols].values
y = att_df[target_par].values

# Remove the units from the column names
cols = [c[:c.index('[')] if '[' in c else c for c in cols]
cols = [c.strip() for c in cols]

# Pick subset for training and validation
np.random.seed(seed)
use_indices = np.random.permutation(len(X))[:ntrain + nval]
X_train = X[use_indices[:ntrain]]
X_val = X[use_indices[ntrain:]]
y_train = y[use_indices[:ntrain]]
y_val = y[use_indices[ntrain:]]

# Normalise y by the mean and std
y_mean = np.mean(y_train)
y_std = np.std(y_train)
y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std

print('Target parameter:', target_par)
print('Fitting using properties:', cols)
print(f'Training shape: {X_train.shape}, Validation shape: {X_val.shape}')

# Setup operon regressor
reg = SymbolicRegressor(
            allowed_symbols='add,sub,mul,div,sqrt,square,pow,log,constant,variable',
            offspring_generator='basic',
            optimizer_iterations=1000,
            max_length=max_length,
            initialization_method='btc',
            n_threads=multiprocessing.cpu_count(),
            objectives = ['rmse', 'length'],
            epsilon = 1e-6,
            random_state=None,
            reinserter='keep-best',
            max_evaluations=int(1e12),
            symbolic_mode=False,
            time_limit=time_limit,
            generations=int(1e12),
            )
print('Fitting')
reg.fit(X_train, y_train)

# Analyse results
res = [(s['tree'],  s['model']) for s in reg.pareto_front_]
mse = MSE()
all_length = [model.Length for model, _ in res]
# Find nearest model length to ilen
nearest_length = min(all_length, key=lambda x: abs(x - ilen))
print(f'\nNearest model length to {ilen}: {nearest_length}')
all_mse_train = [None for _ in res]
all_mse_val = [None for _ in res]
for i, (model, model_str) in enumerate(res):
    y_pred_train = reg.evaluate_model(model, X_train)
    all_mse_train[i] = mse(y_train, y_pred_train)
    y_pred_val = reg.evaluate_model(model, X_val)
    all_mse_val[i] = mse(y_val, y_pred_val)
    # print(model.Length, model_str)
    if model.Length == nearest_length:
        new_eq, values = convert_operon_fun(model_str, cols, do_replace_floats=True)
        y_pred_train_saved = y_pred_train.copy()
        y_pred_val_saved = y_pred_val.copy()
        print(f'Nearest model: {new_eq}')
        print('Values:', values)
        print(f'Train MSE: {all_mse_train[i]:.4f}, Validation MSE: {all_mse_val[i]:.4f}')

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].set_title(f'Symbolic Regression Results for {target_par}')
axs[0].plot(all_length, all_mse_train, 'o-', label='Train MSE')
axs[0].plot(all_length, all_mse_val, 'o-', label='Validation MSE')
axs[0].set_xlabel('Model Length')
axs[0].set_ylabel('MSE')
axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[0].set_ylim(None, 1.0)
axs[0].axvline(nearest_length, color='r', linestyle='--', label=f'Chosen Length: {nearest_length}')
axs[0].legend()
axs[1].plot(y_train, y_pred_train_saved, 'o', label='Train Predictions', ms=0.5)
axs[1].plot(y_val, y_pred_val_saved, 'o', label='Validation Predictions', ms=0.5)
axs[1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', label='y=x')
axs[1].set_xlabel('True Values')
axs[1].set_ylabel('Predicted Values')
axs[1].set_title(f'Predictions for {target_par}')
axs[1].set_xlim(y_train.min(), y_train.max())
axs[1].set_ylim(y_train.min(), y_train.max())
axs[1].legend()
fig.tight_layout()
plt.show()
