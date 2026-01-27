import pandas as pd
import re
from os.path import join as pjoin
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from utils import OperonArgs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def load_data(ini_file, name):
    """
    Load the data for optimising b and D parameters.

    Args:
        :ini_file (str): Path to the ini file with parameters.
        :name (str): 'train' or 'val' to specify which dataset to load

    Returns:
        :x (torch.Tensor): Wavelengths normalized by lambda_V, shape (nlam,).
        :ytrue (torch.Tensor): True attenuation curves normalized by A_V, shape (ngal, nlam).
        :D (np.ndarray): Initial D parameters for each galaxy, shape (ngal, 4).
        :b (np.ndarray): Initial b parameters, shape (6,).
    """


    args = OperonArgs(ini_file)

    fname = args.selection.train_file if name == 'train' else args.selection.val_file
    run_name = f'{args.in_param}_{str(args.version_num)}'
    fname_outer = f'{args.fit_dir}/{run_name}/{run_name}_outer_{name}.csv'

    df_orig = pd.read_csv(fname)

    # Load data with C0
    run_name = f'{args.in_param}_{str(args.version_num)}'
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

    # Get initial b values
    b = np.zeros(6)
    b[0] = 0.944124
    b[1] = a[1]
    b[2] = a[2] * a[6]
    b[3] = a[6] * a[7]
    b[4] = - a[4] / a[5]
    b[5] = a[5] ** 2
    print(b)

    # Get the true data
    attenuation_cols = [col for col in df_orig.columns if re.match(r'A_\d+A', col)]
    lam_arr = torch.tensor([int(re.search(r'_(\d+)A', col).group(1)) / 1e4 for col in attenuation_cols])
    sort_idx = torch.argsort(lam_arr)
    lam_arr = lam_arr[sort_idx]
    mask = (lam_arr < args.lam_max) & (lam_arr > args.lam_min)
    lam_arr = lam_arr[mask]
    attenuation_cols = [attenuation_cols[i] for i in sort_idx if mask[i]]
    A_v_col = f'A_{int(args.lambda_V*1e4)}A'
    x = lam_arr / args.lambda_V
    nlam = len(x)
    curves_flat = df_orig[attenuation_cols].values
    ytrue = curves_flat / df_orig[A_v_col].values[:,None]

    return x, ytrue, D, b


class GalaxyDataset(Dataset):
    def __init__(self, x, ytrue):
        galaxies = []
        n_gal = ytrue.shape[0]
        for i in range(n_gal):
            galaxies.append({'x': x.numpy(), 'y': ytrue[i, :].astype(np.float32)})
        # galaxies: list of dicts with 'x' (np array) and 'y' (np array)
        self.galaxies = galaxies
        print(f'Loaded dataset with {len(self.galaxies)} galaxies.')

    def __len__(self):
        return len(self.galaxies)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.galaxies[idx]['x'], dtype=torch.float32)
        y = torch.tensor(self.galaxies[idx]['y'], dtype=torch.float32)
        return x, y, idx


# Model function (vectorised)
def model_torch(x, b, D):
    """
    The model to be optimise

    Args:
        x: (N,) input array of lamda/lambda_V
        b: (6,) array of b parameters
        D: (4,) array of D parameters for this galaxy

    Returns:
        y: (N,) output array of predicted A(lambda)/A_V
    """
    y = (
        D[0] * (torch.exp(-b[5] * (x - b[4])**2) - torch.exp(-b[5] * (1. - b[4])**2))
        + (b[2] + b[3] * x) * (D[1] + D[2] * x) * (torch.exp(b[1] * x) - torch.exp(b[1]))
        + torch.exp(D[3] * (torch.tanh(b[0]) - torch.tanh(b[0] * x)))
    )
    return y


def main():

    lr = 1e-2
    factor = 0.5
    patience = 10
    batch_size = 128
    nepoch = 1000
    ini_file = 'conf/iob_44.ini'
    num_workers = 0

    x, ytrue, initial_Ds, initial_b = load_data(ini_file, 'train')

    dataset = GalaxyDataset(x, ytrue)
    n_gal = len(dataset)
    
    # learnable global parameters
    b = torch.tensor(initial_b, dtype=torch.float32, requires_grad=True, device=device)

    # per-galaxy D parameters (stacked)
    Ds = torch.tensor(initial_Ds, dtype=torch.float32, requires_grad=True, device=device)  # shape (n_gal, 4)

    optimizer = optim.Adam([b, Ds], lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda batch: batch)

    # Compute initial using initial D and b
    print('Computing initial MSE with initial D and b...')
    with torch.no_grad():
        ypred_init = []
        for i in range(n_gal):
            x_gal = torch.tensor(dataset.galaxies[i]['x'], dtype=torch.float32, device=device)
            Dg = Ds[i]
            y_hat = model_torch(x_gal, b, Dg)
            ypred_init.append(y_hat.cpu().numpy())
        ypred_init = np.array(ypred_init)
        initial_rmse = np.sqrt(np.mean((ypred_init - ytrue)**2))
        print(f'Initial RMSE with initial D and b: {initial_rmse:.6f}')

    all_loss = np.zeros(nepoch)
    all_rmse = np.zeros(nepoch)
    all_b = np.zeros((nepoch, len(initial_b)))

    for epoch in range(nepoch):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = 0.0
            for x, y, idx in batch:
                # print(x.shape, y.shape, idx)
                x = x.to(device)
                y = y.to(device)
                Dg = Ds[idx]
                y_hat = model_torch(x, b, Dg)
                # MSE per-galaxy
                loss = loss + torch.mean((y - y_hat)**2)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        all_loss[epoch] = epoch_loss
        all_b[epoch,:] = b.detach().cpu().numpy()
        print(f"Epoch {epoch}, loss: {epoch_loss:.4f}")
        print(f"  b parameters: {b.detach().cpu().numpy()}")
        scheduler.step(epoch_loss) 

        # Get overall MSE on training set
        with torch.no_grad():
            ypred = []
            for i in range(n_gal):
                x_gal = torch.tensor(dataset.galaxies[i]['x'], dtype=torch.float32, device=device)
                Dg = Ds[i]
                y_hat = model_torch(x_gal, b, Dg)
                ypred.append(y_hat.cpu().numpy())
            ypred = np.array(ypred)
            rmse = np.sqrt(np.mean((ypred - ytrue)**2))
            print(f'  Training RMSE: {rmse:.6f}')
            all_rmse[epoch] = rmse

    # Save final parameters
    final_b = b.detach().cpu().numpy()
    final_Ds = Ds.detach().cpu().numpy()
    np.savetxt('opt_fun_final_b_params.txt', final_b)
    np.savetxt('opt_fun_final_D_params.txt', final_Ds)
    np.savetxt('opt_fun_training_loss.txt', all_loss)
    np.savetxt('opt_fun_training_rmse.txt', all_rmse)
    np.savetxt('opt_fun_b_params_per_epoch.txt', all_b)

    # Plot the training loss and RMSE
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    ax[0].plot(np.arange(nepoch), all_loss)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Training Loss (MSE)')
    ax[1].plot(np.arange(nepoch), all_rmse)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Training RMSE')
    plt.tight_layout()
    plt.savefig('opt_fun_training_loss_rmse.png', bbox_inches='tight')
    plt.clf()

    # Plot the evolution of each of the b parameters
    ncol = 3
    nrow = int(np.ceil(len(final_b) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4 * nrow))
    ax = ax.flatten()
    for i in range(len(final_b)):
        ax[i].plot(np.arange(nepoch), all_b[:, i])
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel(f'b[{i}] value')
    for i in range(len(final_b), nrow * ncol):
        fig.delaxes(ax[i])
    plt.tight_layout()
    plt.savefig('opt_fun_b_params_evolution.png', bbox_inches='tight')
    plt.clf()


    return


if __name__ == '__main__':
    main()

