import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import scipy.optimize
import matplotlib.pyplot as plt
import sys

from sr_fun import compute_initial_B, compute_Av

sys.path.insert(0, '../literature_fits')
from fit_literature import run_fit, find_nearest, load_data

# Import MPI here
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def optimise_gal(gal_id, los, lam_arr, Alam_arr_cut, IOB, do_plot=False):
        
    if do_plot:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ### Normalising ad A_lambda to Av
    v_index = find_nearest(lam_arr,0.551)
    Av = Alam_arr_cut[v_index]
    lam_v = lam_arr[v_index]
    Alam_Av_arr_cut = Alam_arr_cut/Av
    lam_cut = lam_arr[lam_arr < 1.]
    x = lam_cut / lam_v  # Normalise by lambda_V

    # Estimate the initial B values from IOB parameters
    all_p0 = compute_initial_B(IOB)
    all_p0 = [[float(p) for p in all_p0]]

    m = lam_cut>0.12
    if len(m) != len(Alam_Av_arr_cut):
        print(f"Warning: Mismatch in length of lam_cut and Alam_Av_arr_cut for galaxy {gal_id}, LoS {los}. {len(m)} vs {len(Alam_Av_arr_cut)}", flush=True)

    bounds = ([-np.inf] * len(all_p0[0]), [np.inf] * len(all_p0[0]))

    try:
        fit_nb = compute_Av(x[m], *all_p0[0])
        popt, success = run_fit(compute_Av, x[m], Alam_Av_arr_cut[m], 
                                          bounds=bounds,
                                          all_p0=all_p0)
        fit_nb = compute_Av(x[m], *popt)
        if do_plot:
            ax.plot(x[m],fit_nb,ls='-', lw=4., alpha=0.2, label='SR fit')
        rmse = np.sqrt(np.mean((Alam_Av_arr_cut[m] - fit_nb)**2))

    except RuntimeError as e:
        print(f"Galaxy ID: {gal_id}, LoS: {los}, Fit failed with error: {e}")
        popt = [None] * 4
        success = False
        rmse = None

    if do_plot:
        ax.plot(x, Alam_Av_arr_cut,'.', lw=2., alpha=0.5,)
        ax.set_xlabel(r"$\lambda / \lambda_V$")
        ax.set_ylabel(r'$A(\lambda)/A_V$')
        ax.set_title(f'Galaxy {int(gal_id)}, LoS {los}')
        ax.legend()
        fig.tight_layout()
        # plt.show()
        plt.savefig(f"galaxy_{int(gal_id)}_los_{los}.png")

    return popt, rmse, success


def main():

    # Load the data
    fname = '../data/gal_los_iobcomp_attcurve_galprop.dat'
    ids, att_groups, attenuation_cols, lam_arr = load_data(fname)

    if rank == 0:
        print('Number of galaxies:', len(ids), flush=True)
        print('Number of galaxies per rank:', len(ids) // size, flush=True)

    # Split the galaxy IDs for parallel processing
    n_galaxies = len(ids)
    chunk_size = n_galaxies // size + (n_galaxies % size > 0)
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, n_galaxies)

    local_results = []

    # Process each galaxy assigned to this rank
    for gal_id in tqdm(ids[start_idx:end_idx], desc="Processing galaxies", disable=(rank != 0)):
        group = att_groups.get_group(gal_id)
        for los in group['los'].values:
            row = group[group['los'] == los].iloc[0] # occasionally multiple rows for same galaxy and los so pick first
            Alam_arr_cut = row[attenuation_cols].values.flatten()
            iob_cols = sorted([col for col in group.columns if col.startswith('IOB')])
            IOB = row[iob_cols].values.flatten().astype(float)
            popt, rmse, success = optimise_gal(gal_id, los, lam_arr, Alam_arr_cut, IOB, do_plot=False)
            local_results.append((gal_id, los, popt, rmse, success))

    # Combine results from all ranks and save to a file
    all_results = comm.gather(local_results, root=0)
    if rank == 0:
        flat_results = [item for sublist in all_results for item in sublist]
        df = pd.DataFrame(flat_results, columns=['gal_id', 'los', 'popt', 'rmse', 'success'])
        df.to_csv("../data/galaxy_sr_optimization_results.csv", index=False)

    return

if __name__ == "__main__":
    main()
