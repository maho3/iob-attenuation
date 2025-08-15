import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import scipy.optimize
import matplotlib.pyplot as plt

from attenuation_curves import Att_Curve_2param, Li_08_fit_noratio

# Import MPI here
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def run_fit(fun, x, y, bounds=None, all_p0=None):
    """
    Run a curve fit with optional bounds and initial parameters.
    We can try multiple initial guesses if needed.
    
    Parameters
    ----------
    fun : callable
        The function to fit.
    x : array_like
        The independent variable.
    y : array_like
        The dependent variable.
    bounds : tuple, optional
        Bounds for the parameters (min, max).
    all_p0 : array_like, optional
        Initial guess for the parameters.
    
    Returns
    -------
    popt : array
        Optimal values for the parameters.
    success : bool
        Whether the fit was successful.
    """

    def to_opt(p):
        try:
            with np.errstate(over='raise', invalid='raise'):
                ypred = fun(x, *p)
                residuals = y - ypred
                mse = np.sum(residuals**2)
            return mse
        except (RuntimeWarning, FloatingPointError):
            return 1e30 # Return a large error if the fit fails

    b = [(low, high) for low, high in zip(bounds[0], bounds[1])]

    if all_p0 is None:
        # best_popt, best_pcov = curve_fit(fun, x, y, bounds=bounds)
        res = scipy.optimize.minimize(to_opt, bounds=b, x0=np.ones(len(b)))
        best_popt = res.x
        best_success = res.success
    else:
        best_error = np.inf
        best_popt = None
        for p0 in all_p0:
            try:
                # popt, pcov = curve_fit(fun, x, y, p0=p0, bounds=bounds)
                res = scipy.optimize.minimize(to_opt, p0, bounds=b)
                popt = res.x
                error = to_opt(popt)
                if error < best_error:
                    best_error = error
                    best_popt = popt
                    best_success = res.success
            except RuntimeError:
                continue
        if best_popt is None:
            raise RuntimeError("All fits failed. No valid parameters found.")

    return best_popt, best_success


def load_data(fname):

    ### Reading TNG Outputs
    # Load attenuation curves and galaxy info
    att_df = pd.read_csv(fname, sep='\t')

    # Identify attenuation columns (formatted like A_1000A)
    attenuation_cols = [col for col in att_df.columns if re.match(r'A_\d+A', col)]

    # Extract the wavelength in angstroms and convert to microns
    lam_arr = np.array([int(re.search(r'_(\d+)A', col).group(1)) / 1e4 for col in attenuation_cols])

    # Optional: sort by increasing wavelength (just in case)
    sort_idx = np.argsort(lam_arr)
    lam_arr = lam_arr[sort_idx]
    attenuation_cols = [attenuation_cols[i] for i in sort_idx]

    # Extract galaxy-level properties (duplicated across LoS)
    ids = att_df['galaxy_id'].unique()

    # Group attenuation data by galaxy
    att_groups = att_df.groupby('galaxy_id')

    return ids, att_groups, attenuation_cols, lam_arr


def optimise_gal(gal_id, los, lam_arr, Alam_arr_cut, do_plot=False, lambda_v=0.5542):
        
    if do_plot:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ### Normalising ad A_lambda to Av
    v_index = find_nearest(lam_arr, lambda_v)
    Av = Alam_arr_cut[v_index]
    Alam_Av_arr_cut = Alam_arr_cut/Av
    lam_cut = lam_arr[lam_arr < 1.]

    # Initial guesses for 4-parameter fit
    all_p0_4par = [
        [44.9, 7.56, 61.2, 0.],  # 'Calzetti'
        [38.7, 3.83, 6.34, 0.], #'SMC'
        [14.4, 6.52, 2.04, 0.0519], # 'MW'
        [4.47, 2.39, -0.988, 0.0221], # 'LMC'
        [1.0, 1.0, 1.0, 1.0],  # Generic initial guess
    ]

    m = lam_cut>0.12
    if len(m) != len(Alam_Av_arr_cut):
        print(f"Warning: Mismatch in length of lam_cut and Alam_Av_arr_cut for galaxy {gal_id}, LoS {los}. {len(m)} vs {len(Alam_Av_arr_cut)}", flush=True)

    # 4 parameter fit
    try:
        popt_4par, success_4par = run_fit(Li_08_fit_noratio, lam_cut[m], Alam_Av_arr_cut[m], 
                            bounds=([-np.inf,-np.inf,-np.inf,0],[np.inf,np.inf,np.inf,1.]),
                            all_p0=all_p0_4par)
        fit_nb = Li_08_fit_noratio(lam_cut[m],*popt_4par)
        if do_plot:
            ax.plot(lam_cut[m]*1e4,fit_nb,ls='-', lw=4., alpha=0.2, label='4-parameter fit')
        rmse_4par = np.sqrt(np.mean((Alam_Av_arr_cut[m] - fit_nb)**2))
    except RuntimeError as e:
        print(f"Galaxy ID: {gal_id}, LoS: {los}, Fit failed with error: {e}")
        popt_4par = [None] * 4
        success_4par = False
        rmse_4par = None

    # 2 parameter fit
    try:
        popt_2par, success_2par = run_fit(Att_Curve_2param, 1e4*lam_cut[m], Alam_Av_arr_cut[m], 
                            bounds = ([-np.inf,-np.inf],[np.inf,np.inf]))
        fit_nb = Att_Curve_2param(1e4*lam_cut[m],B=popt_2par[0],delta=popt_2par[1])
        if do_plot:
            ax.plot(lam_cut[m]*1e4,fit_nb,ls='-',lw=4., alpha=0.2, label='2-parameter fit')
        rmse_2par = np.sqrt(np.mean((Alam_Av_arr_cut[m] - fit_nb)**2))
    except RuntimeError as e:
        print(f"Galaxy ID: {gal_id}, LoS: {los}, Fit failed with error: {e}")
        popt_2par = [None, None]   
        success_2par = False 
        rmse_2par = None
    
    if do_plot:
        ax.plot(lam_cut*1e4, Alam_Av_arr_cut,'.', lw=2., alpha=0.5,)
        ax.set_xlabel('Wavelength (Angstrom)')
        ax.set_ylabel(r'$A(\lambda)/A_V$')
        ax.set_title(f'Galaxy {int(gal_id)}, LoS {los}')
        ax.legend()
        fig.tight_layout()
        plt.show()

    return popt_2par, popt_4par, rmse_2par, rmse_4par, success_2par, success_4par


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
            popt_2par, popt_4par, rmse_2par, rmse_4par, success_2par, success_4par = optimise_gal(gal_id, los, lam_arr, Alam_arr_cut, do_plot=False)
            local_results.append((gal_id, los, popt_2par, popt_4par, rmse_2par, rmse_4par, success_2par, success_4par))

    # Combine results from all ranks and save to a file
    all_results = comm.gather(local_results, root=0)
    if rank == 0:
        flat_results = [item for sublist in all_results for item in sublist]
        df = pd.DataFrame(flat_results, columns=['gal_id', 'los', 'popt_2par', 'popt_4par', 'rmse_2par', 'rmse_4par', 'success_2par', 'success_4par'])
        df.to_csv("../data/galaxy_optimization_results.csv", index=False)

    return

if __name__ == "__main__":
    main()
