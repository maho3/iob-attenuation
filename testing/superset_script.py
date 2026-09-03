import os
import sys
from itertools import chain

import numpy as np
import pandas as pd
from mpi4py import MPI
from tqdm import tqdm

sys.path.insert(0, "../literature_fits")
import attenuation_curves
import fit_literature

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Bounds same as SB26 paper
bounds_2par = ([0, -100], [100, 100])
bounds_4par = ([-1e3, 0, -1e3, 0], [1e3, 100, 1e3, 0.8])
bounds_SB26 = ([0, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf])


def get_data(
    in_data_dir, dust_mixture, full_Av_name="A_5542A", lam_min=0.1, lam_max=1.0
):

    fname = os.path.join(
        in_data_dir, f"iob_codes_plus_Acurve_galprop_{dust_mixture}_all.dat"
    )

    df = pd.read_csv(fname, sep="\t", header=0)

    # Read wavelengths and attenuation curves
    use_cols = [x for x in df.columns if x.startswith("A_")]
    lam = np.array([float(x[2:-1]) for x in use_cols])
    m = (lam >= lam_min * 1e4) & (lam <= lam_max * 1e4)
    use_cols = [use_cols[i] for i in range(len(use_cols)) if m[i]]
    lam = lam[m]

    # Read attenuation curves
    A = df[use_cols].values
    A_V = df[full_Av_name].values
    A_over_Av = A / A_V[:, np.newaxis]

    # Get best-fit values of new attenuation curve
    new_pars = ["B_0", "B_1s", "B_2s", "B_3"]
    SB26_pars = df[new_pars].values

    # Store the galaxy_id and los info
    id_data = df[["galaxy_id", "los"]]

    return id_data, lam, A_over_Av, A_V, SB26_pars


def func_SB26(lam, B0, B1s, B2s, B3):

    x = lam / 5542
    c = [0.4002, 285.6, 0.2092, 9.223, 1.016]
    B1 = B1s * 1e3
    B2 = B2s * 1e3

    y = (
        B0 * (np.exp(-c[1] * (x - c[0]) ** 2) - np.exp(-c[1] * (1.0 - c[0]) ** 2))
        + (B1 + B2 * (x - c[2])) * (x - c[2]) * (np.exp(-c[3] * x) - np.exp(-c[3]))
        + np.exp(B3 * (np.tanh(c[4]) - np.tanh(c[4] * x)))
    )

    return y


def run_fits(lam, gid, los, A_over_Av_gal, SB26_pars_gal, dust_mixture):

    # Initial guesses for 4-parameter fit
    all_p0_4par = [
        [44.9, 7.56, 61.2, 0.0],  # 'Calzetti'
        [38.7, 3.83, 6.34, 0.0],  #'SMC'
        [14.4, 6.52, 2.04, 0.0519],  # 'MW'
        [4.47, 2.39, -0.988, 0.0221],  # 'LMC'
        [1.0, 1.0, 1.0, 1.0],  # Generic initial guess
    ]

    all_p0_SB26 = [
        [0.1, 0.1, 0.1, 0.1],  # Generic initial guess
        [1.0, 1.0, 1.0, 1.0],  # Generic initial guess
        SB26_pars_gal.tolist(),  # Use the best-fit parameters from before as an initial guess
    ]
    if dust_mixture in ["SMC", "stellar"]:
        all_p0_SB26.append(
            [0.0, 0.1, 0.1, 0.1]
        )  # Add a generic initial guess for SMC and stellar
        all_p0_SB26.append(
            [0.0, 1.0, 1.0, 1.0]
        )  # Add a generic initial guess for SMC and stellar

    try:
        popt_4par, success_4par = fit_literature.run_fit(
            attenuation_curves.Li_08_fit_noratio,
            lam * 1e-4,
            A_over_Av_gal,
            bounds=bounds_4par,
            all_p0=all_p0_4par,
        )
        fit_nb = attenuation_curves.Li_08_fit_noratio(lam * 1e-4, *popt_4par)
        rmse_4par = np.sqrt(np.mean((A_over_Av_gal - fit_nb) ** 2))
    except RuntimeError as e:
        print(f"Galaxy ID: {gid}, LoS: {los}, 4par fit failed with error: {e}")
        popt_4par = [None] * 4
        success_4par = False
        rmse_4par = None
    res_4par = {"params": popt_4par, "rmse": rmse_4par, "success": success_4par}

    # 2 parameter fit
    try:
        popt_2par, success_2par = fit_literature.run_fit(
            attenuation_curves.Att_Curve_2param, lam, A_over_Av_gal, bounds=bounds_2par
        )
        fit_nb = attenuation_curves.Att_Curve_2param(
            lam, B=popt_2par[0], delta=popt_2par[1]
        )
        rmse_2par = np.sqrt(np.mean((A_over_Av_gal - fit_nb) ** 2))
    except RuntimeError as e:
        print(f"Galaxy ID: {gid}, LoS: {los}, 2par fit failed with error: {e}")
        popt_2par = [None, None]
        success_2par = False
        rmse_2par = None
    res_2par = {"params": popt_2par, "rmse": rmse_2par, "success": success_2par}

    # SB26 fit
    try:
        popt_SB26, success_SB26 = fit_literature.run_fit(
            func_SB26, lam, A_over_Av_gal, bounds=bounds_SB26
        )
        fit_nb = func_SB26(
            lam, B0=popt_SB26[0], B1s=popt_SB26[1], B2s=popt_SB26[2], B3=popt_SB26[3]
        )
        rmse_SB26 = np.sqrt(np.mean((A_over_Av_gal - fit_nb) ** 2))
    except RuntimeError as e:
        print(f"Galaxy ID: {gid}, LoS: {los}, SB26 fit failed with error: {e}")
        popt_SB26 = [None, None, None, None]
        success_SB26 = False
        rmse_SB26 = None
    res_SB26 = {"params": popt_SB26, "rmse": rmse_SB26, "success": success_SB26}

    # Try to fit the SB26 function to the 4par fit results and see if we can get a good fit
    A_4par = attenuation_curves.Li_08_fit_noratio(lam * 1e-4, *popt_4par)
    try:
        popt_SB26_from_4par, success_SB26_from_4par = fit_literature.run_fit(
            func_SB26, lam, A_4par, bounds=bounds_SB26, all_p0=all_p0_SB26
        )
        fit_nb = func_SB26(
            lam,
            B0=popt_SB26_from_4par[0],
            B1s=popt_SB26_from_4par[1],
            B2s=popt_SB26_from_4par[2],
            B3=popt_SB26_from_4par[3],
        )
        rmse_SB26_from_4par = np.sqrt(np.mean((A_4par - fit_nb) ** 2))
    except RuntimeError as e:
        print(
            f"Galaxy ID: {gid}, LoS: {los}, SB26 fit from 4par failed with error: {e}"
        )
        popt_SB26_from_4par = [None, None, None, None]
        success_SB26_from_4par = False
        rmse_SB26_from_4par = None
    res_SB26_from_4par = {
        "params": popt_SB26_from_4par,
        "rmse": rmse_SB26_from_4par,
        "success": success_SB26_from_4par,
    }

    # Try to fit the SB26 function to the 2par fit results and see if we can get a good fit
    A_2par = attenuation_curves.Att_Curve_2param(
        lam, B=popt_2par[0], delta=popt_2par[1]
    )
    try:
        popt_SB26_from_2par, success_SB26_from_2par = fit_literature.run_fit(
            func_SB26, lam, A_2par, bounds=bounds_SB26, all_p0=all_p0_SB26
        )
        fit_nb = func_SB26(
            lam,
            B0=popt_SB26_from_2par[0],
            B1s=popt_SB26_from_2par[1],
            B2s=popt_SB26_from_2par[2],
            B3=popt_SB26_from_2par[3],
        )
        rmse_SB26_from_2par = np.sqrt(np.mean((A_2par - fit_nb) ** 2))
    except RuntimeError as e:
        print(
            f"Galaxy ID: {gid}, LoS: {los}, SB26 fit from 2par failed with error: {e}"
        )
        popt_SB26_from_2par = [None, None, None, None]
        success_SB26_from_2par = False
        rmse_SB26_from_2par = None
    res_SB26_from_2par = {
        "params": popt_SB26_from_2par,
        "rmse": rmse_SB26_from_2par,
        "success": success_SB26_from_2par,
    }

    return res_4par, res_2par, res_SB26, res_SB26_from_4par, res_SB26_from_2par


def main():

    dust_mixture = "MW"
    dirname = (
        "../data/deg_final_maybe"
    )

    # Load data
    if rank == 0:
        id_data, lam, A_over_Av, A_V, SB26_pars = get_data(dirname, dust_mixture)
        id_chunks = np.array_split(np.asarray(id_data), size, axis=0)
        aov_chunks = np.array_split(np.asarray(A_over_Av), size, axis=0)
        av_chunks = np.array_split(np.asarray(A_V), size, axis=0)
        sb_chunks = np.array_split(np.asarray(SB26_pars), size, axis=0)
    else:
        lam = None
        id_chunks = aov_chunks = av_chunks = sb_chunks = None

    lam = comm.bcast(lam, root=0)

    id_local = comm.scatter(id_chunks, root=0)
    A_over_Av_local = comm.scatter(aov_chunks, root=0)
    A_V_local = comm.scatter(av_chunks, root=0)
    SB26_pars_local = comm.scatter(sb_chunks, root=0)

    if rank == 0:
        print(f"Total number of galaxies: {len(id_data)}", flush=True)
        print(f"Number of galaxies per process: {len(id_local)}", flush=True)
    comm.Barrier()

    id_local = id_local[:300].astype(int)
    A_V_local = A_V_local[:300]

    for idx in tqdm(range(len(id_local)), desc="Processing galaxies", disable=(rank != 0)):

        gid = id_local[idx, 0]
        los = id_local[idx, 1]
        A_over_Av_gal = A_over_Av_local[idx]
        SB26_pars_gal = SB26_pars_local[idx]

        res_4par, res_2par, res_SB26, res_SB26_from_4par, res_SB26_from_2par = run_fits(
            lam, gid, los, A_over_Av_gal, SB26_pars_gal, dust_mixture
        )

        arr = [
            [
                *(value if value is not None else np.nan for value in r["params"]),
                r["rmse"] if r["rmse"] is not None else np.nan,
                bool(r["success"]),
            ]
            for r in [
                        res_4par, res_2par, res_SB26, res_SB26_from_4par, res_SB26_from_2par
                    ]
        ]
        arr = list(chain.from_iterable(arr))

        if idx == 0:
            res_local = np.empty((len(id_local), len(arr)), dtype=object)
        res_local[idx] = arr.copy()

    res_local = np.array(res_local)
    output_local = np.hstack((id_local, A_V_local[:, np.newaxis], res_local))

    comm.Barrier()


    # Save the results to .txt file
    out_fnam = f"fit_results_{dust_mixture}.txt"
    for r in range(size):
        if rank == r:
            print(f"Rank {rank} writing results to {out_fnam}", flush=True)

            # Delete file if it exists and rank is 0
            if rank == 0 and os.path.exists(out_fnam):
                os.remove(out_fnam)

            # Add header to the file if rank is 0
            if rank == 0:
                header = (
                    "galaxy_id los A_V "
                    "4par_c1 4par_c2 4par_c3 4par_c4 4par_rmse 4par_success "
                    "2par_B 2par_delta 2par_rmse 2par_success "
                    "SB26_B0 SB26_B1s SB26_B2s SB26_B3 SB26_rmse SB26_success "
                    "SB26_from_4par_B0 SB26_from_4par_B1s SB26_from_4par_B2s SB26_from_4par_B3 SB26_from_4par_rmse SB26_from_4par_success "
                    "SB26_from_2par_B0 SB26_from_2par_B1s SB26_from_2par_B2s SB26_from_2par_B3 SB26_from_2par_rmse SB26_from_2par_success"
                )
                with open(out_fnam, "w") as f:
                    f.write(header + "\n")

            # Append results to the file
            with open(out_fnam, "a") as f:
                np.savetxt(
                    f,
                    output_local,
                    delimiter=" ",
                    fmt=[
                        "%d",
                        "%d",
                        "%.6f",
                        *(["%.6f"] * 5 + ["%s"]),
                        *(["%.6f"] * 3 + ["%s"]),
                        *(["%.6f"] * 5 + ["%s"]) * 3,
                    ],
                )
        comm.Barrier()
    


if __name__ == "__main__":
    main()
