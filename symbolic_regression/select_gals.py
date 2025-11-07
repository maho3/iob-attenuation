import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import corner

def load_df(infile):
    """
    Load the data file into a pandas DataFrame, handling duplicate column names
    and stripping units from column names.

    Args:
        :infile (str): Path to the input data file.

    Returns:
        :df (pd.DataFrame): DataFrame containing the data with proper column names.
    """

    # Read the first line to extract column names (before units)
    with open(infile, 'r') as f:
        header_line = f.readline().strip()
    
    # Split by whitespace and extract base column names (removing units in brackets)
    raw_cols = header_line.split()
    col_names = []
    col_names = [col for col in raw_cols if '[' not in col and ']' not in col]

    # Get duplicate col names
    duplicates = {}
    for i in range(len(col_names)):
        col = col_names[i]
        if col in duplicates:
            duplicates[col] += 1
            col_names[i] = f"{col}_{duplicates[col]}"
        else:
            duplicates[col] = 1
    dup_cols = {col: count for col, count in duplicates.items() if count > 1}
    print("Duplicate columns found:", dup_cols)

    d = np.loadtxt(infile, skiprows=1)[0]
    assert len(d) == len(col_names), f"Column name count {len(col_names)} does not match data columns {len(d)}"

    df = pd.read_csv(infile, sep=r"\s+", skiprows=1, names=col_names)

    return df


def check_inconsistency(df, cols_to_check):
    """
    Check for inconsistencies in the specified columns of the DataFrame.

    Args:
        :df (pd.DataFrame): The DataFrame to check.
        :cols_to_check (list): List of column names to check for inconsistencies.

    Returns:
        :df (pd.DataFrame): DataFrame with inconsistent galaxies removed.
    """

    print(f"Checking columns: {cols_to_check}")

    def has_inconsistencies(group):
        for c in cols_to_check:
            vals = np.unique(group[c].dropna())
            if len(vals) > 1:
                return True
        return False
    
    bad_ids = []
    for gid, group in tqdm(df.groupby('galaxy_id')):
        if has_inconsistencies(group):
            bad_ids.append(gid)

    if bad_ids:
        # pd.DataFrame({galid_col: bad_ids}).to_csv(inconsistency_report, sep="\t", index=False)
        print(f"Found {len(bad_ids)} inconsistent galaxies — removed from dataset.")
    else:
        print(f'All galaxies consistent across columns tested.')

    # Remove bad galaxies from dataframe
    df.drop(df[df['galaxy_id'].isin(bad_ids)].index, inplace=True)

    return df


def select_adaptive_bins(df, min_per_bin_mult, nbins_max, n_per_bin):
    """
    Select galaxies using adaptive bins in logMstar, ensuring uniform coverage in Sigma_SFR.

    Args:
        :df (pd.DataFrame): Input DataFrame with galaxy data.
        :min_per_bin_mult (int): Multiplier for minimum galaxies per bin.
        :nbins_max (int): Maximum number of bins.
        :n_per_bin (int): Number of galaxies to select per bin.

    Returns:
        :selected (pd.DataFrame): DataFrame of selected galaxies.
        :G (pd.DataFrame): Aggregated per-galaxy DataFrame.
        :edges (np.ndarray): Edges of the adaptive bins.
    """

    av_col = 'A_5542A'
    logm_col = 'logMstar'
    sig_col = 'Sigma_SFR'
    galid_col = 'galaxy_id'

    use_cols = list(df.columns[list(df.columns).index('logMstar'):])
    print("Applying adaptive binning")

    agg_dict = {c: "first" for c in use_cols}
    agg_dict.update({
        av_col: "median",
        logm_col: "first",
        sig_col: "first",
    })

    G = (
        df.groupby(galid_col, sort=False)
        .agg(agg_dict)
        .rename(columns={av_col: "Av_med"})
        .reset_index()
        .dropna(subset=[logm_col, sig_col, "Av_med"])
    )

    n_gal = len(G)
    if n_gal == 0:
        raise RuntimeError("No galaxies left after aggregation.")

    # --- Choose effective number of bins so each has enough galaxies to sample from ---
    min_per_bin = max(n_per_bin * min_per_bin_mult, n_per_bin)
    n_bins_eff = max(1, min(nbins_max, n_gal // max(1, min_per_bin)))

    # If sample is small, force at least 1 bin
    if n_bins_eff < 1:
        n_bins_eff = 1

    # --- Build quantile edges (equal-count bins) ---
    q = np.linspace(0.0, 1.0, n_bins_eff + 1)
    edges = np.quantile(G[logm_col].to_numpy(), q)
    # Ensure strictly increasing edges; collapse duplicates if distribution is very discrete
    edges = np.unique(edges)
    if len(edges) < 2:
        # fall back to a single wide bin
        edges = np.array([G[logm_col].min(), G[logm_col].max()])

    # Recompute labels with final edges
    n_bins_eff = len(edges) - 1
    bin_labels = np.arange(n_bins_eff)

    # Assign bins on the aggregated, per-galaxy table
    G["mstar_bin"] = pd.cut(G[logm_col], bins=edges, labels=bin_labels, include_lowest=True)

    # --- Selection: uniformly in Sigma_SFR within each adaptive bin ---
    def pick_uniform_by_sigma_sfr_agg(df_bin, n_pick=n_per_bin):
        """df_bin is already one-row-per-galaxy with columns: galaxy_id, logM, Sigma_SFR, Av_med, (>=61 cols)."""
        if df_bin.empty:
            return df_bin

        g = df_bin.sort_values(sig_col).reset_index(drop=True)

        if len(g) <= n_pick:
            return g.copy()

        # uniform-in-range pick using quantiles of Sigma_SFR
        quantiles = (np.arange(n_pick) + 0.5) / n_pick
        targets = g[sig_col].quantile(quantiles).to_numpy()

        chosen_idx, remaining = [], np.arange(len(g))
        for tv in targets:
            diffs = np.abs(g.loc[remaining, sig_col].to_numpy() - tv)
            j = remaining[np.argmin(diffs)]
            chosen_idx.append(j)
            remaining = remaining[remaining != j]
            if len(remaining) == 0:
                break
        return g.loc[sorted(set(chosen_idx))].copy()

    selected_rows = []
    for b in tqdm(bin_labels):
        bin_mask = (G["mstar_bin"] == b)
        G_bin = G[bin_mask]
        if G_bin.empty:
            continue
        chosen = pick_uniform_by_sigma_sfr_agg(G_bin, n_per_bin)
        if not chosen.empty:
            b = int(b)
            chosen["mstar_bin"] = b
            chosen["bin_left"]  = edges[b]
            chosen["bin_right"] = edges[b+1]
            selected_rows.append(chosen)

    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()

    return selected, G, edges


def select_extremes(selected, G, edges, n_high, n_low, sfr_high_thresh, sfr_low_thresh, q_high, q_low, rng_seed=42):
    """
    Select extreme galaxies based on high/low Av and Sigma_SFR thresholds.

    Args:
        :selected (pd.DataFrame): Currently selected galaxies.
        :G (pd.DataFrame): Aggregated per-galaxy DataFrame.
        :edges (np.ndarray): Edges of the adaptive bins.
        :n_high (int): Number of high-Av, high-Sigma_SFR galaxies
        :n_low (int): Number of low-Av, low-Sigma_SFR galaxies
        :sfr_high_thresh (float): Sigma_SFR threshold for high selection.
        :sfr_low_thresh (float): Sigma_SFR threshold for low selection.
        :q_high (float): Quantile threshold for high Av selection.
        :q_low (float): Quantile threshold for low Av selection.
        :rng_seed (int): Random seed for reproducibility.

    Returns:
        :selected (pd.DataFrame): Updated selected galaxies including extremes.
        :extremes (pd.DataFrame): DataFrame of newly added extreme galaxies.    
    """

    rng = np.random.default_rng(rng_seed)

    galid_col = 'galaxy_id'
    av_col = 'A_5542A'
    logm_col = 'logMstar'
    sig_col = 'Sigma_SFR'

    # Work on the one-row-per-galaxy table G; ensure consistent id
    if "galaxy_id" not in G.columns:
        G = G.rename(columns={galid_col: "galaxy_id"})

    # Selected galaxy IDs (per-galaxy)
    sel_ids = set(selected["galaxy_id"].unique() if "galaxy_id" in selected.columns
                else selected[galid_col].unique())

    # Candidate pool: galaxies NOT already selected
    pool = G[~G["galaxy_id"].isin(sel_ids)].copy()

    # Av thresholds on the whole per-galaxy pool
    av80 = np.quantile(G["Av_med"].to_numpy(), q_high)
    av20 = np.quantile(G["Av_med"].to_numpy(), q_low)

    # Define pools
    high_high_pool = pool[(pool["Av_med"] >= av80) & (pool[sig_col] >= sfr_high_thresh)]
    low_low_pool   = pool[(pool["Av_med"] <= av20) & (pool[sig_col] <= sfr_low_thresh)]

    # Random picks (without replacement)
    high_pick = high_high_pool.sample(n=min(n_high, len(high_high_pool)),
                                    replace=False, random_state=rng_seed) if len(high_high_pool) else high_high_pool
    low_pick  = low_low_pool.sample(n=min(n_low,  len(low_low_pool)),
                                    replace=False, random_state=rng_seed+1) if len(low_low_pool)  else low_low_pool

    # Combine extremes and tag
    extremes = pd.concat([high_pick, low_pick], ignore_index=True)
    if len(extremes):
        extremes = extremes.drop_duplicates(subset=["galaxy_id"]).copy()
        extremes["flag_extreme"] = np.where(extremes["Av_med"] >= av80, "highAv_highSigma", "lowAv_lowSigma")

        # attach bin metadata for consistency
        extremes["mstar_bin"] = pd.cut(extremes[logm_col], bins=edges,
                                    labels=np.arange(len(edges) - 1), include_lowest=True)
        bi = extremes["mstar_bin"].astype(int)
        extremes["bin_left"]  = edges[bi]
        extremes["bin_right"] = edges[bi + 1]

        # merge into per-galaxy selection (no duplicates)
        before_n = selected["galaxy_id"].nunique() if "galaxy_id" in selected.columns else selected[galid_col].nunique()
        # normalize id col name in selected
        if "galaxy_id" not in selected.columns:
            selected = selected.rename(columns={galid_col: "galaxy_id"})
        selected = pd.concat([selected, extremes], ignore_index=True)
        selected = selected.drop_duplicates(subset=["galaxy_id"], keep="first").reset_index(drop=True)
        after_n = selected["galaxy_id"].nunique()
        print(f"Added {after_n - before_n} extreme galaxies (target 30).")
    else:
        print("No candidates found for extremes under current thresholds.")

    return selected, extremes


def get_attenuation_curves(df, selected, seed=12345):
    """
    Extract attenuation curves for selected galaxies. One line-of-sight (LOS) per galaxy is chosen randomly.

    Args:
        :df (pd.DataFrame): Original DataFrame with all LOS data.
        :selected (pd.DataFrame): DataFrame of selected galaxies.
        :seed (int): Random seed for reproducibility.

    Returns:
        :df_final (pd.DataFrame): DataFrame with one LOS per selected galaxy.
    """

    selected_gals = selected['galaxy_id']
    
    # For each gal in selected, extract one los randomly and save its attenuation curve
    df_final = pd.DataFrame()
    rng = np.random.default_rng(seed)
    for gid in tqdm(selected_gals):
        gal_df = df[df['galaxy_id'] == gid]
        if gal_df.empty:
            continue
        los_indices = gal_df.index.to_numpy()
        chosen_idx = rng.choice(los_indices)

        # Add all columns for the chosen LOS
        df_final = pd.concat([df_final, gal_df.loc[[chosen_idx]]], axis=0)

    return df_final

def plot_precut_av(df):
    """
    Plot pre-cut logMstar vs Sigma_SFR colored by median Av.

    Args:
        :df (pd.DataFrame): Input DataFrame with galaxy data.

    Returns:
        None
    """

    av_col = 'A_5542A'
    logm_col = 'logMstar'
    sig_col = 'Sigma_SFR'

    av_vals = np.log10(np.clip(df[av_col].values, 1e-6, None))
    av_min, av_max = np.nanmin(av_vals), np.nanmax(av_vals)
    norm = mcolors.Normalize(vmin=av_min, vmax=av_max)

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(df[logm_col], df[sig_col], c=av_vals, cmap="magma",
                    norm=norm, s=6, alpha=0.6)
    cbar = plt.colorbar(sc)
    plt.yscale('log')
    cbar.set_label("Median $\\log A_V$ across LOS (actual values)")
    plt.title("PRE-cut: all galaxies (color = median Av)")
    plt.xlabel("$\\log (M_{\\star}/M_{\\odot})$")
    plt.ylabel("$\\Sigma_{\\mathrm{SFR}}\\,(M_{\\odot}\\,\\mathrm{yr}^{-1}\\,\\mathrm{kpc}^{-2})$")

    plt.ylim(0.2e-5,1e2)
    plt.xlim(8.9,12)
    plt.tight_layout()
    preplot = 'figs/precut_logMstar_vs_SigmaSFR_colored_by_Av.png'
    plt.savefig(preplot, dpi=150)
    plt.close()
    print(f"Saved pre-cut plot: {preplot}")

    return


def plot_postcut_av(df, selected):
    """
    Plot post-cut logMstar vs Sigma_SFR colored by median Av.

    Args:
        :df (pd.DataFrame): Input DataFrame with galaxy data.
        :selected (pd.DataFrame): DataFrame of selected galaxies.

    Returns:
        None
    """

    av_col = 'A_5542A'
    logm_col = 'logMstar'
    sig_col = 'Sigma_SFR'

    av_vals = np.log10(np.clip(df[av_col].values, 1e-6, None))
    av_min, av_max = np.nanmin(av_vals), np.nanmax(av_vals)
    norm = mcolors.Normalize(vmin=av_min, vmax=av_max)

    av_vals_sel = np.log10(np.clip(selected["Av_med"].values, 1e-6, None))
    norm_sel = mcolors.Normalize(vmin=np.nanmin(av_vals_sel), vmax=np.nanmax(av_vals_sel))

    plt.figure(figsize=(7, 5))
    plt.scatter(selected[logm_col], selected[sig_col], s=25, facecolor='None', edgecolor='dimgrey', zorder=1000, lw=0.5)
    # add pre cut values in background
    sc = plt.scatter(df[logm_col], df[sig_col], c=av_vals, cmap="magma", norm=norm, s=5, alpha=0.3, zorder=1)
    cbar = plt.colorbar(sc)
    cbar.set_label("Median $\\log A_V$ across LOS (actual values)")
    plt.title("Post-cut: selected galaxies (adaptive logM★ bins; color = median Av)")
    plt.xlabel("$\\log (M_{\\star}/M_{\\odot})$")
    plt.ylabel("$\\Sigma_{\\mathrm{SFR}}\\,(M_{\\odot}\\,\\mathrm{yr}^{-1}\\,\\mathrm{kpc}^{-2})$")
    plt.yscale('log')
    plt.ylim(0.2e-5, 1e2)
    plt.xlim(8.9, 12)
    plt.tight_layout()
    postplot = "figs/postcut_logMstar_vs_SigmaSFR_colored_by_Av_median.png"
    plt.savefig(postplot, dpi=150)
    plt.close()
    print(f"Saved post-cut plot: {postplot}")

    return


def plot_postcutplus_av(df, selected, extremes):
    """
    Plot post-cut + Av-based additions (including extremes).

    Args:
        :df (pd.DataFrame): Input DataFrame with galaxy data.
        :selected (pd.DataFrame): DataFrame of selected galaxies.
        :extremes (pd.DataFrame): DataFrame of newly added extreme galaxies.

    Returns:
        None
    """

    plt.figure(figsize=(7, 5))

    galid_col = 'galaxy_id'
    av_col = 'A_5542A'
    logm_col = 'logMstar'
    sig_col = 'Sigma_SFR'

    selected_plus = selected.copy()

    av_vals = np.log10(np.clip(df[av_col].values, 1e-6, None))
    av_min, av_max = np.nanmin(av_vals), np.nanmax(av_vals)
    norm = mcolors.Normalize(vmin=av_min, vmax=av_max)

    # background: all galaxies colored by Av
    sc = plt.scatter(df[logm_col], df[sig_col],
                    c=av_vals, cmap="magma", norm=norm,
                    s=10, alpha=0.5, linewidths=0, zorder=1)
    cbar = plt.colorbar(sc)
    cbar.set_label("Median $\\log_{10}(A_V)$ across LOS")

    # selected subsample (including extremes): hollow grey rings
    plt.scatter(selected_plus[logm_col], selected_plus[sig_col],
                s=25, facecolors='none', edgecolors='dimgrey',
                linewidths=0.6, zorder=1000, label="Selected subsample")

    # highlight extremes if present
    try:
        # ensure 'galaxy_id' present on both
        if "galaxy_id" not in selected_plus.columns and galid_col in selected_plus.columns:
            selected_plus = selected_plus.rename(columns={galid_col: "galaxy_id"})
        if not extremes.empty:
            if "galaxy_id" not in extremes.columns and galid_col in extremes.columns:
                extremes = extremes.rename(columns={galid_col: "galaxy_id"})
            ext_mask = selected_plus["galaxy_id"].isin(extremes["galaxy_id"])
            if ext_mask.any():
                plt.scatter(selected_plus.loc[ext_mask, logm_col], selected_plus.loc[ext_mask, sig_col],
                            s=25, facecolors='none', edgecolors='magenta',
                            linewidths=1.6, zorder=1100, label="Added extremes")
    except Exception as e:
        print(f"(Plot note) Extremes highlight skipped: {e}")

    plt.xlabel("$\\log (M_{\\star}/M_{\\odot})$")
    plt.ylabel("$\\Sigma_{\\mathrm{SFR}}\\,(M_{\\odot}\\,\\mathrm{yr}^{-1}\\,\\mathrm{kpc}^{-2})$")
    plt.yscale('log')
    plt.title("Post-cut + Av-based additions (purple = extremes)")
    plt.ylim(0.2e-5, 1e2)
    plt.xlim(8.9, 12)
    plt.tight_layout()
    plt.legend(frameon=False)

    # save 
    postplot_plus = "figs/postcut_logMstar_vs_SigmaSFR_PLUS.png"
    plt.savefig(postplot_plus, dpi=150)
    plt.show()
    plt.close()

    print(f"Augmented plot: {postplot_plus}")

    return


def plot_iob_distributions(df, df_final):
    """
    Plot IOB distributions before and after selection.

    Args:
        :df (pd.DataFrame): Original DataFrame with all LOS data.
        :df_final (pd.DataFrame): DataFrame with selected galaxies.

    Returns:
        None
    """

    iob_cols = [col for col in df.columns if col.startswith('IOB')]

    data_before = df[iob_cols].to_numpy()
    data_after  = df_final[iob_cols].to_numpy()
    figure = corner.corner(data_before, labels=iob_cols, color='blue', label_kwargs={"fontsize":8},
                           hist_kwargs={"density":True, "color":"blue",}, smooth=1.0, plot_datapoints=False,
                           show_titles=False, title_args={"fontsize":8})
    corner.corner(data_after, fig=figure, labels=iob_cols, color='red', label_kwargs={"fontsize":8},
                    hist_kwargs={"density":True, "color":"red"}, smooth=1.0, plot_datapoints=False,
                    show_titles=False, title_args={"fontsize":8})
    # Add legend
    ax = figure.get_axes()[len(iob_cols)-1]
    ax.plot([], [], color='blue', label='Before selection')
    ax.plot([], [], color='red', label='After selection')
    ax.legend(frameon=False, fontsize=16)
    plotfile = "figs/iob_distributions_before_after_selection.png"
    figure.savefig(plotfile, dpi=150)
    print(f"Saved IOB distributions plot: {plotfile}")

    return


def select_gals(infile, exclude_file):

    df = load_df(infile)

    plot_precut_av(df)
    
    df_exclude = pd.read_csv(exclude_file, sep="\t")
    ids_to_exclude = df_exclude['galaxy_id'].unique()
    print(f"Excluding {len(ids_to_exclude)} galaxies from previous selections.")
    df = df[~df['galaxy_id'].isin(ids_to_exclude)].reset_index(drop=True)

    CHECK_START   = 61   # inclusive
    CHECK_END     = 70   # inclusive
    cols_to_check = list(df.columns[CHECK_START:CHECK_END + 1])
    print(df['galaxy_id'].nunique(), "unique galaxies before inconsistency check.")
    df = check_inconsistency(df, cols_to_check)
    print(df['galaxy_id'].nunique(), "unique galaxies after inconsistency check.")

    # Select galaxies using adaptive bins in logMstar
    selected, G, edges = select_adaptive_bins(df, 3, 30, 15)
    plot_postcut_av(df, selected)

    # Select some extremes based on Av and Sigma_SFR
    n_high = 24
    n_low  = 6
    sfr_high_thresh = 1e-1
    sfr_low_thresh  = 1e-3
    q_high = 0.84
    q_low  = 0.20
    rng_seed = 42
    selected, extremes = select_extremes(selected, G, edges, n_high, n_low, sfr_high_thresh, sfr_low_thresh, q_high, q_low, rng_seed=rng_seed)

    plot_postcutplus_av(df, selected, extremes)

    df_final = get_attenuation_curves(df, selected, seed=12345)

    plot_iob_distributions(df, df_final)

    # Check that none of the galaxy ids match those from the exclude file
    final_ids = set(df_final['galaxy_id'].unique())
    overlap_ids = final_ids.intersection(set(ids_to_exclude))
    if overlap_ids:
        raise RuntimeError(f"Selected galaxies include excluded IDs: {overlap_ids}")

    return None

if __name__ == "__main__":
    infile = "../data/gal_los_iobcomp_attcurve_galprop.dat"
    exclude_file = '../data/selected_galaxy_ids_by_logMstar_bin_with_extremes.txt'
    df = select_gals(infile, exclude_file)