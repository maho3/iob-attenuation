import pandas as pd
from sklearn.decomposition import PCA
from gplearn.genetic import SymbolicRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from librerie import*
from scipy.interpolate import interp1d
import os
from pysr import PySRRegressor
import numpy as np


###################useful constants##################
alphab=2.6e-13
G = 6.67e-8#cgs
bi = -2.5 #UV spectral slope for MW
yr=3.154e+7#in s
#p0=numpy.array([10**4.0,10**8.0])
Msol=1.99e33#g
Lsol=3.9e33#erg/s
mp=1.66e-24#g, proton mass
kb = 1.38064852e-16#cgs
h = 6.63e-27 #ergs
c = 3e10 #cm/s
pc=3.09e18#cm
kpc=1e3*pc
To = 2.725#TCMB
Zsole=0.0142
D=1.0/163.0#dust to gas ratio MW
cost_fd=1.0
kv=3.4822*1e4#cm^2/g, MW dust Draine+03


# --- 1. Load TNG data ---
# attenuation_curves: shape (N_galaxies, N_wavelengths)
# galaxy_props: shape (N_galaxies, N_properties)

# Load wavelength grid [micron]
lam = np.loadtxt('/Users/lsommovigo/Desktop/Scripts/txt_files/LTU_att_curve/wavelengths.txt')

# Load galaxy properties TNG50 and TNG100
ids_50, mstar_50, gas_mass_50, sfr_50, sfr_compact_50, gas_Z_50, rstar_50, rgas_50 = np.loadtxt(
    '/Users/lsommovigo/Desktop/Scripts/txt_files/LTU_att_curve/all_galaxy_properties_snap_93.dat',
    delimiter=',', unpack=True)
##
ids_100, mstar_100, gas_mass_100, sfr_100, sfr_compact_100, gas_Z_100, rstar_100, rgas_100 = np.loadtxt(
    '/Users/lsommovigo/Desktop/Scripts/txt_files/LTU_att_curve/tng100_all_galaxy_properties_snap_93.dat',
    delimiter=',', unpack=True)

dust_mass_50=10**gas_mass_50 * (gas_Z_50/Zsole/163.)
dust_mass_100=10**gas_mass_100 * (gas_Z_100/Zsole/163.)
sigma_SFR_50=sfr_50/(np.pi * rstar_50**2.)
sigma_SFR_100=sfr_100/(np.pi * rstar_100**2.)
sSFR_100=sfr_100/mstar_100
sSFR_50=sfr_50/mstar_50

# Combine all TNG IDs and properties into one consistent array
ids = np.concatenate([ids_50, ids_100])
properties = np.vstack([
    np.concatenate([ids_50, ids_100]),
    np.concatenate([mstar_50, mstar_100]),
    np.concatenate([gas_mass_50, gas_mass_100]),
    np.concatenate([sfr_50, sfr_100]),
    np.concatenate([sfr_compact_50, sfr_compact_100]),
    np.concatenate([gas_Z_50, gas_Z_100]),
    np.concatenate([rstar_50, rstar_100]),
    np.concatenate([rgas_50, rgas_100]),
    np.concatenate([dust_mass_50, dust_mass_100]),
    np.concatenate([sigma_SFR_50, sigma_SFR_100]),
    np.concatenate([sSFR_50, sSFR_100])
]).T  # shape: (n_galaxies, n_features)

# Paths to attenuation curves
TNG50_path = "/Users/lsommovigo/Desktop/Scripts/txt_files/LTU_att_curve/all_halos_output_snap93/output_snapnum93_shalo"
TNG100_path = "/Users/lsommovigo/Desktop/Scripts/txt_files/LTU_att_curve/TNG100_all_halos_output_snap93/output_snapnum93_shalo"

# Interpolation grid (common lambda grid)
#common_lam = np.linspace(0.1, 10., 300)

# Find index closest to V band (to normalize Att. curve)
def find_nearest(array, value):
    return np.abs(array - value).argmin()

log_attenuation_curves = []
galaxy_properties = []
galaxy_ids = []
los_indices = []

n_los = 51  ## of los per galaxy

for i, galaxy_id in enumerate(ids):
    file_base = TNG50_path if galaxy_id in ids_50 else TNG100_path
    file_path = os.path.join(file_base + str(int(round(galaxy_id))) + '.txt')

    if not os.path.exists(file_path):
        continue

    try:
        Flux_ratio = np.loadtxt(file_path)
    except Exception:
        continue

    for ll in range(n_los):
        Fr_arr = Flux_ratio[ll]
        valid = Fr_arr > 0
        if not np.any(valid):
            continue

        lam_arr = lam[valid]
        Alam_arr = -2.5 * np.log10(Fr_arr[valid])

        # Normalize to A(V)
        v_index = find_nearest(lam_arr, 0.551)
        Av = Alam_arr[v_index]
        if not np.isfinite(Av) or Av == 0:
            continue
        Alam_Av_arr = Alam_arr / Av
        
        # Store results
        if len(Alam_Av_arr[lam_arr<1.])!=50:
            print('err')
        
        # delete curves which are too weird
        if max(Alam_Av_arr[lam_arr<1.])<30 and min(Alam_Av_arr[lam_arr<1.])>0:
            log_attenuation_curves.append(np.log10(Alam_Av_arr[lam_arr<1.]))#Alam_interp)
            galaxy_properties.append(properties[i])
            galaxy_ids.append(galaxy_id)
            los_indices.append(ll)
        #print(properties[i][0])
        #plt.plot(1e4*lam_arr[lam_arr<1.],Alam_Av_arr[lam_arr<1.])
        #plt.show()

# Convert to numpy arrays
log_attenuation_curves = np.array(log_attenuation_curves)  # shape (n_valid_samples, n_wavelengths)
galaxy_props = np.array(galaxy_properties)    # shape (n_valid_samples, n_features)
galaxy_ids = np.array(galaxy_ids)
los_indices = np.array(los_indices)

print("Collected:", log_attenuation_curves.shape[0], "valid curves.")




'''
# ----------------------------------------------------------------------------------
# BLOCK 0: Perform PCA on log-attenuation curves (with optional wavelength weighting),
#        and evaluate how reconstruction quality (MSE) and variance explained vary
#        as a function of the number of components. This helps determine the
#        optimal n_components 
# ----------------------------------------------------------------------------------

# Scale the data
scaler = StandardScaler()
log_attenuation_scaled = scaler.fit_transform(log_attenuation_curves)


### reweighting att curves
lam_arr_valid=lam_arr[lam_arr<1.]
weights = np.ones_like(lam_arr_valid)
weights[lam_arr_valid < 0.3] = 1.#5.0  # Double importance for FUV
weights[(lam_arr_valid > 0.19) & (lam_arr_valid < 0.24)] = 1#0.0  # UV bump

# Apply weights before PCA
weighted_curves = log_attenuation_scaled * weights



# Set the range of components to test
component_range = [1, 2, 3, 4, 5]

# Store results
reconstructed_dict = {}
explained_variance_dict = {}
mse_dict = {}

for n in component_range:
    # Fit PCA
    pca = PCA(n_components=n)
    pcs = pca.fit_transform(log_attenuation_scaled)

    # Reconstruct from PCs
    reconstructed = scaler.inverse_transform(pca.inverse_transform(pcs))
    
    # Store
    reconstructed_dict[n] = reconstructed
    explained_variance_dict[n] = np.sum(pca.explained_variance_ratio_)
    mse_dict[n] = mean_squared_error(10**log_attenuation_curves, 10**reconstructed)


    print(f"PCA with {n} components — explained variance: {explained_variance_dict[n]:.3f}, MSE: {mse_dict[n]:.4f}")


plt.figure()
plt.plot(component_range, [explained_variance_dict[n] for n in component_range], marker='h',mew=0.8,mec='black',ls='--', color='crimson',ms=12,alpha=0.8)
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot (PCA)',fontsize=16,color='grey')
plt.grid(True,lw=0.6)
plt.tight_layout()
plt.show()


plt.figure()
plt.plot(component_range, [mse_dict[n] for n in component_range], marker='h',mew=0.8,mec='black',ls='--', color='crimson',ms=12,alpha=0.8)
plt.xlabel('Number of PCA Components')
plt.ylabel('Mean Squared Error (Reconstruction)')
plt.title('Reconstruction Quality vs Components',fontsize=16,color='grey')
plt.grid(True,lw=0.6)
plt.tight_layout()
plt.show()

def plot_recon_comparison_linear(idx=0):
    fig = plt.figure(figsize=(7, 8))
    ax = fig.add_subplot()

    true_curve = 10**log_attenuation_curves[idx]

    ax.plot(1e4 * lam_arr_valid, true_curve, label='True', linewidth=4, color='black', zorder=-1000, alpha=0.4)

    for n in component_range:
        recon_curve = 10**reconstructed_dict[n][idx]
        ax.plot(1e4 * lam_arr_valid, recon_curve, '--', label=f'{n} PC{"s" if n > 1 else ""}',
                lw=2.5 + 0.1 * 2, alpha=0.7, dashes=[n, n])

    x_tick_positions = [2175, 3543, 4770, 6231, 7625, 9134]
    ax.set_xticks(x_tick_positions)
    ax.set_xlabel('Wavelength $[\dot{A}]$')
    ax.set_xscale('log')
    ax.set_ylabel('$A_{\lambda}/A_V$')
    ax.set_title('Reconstruction Comparison (Linear Space)', fontsize=16, color='grey')
    ax.legend()
    ax.grid(True, lw=0.6, ls=':')
    ax.set_xlim(1e3, 9e3)
    ax.set_ylim(0, 12.9)
    plt.tight_layout()

    print('gal id =', int(galaxy_properties[idx][0]), '\n')
    plt.show()


def plot_recon_comparison(idx=0):
    fig=plt.figure(figsize=(7, 8))
    ax = fig.add_subplot()
    ax.plot(1e4*lam_arr_valid, log_attenuation_curves[idx], label='True', linewidth=4,color='black',zorder=-1000,alpha=0.4)

    for n in component_range:
        ax.plot(1e4*lam_arr_valid, reconstructed_dict[n][idx], '--', label=f'{n} PC{"s" if n > 1 else ""}', lw=2.5+0.1*2,alpha=0.7, dashes=[n,n])
    
    x_tick_positions = [2175, 3543, 4770, 6231, 7625, 9134]
    ax.set_xticks(x_tick_positions)
    ax.set_xlabel('Wavelength $[\dot{A}]$')
    ax.set_xscale('log')
    ax.set_ylabel('$A_{\lambda}/A_V$')
    ax.set_title(f'Reconstruction Comparison', fontsize=16,color='grey')
    ax.legend()
    ax.grid(True,lw=0.6,ls=':')
    ax.set_xlim(1e3,9e3)
    ax.set_ylim(0,12.9)
    plt.tight_layout()
    
    #gal id
    print('gal id =', int(galaxy_properties[idx][0]),'\n')
    plt.show()

# Example: Plot for one random line of sight and source

def get_index_by_id_los(target_id, target_los):
    for idx, (gid, los) in enumerate(zip(galaxy_ids, los_indices)):
        if gid == target_id and los == target_los:
            return idx
    raise ValueError(f"Could not find (ID={target_id}, LoS={target_los}) in stored data.")


for los in (47, 33, 32, 25):
    try:
        idx = get_index_by_id_los(42, los)
        print(f"Plotting: ID = 42, LoS = {los}, idx = {idx}")
        plot_recon_comparison_linear(idx=idx)
    except ValueError as e:
        print(e)




# ----------------------------------------------------------------------------------
# BLOCK1 : Perform PCA on log10-transformed attenuation curves (A_lam / AV),
#        using 3 components, and apply symbolic regression to predict each
#        PC from galaxy properties. Reconstruct the attenuation curves
#        from (a) true PCs and (b) symbolically predicted PCs, and compare
#        them in linear attenuation space (10^log(A_lam / AV)).
#        NB: This relies on gplearn (simpler version than the one after)
#        NB: This sucks ass
# ----------------------------------------------------------------------------------

# --- 2. PCA on log_attenuation curves ---
scaler = StandardScaler()
log_attenuation_scaled = scaler.fit_transform(log_attenuation_curves)

pca = PCA(n_components=3)
pcs = pca.fit_transform(log_attenuation_scaled)

# --- 3. Symbolic regression on each PC ---
X = galaxy_props# input features

symbolic_models = []
for i in range(pcs.shape[1]):
    y = pcs[:, i]  # target = i-th PC

    model = SymbolicRegressor(
        population_size=1000,
        generations=40,
        stopping_criteria=0.01,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        verbose=1,
        parsimony_coefficient=1e-5,
        random_state=42
    )
    model.fit(X, y)
    print(f"Best expression for PC{i+1}:\n", model._program)
    symbolic_models.append(model)



def plot_true_vs_reconstructed(log_attenuation_curves, reconstructed_curves, wavelengths=None, label=None, n_examples=10, random_seed=42):
    """
    Plots true vs reconstructed log_attenuation curves for a few random samples.
    
    Parameters:
    - log_attenuation_curves: np.ndarray, shape (n_samples, n_wavelengths)
    - reconstructed_curves: np.ndarray, same shape as log_attenuation_curves
    - wavelengths: np.ndarray or list, optional. Shape (n_wavelengths,)
    - n_examples: int, number of random examples to plot
    - random_seed: int, seed for reproducibility
    """
    np.random.seed(random_seed)
    n_samples = log_attenuation_curves.shape[0]
    indices = np.random.choice(n_samples, size=n_examples, replace=False)

    if wavelengths is None:
        wavelengths = np.arange(log_attenuation_curves.shape[1])

    for i, idx in enumerate(indices):
        plt.figure(figsize=(6, 4))
        #print(wavelengths.min(), wavelengths.max())
        plt.plot(wavelengths*1e4, 10**log_attenuation_curves[idx], label='True', linewidth=2)
        plt.plot(wavelengths*1e4, 10**reconstructed_curves[idx], '--', label=label, linewidth=2)
        plt.xlabel('$\lambda [\dot{A}]$')
        plt.ylabel('$A_\lambda/A_V$')
        plt.title(f'LoS #{idx} — True vs. Reconstructed')
        plt.xlim(1e3,1e4)
        plt.ylim(0,20)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



# Reconstruct log_attenuation curves from predicted PCs
predicted_pcs = np.column_stack([model.predict(X) for model in symbolic_models])

#with this I check how much information I lose by compressing -> decompressing via PCA (NO regression!)
# if it looks good PCA is doing a good job
reconstructed_direct = scaler.inverse_transform(pca.inverse_transform(pcs))
plot_true_vs_reconstructed(log_attenuation_curves, reconstructed_direct, wavelengths=lam_arr[lam_arr<1.],label='invert PCA')


#with this I test if the symbolic model (fed only galaxy properties) correctly predict log_attenuation shapes
# if it looks good model is learnable and interpretable
reconstructed_log_attenuation = scaler.inverse_transform(pca.inverse_transform(predicted_pcs))
plot_true_vs_reconstructed(log_attenuation_curves, reconstructed_log_attenuation, wavelengths=lam_arr[lam_arr<1.], label='reconstructed')


# ---  Evaluate reconstruction quality in linear space ---
from sklearn.metrics import mean_squared_error

mse_true = mean_squared_error(10**log_attenuation_curves, 10**reconstructed_direct)
mse_pred = mean_squared_error(10**log_attenuation_curves, 10**reconstructed_log_attenuation)

print(f"MSE (PCA only): {mse_true:.4f}")
print(f"MSE (Symbolic prediction): {mse_pred:.4f}")
'''


# ----------------------------------------------------------------------------------
# New attempt from Gal prop to Att Curves:
# ----------------------------------------------------------------------------------
# This script models attenuation curves using a two-step symbolic regression strategy.
#
# Phase 1:
# 1. Performs PCA on log10-transformed attenuation curves (A_lambda / A_V),
#    reducing them to a small number of principal components (PCs).
# 2. Uses symbolic regression (via PySR) to express the attenuation curve shape
#    as a single symbolic function of log10(wavelength) and PCs.
#
# Phase 2 (added at the end):
#    Symbolically regress each PCA component from galaxy properties,
#    to enable prediction of attenuation curve shape from physical inputs.
# ----------------------------------------------------------------------------------

# Step 1: Standardize log-attenuation curves
scaler = StandardScaler()
log_attenuation_scaled = scaler.fit_transform(log_attenuation_curves)

# Step 2: PCA to extract PCs
n_components = 3
pca = PCA(n_components=n_components)
pcs = pca.fit_transform(log_attenuation_scaled)

# Step 3: Fit a single symbolic model for log(A_lambda / A_V) as a function of log10(lambda) and PCs
n_samples, n_wavelengths = log_attenuation_scaled.shape

# Construct input matrix: [log10(lambda), PC1, PC2, PC3]
log_lambda = np.log10(lam_arr[lam_arr<1.])  # micron
log_lambda_tiled = np.tile(log_lambda, n_samples).reshape(-1, 1)
pcs_repeated = np.repeat(pcs, n_wavelengths, axis=0)
X_combined = np.hstack([log_lambda_tiled, pcs_repeated])
y_combined = log_attenuation_scaled.flatten()

## resampling
from sklearn.utils import resample
X_sub, y_sub = resample(X_combined, y_combined, n_samples=20000, random_state=42)


symbolic_model = PySRRegressor(
    niterations=40,
    population_size=1000,
    model_selection="best",
    maxsize=30,
    maxdepth=7,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log", "sin", "cos", "sqrt"],
    loss="loss(x, y) = (x - y)^2",
    verbosity=1,
    progress=True,
    random_state=42,
)
symbolic_model.fit(X_sub, y_sub)
print("Best symbolic model:")
print(symbolic_model.get_best())

# Step 4: Predict and reconstruct
predicted_log_curve = symbolic_model.predict(X_sub).reshape(n_samples, n_wavelengths)
reconstructed_A = 10 ** scaler.inverse_transform(predicted_log_curve)
true_A = 10 ** log_attenuation_curves

mse_symbolic = mean_squared_error(true_A, reconstructed_A)
print(f"MSE (log(lambda), PCs → Curve): {mse_symbolic:.4f}")

# Step 5: Plot a sample reconstruction
def plot_curve_model(idx):
    plt.figure(figsize=(6, 4))
    plt.plot(common_lam * 1e4, true_A[idx], label="True", lw=2, color="black")
    plt.plot(common_lam * 1e4, reconstructed_A[idx], label="Symbolic", lw=2, ls=":")
    plt.xlabel("Wavelength [$\u212b$]")
    plt.ylabel(r"$A_\lambda / A_V$")
    plt.title(f"LoS #{idx}: log(lambda), PCs → Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example
plot_curve_model(idx=np.random.randint(len(true_A)))

# Example
plot_curve_model(idx=np.random.randint(len(true_A)))

# ----------------------------------------------------------------------------------
# Phase 2: Predict PCA components from galaxy properties using symbolic regression
# ----------------------------------------------------------------------------------
symbolic_pc_models = []

for i in range(n_components):
    y = pcs[:, i]
    X = galaxy_props

    model = PySRRegressor(
        niterations=40,
        population_size=1000,
        model_selection="best",
        maxsize=20,
        maxdepth=5,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sin", "cos", "sqrt"],
        loss="loss(x, y) = (x - y)^2",
        verbosity=1,
        progress=True,
        random_state=42,
    )
    model.fit(X, y)
    symbolic_pc_models.append(model)
    print(f"Best symbolic expression for PC{i+1}:")
    print(model.get_best())

# Predict PCs from galaxy properties, then reconstruct curve from symbolic models
predicted_pcs = np.column_stack([model.predict(galaxy_props) for model in symbolic_pc_models])
predicted_log_A = np.column_stack([model.predict(predicted_pcs) for model in symbolic_curve_models])
predicted_A = 10 ** scaler.inverse_transform(predicted_log_A)

mse_final = mean_squared_error(true_A, predicted_A)
print(f"MSE (Galaxy Props → PCs → Curve): {mse_final:.4f}")

# Optional: Plot final end-to-end reconstruction
plot_curve_model(idx=np.random.randint(len(true_A)))


# ----------------------------------------------------------------------------------
# BLOCK: Perform PCA on log10-transformed attenuation curves (A_lam / AV),
#        using 3 components, and apply symbolic regression to predict each
#        PC from galaxy properties. Reconstruct the attenuation curves
#        from (a) true PCs and (b) symbolically predicted PCs, and compare
#        them in linear attenuation space (10^log(A_lam / AV)).
#        NB: This relies on PySr
# ----------------------------------------------------------------------------------

# Step 1: Standardize log-attenuation curves
scaler = StandardScaler()
log_attenuation_scaled = scaler.fit_transform(log_attenuation_curves)

# Step 2: PCA (choose fixed n_components=3, maybe change?)
n_components = 3
from sklearn.decomposition import PCA
pca = PCA(n_components=n_components)
pcs = pca.fit_transform(log_attenuation_scaled)

# Step 3: Symbolic regression with PySR for each PC
symbolic_models = []
for i in range(n_components):
    y = pcs[:, i]
    model = PySRRegressor(
        niterations=40,
        population_size=1000,
        model_selection="best",
        maxsize=20,
        maxdepth=5,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sin", "cos", "sqrt"],
        loss="loss(x, y) = (x - y)^2",
        verbosity=1,
        progress=True,
        random_state=42,
    )
    model.fit(galaxy_props, y)
    symbolic_models.append(model)
    print(f"Best expression for PC{i+1}:")
    print(model.get_best())

# Step 4: Reconstruct attenuation curves from predicted PCs
predicted_pcs = np.column_stack([model.predict(galaxy_props) for model in symbolic_models])
reconstructed_log = scaler.inverse_transform(pca.inverse_transform(predicted_pcs))
reconstructed_direct = scaler.inverse_transform(pca.inverse_transform(pcs))

# Step 5: Compare in linear attenuation space
true_A = 10 ** log_attenuation_curves
recon_A_direct = 10 ** reconstructed_direct
recon_A_symbolic = 10 ** reconstructed_log

mse_direct = mean_squared_error(true_A, recon_A_direct)
mse_symbolic = mean_squared_error(true_A, recon_A_symbolic)

print(f"MSE (PCA only): {mse_direct:.4f}")
print(f"MSE (Symbolic prediction): {mse_symbolic:.4f}")

# Step 6: Plot example reconstructions
def plot_recon_example(idx):
    plt.figure(figsize=(6, 4))
    plt.plot(common_lam * 1e4, true_A[idx], label="True", lw=2, color="black")
    plt.plot(common_lam * 1e4, recon_A_direct[idx], label="PCA", lw=2, ls="--")
    plt.plot(common_lam * 1e4, recon_A_symbolic[idx], label="Symbolic", lw=2, ls=":")
    plt.xlabel("Wavelength [$Å$]")
    plt.ylabel(r"$A_\lambda / A_V$")
    plt.title(f"LoS #{idx}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example
plot_recon_example(idx=np.random.randint(len(true_A)))







