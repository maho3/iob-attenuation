import numpy as np

# FUNCTIONS
# Attenuation curves (see Overleaf)
def Li_08(lam_micron, c1, c2, c3, c4, model):
    """
    Computes an attenuation curve A(λ)/A_V following the Li+08 parameterization,
    with optional override for specific galaxy templates (Calzetti, SMC, MW, LMC).

    Parameters
    ----------
    lam_micron : float or ndarray
        Wavelength(s) in microns.
    c1, c2, c3, c4 : float
        Free parameters controlling the curve shape (ignored if model is specified).
    model : str
        If set to one of 'Calzetti', 'SMC', 'MW', 'LMC', uses fixed parameter values.

    Returns
    -------
    A_lam_v : float or ndarray
        Attenuation curve values A(λ)/A_V at the input wavelength(s).
    """

    # Override parameters for named models
    if model == 'Calzetti':
        c1, c2, c3, c4 = 44.9, 7.56, 61.2, 0.
    elif model == 'SMC':
        c1, c2, c3, c4 = 38.7, 3.83, 6.34, 0.
    elif model == 'MW':
        c1, c2, c3, c4 = 14.4, 6.52, 2.04, 0.0519
    elif model == 'LMC':
        c1, c2, c3, c4 = 4.47, 2.39, -0.988, 0.0221
    elif model == 'None':
        pass  # Use supplied c1–c4

    # Li+08 three-term parameterization, normalized to A_V
    A_lam_v = (
        c1 / ((lam_micron / 0.08)**c2 + (lam_micron / 0.08)**-c2 + c3) +
        (233. * (1 - c1 / (6.88**c2 + 0.145**c2 + c3) - c4 / 4.6)) /
        ((lam_micron / 0.046)**2. + (lam_micron / 0.046)**-2. + 90.) +
        c4 / ((lam_micron / 0.2175)**2. + (lam_micron / 0.2175)**-2. - 1.95)
    )

    return A_lam_v


def Li_08_fit_noratio(lam_micron, c1, c2, c3, c4):
    """
    Computes the Li+08 attenuation curve A(λ)/A_V without overriding parameters.
    Used for direct fitting with custom parameter values.

    Parameters
    ----------
    lam_micron : float or ndarray
        Wavelength(s) in microns.
    c1, c2, c3, c4 : float
        Free parameters controlling the shape of the attenuation curve.

    Returns
    -------
    A_lam_v : float or ndarray
        Attenuation curve values A(λ)/A_V at the input wavelength(s).
    """
    A_lam_v = c1 / ((lam_micron/0.08)**c2 + (lam_micron/0.08)**-c2 + c3)  +   (233. * (1. - c1/(6.88**c2 + 0.145**c2 +c3) - c4/4.6)) / ((lam_micron/0.046)**2. + (lam_micron/0.046)**-2. + 90.) +  c4/ ((lam_micron/0.2175)**2. + (lam_micron/0.2175)**-2. - 1.95)
    return A_lam_v

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def k_calzetti2000(wavelength):
    """Compute the Calzetti et al. (2000) A(λ)/E(B-V)∗

    Given a wavelength grid, this function computes the selective attenuation
    A(λ)/E(B-V)∗ using the formula from Calzetti at al. (2000). This formula
    is given for wavelengths between 120 nm and 2200 nm, but this function
    makes the computation outside.

    Parameters
    ----------
    wavelength: array of floats
        Wavelength grid in nm.

    Returns
    -------
    a numpy array of floats

    """
    wavelength = np.array(wavelength)
    result = np.zeros(len(wavelength))

    # Attenuation between 120 nm and 630 nm
    mask = (wavelength < 630)
    result[mask] = 2.659 * (-2.156 + 1.509e3 / wavelength[mask] -
                            0.198e6 / wavelength[mask] ** 2 +
                            0.011e9 / wavelength[mask] ** 3) + 4.05

    # Attenuation between 630 nm and 2200 nm
    mask = (wavelength >= 630)
    result[mask] = 2.659 * (-1.857 + 1.040e3 / wavelength[mask]) + 4.05

    return result


def k_leitherer2002(wavelength):
    """Compute the Leitherer et al. (2002) A(λ)/E(B-V)∗

    Given a wavelength grid, this function computes the selective attenuation
    A(λ)/E(B-V)∗ using the formula from Leitherer at al. (2002). This formula
    is given for wavelengths between 91.2 nm and 180 nm, but this function
    makes the computation outside.

    Parameters
    ----------
    wavelength: array of floats
        Wavelength grid in nm.

    Returns
    -------
    a numpy array of floats

    """
    wavelength = np.array(wavelength)
    result = (5.472 + 0.671e3 / wavelength -
              9.218e3 / wavelength ** 2 +
              2.620e6 / wavelength ** 3)

    return result


def uv_bump(wavelength, central_wave, gamma, ebump):
    """Compute the Lorentzian-like Drude profile.

    Parameters
    ----------
    wavelength: array of floats
        Wavelength grid in nm.
    central_wave: float
        Central wavelength of the bump in nm.
    gamma: float
        Width (FWHM) of the bump in nm.
    ebump: float
        Amplitude of the bump.

    Returns
    -------
    a numpy array of floats

    """
    return (ebump * wavelength ** 2 * gamma ** 2 /
            ((wavelength ** 2 - central_wave ** 2) ** 2 +
             wavelength ** 2 * gamma ** 2))


def power_law(wavelength, delta):
    """Power law 'centered' on 550 nm..

    Parameters
    ----------
    wavelength: array of floats
        The wavelength grid in nm.
    delta: float
        The slope of the power law.

    Returns
    -------
    array of floats

    """
    return (wavelength / 550) ** delta


def a_vs_ebv(wavelength, bump_wave, bump_width, bump_ampl, power_slope):
    """Compute the complete attenuation curve A(λ)/E(B-V)*

    The Leitherer et al. (2002) formula is used between 91.2 nm and 150 nm, and
    the Calzetti et al. (2000) formula is used after 150 (we do an
    extrapolation after 2200 nm). When the attenuation becomes negative, it is
    kept to 0. This continuum is multiplied by the power law and then the UV
    bump is added.

    Parameters
    ----------
    wavelength: array of floats
        The wavelength grid (in nm) to compute the attenuation curve on.
    bump_wave: float
        Central wavelength (in nm) of the UV bump.
    bump_width: float
        Width (FWHM, in nm) of the UV bump.
    bump_ampl: float
        Amplitude of the UV bump.
    power_slope: float
        Slope of the power law.

    Returns
    -------
    attenuation: array of floats
        The A(λ)/E(B-V)* attenuation at each wavelength of the grid.

    """
    attenuation = np.zeros(len(wavelength))

    # Leitherer et al.
    mask = (wavelength > 91.2) & (wavelength < 150)
    attenuation[mask] = k_leitherer2002(wavelength[mask])
    # Calzetti et al.
    mask = (wavelength >= 150)
    attenuation[mask] = k_calzetti2000(wavelength[mask])
    # We set attenuation to 0 where it becomes negative
    mask = (attenuation < 0)
    attenuation[mask] = 0
    # Power law
    attenuation *= power_law(wavelength, power_slope)

    # As the powerlaw slope changes E(B-V), we correct this so that the curve
    # always has the same E(B-V) as the starburst curve. This ensures that the
    # E(B-V) requested by the user is the actual E(B-V) of the curve.
    wl_BV = np.array([440., 550.])
    EBV_calz = (k_calzetti2000(wl_BV) * power_law(wl_BV, 0.)) # SS
    EBV = (k_calzetti2000(wl_BV) * power_law(wl_BV, power_slope)) # SS
    attenuation *= (EBV_calz[1]-EBV_calz[0]) / (EBV[1]-EBV[0])

    # UV bump
    attenuation += uv_bump(wavelength, bump_wave, bump_width, bump_ampl) # do bump at the end (SS)

    
    return attenuation


def Att_Curve_2param(lamda, B, delta):
    return a_vs_ebv(lamda/10., 217.5, 35., B, delta)/a_vs_ebv(550.*np.ones(2), 217.5, 35., B, delta)[0]