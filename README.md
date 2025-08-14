# iob-attenuation
Code to run ordered compression on dust attenuation curves

# questions
- Why did Laura cut the max. wavelength at 1 micron?
    - Match SDSS filters
    - (We care least about this part of the curve)
- Why did Laura cut the minimum wavelength at 0.12 microns for the 4-parameter fit?
    - Prevent tracing other components
    - Ionising photos are attempted to be fitted by the 4-parameter fit

- Ensure data cut to make sure we don't get the weird LOSs where observer near dust-free regions?
    - See Laura's code

- LOS is fixed relative to the box and not the galaxy, so this is not a property one would include in a fit

IOB/PCA -> Functional form -> Do fit of free parameters with this code -> Can you predict those free parameters from physical properties?
Simple and pretty!

# Done 

Plot ratio of fluxes - not just the attenuation curve
- This is what Laura did in her paper
- Try two validations sets - one with large AV and another with low AV. For 2- and 4- parameter fits, these two different significantly in performance, so it would be interesting to see whether the new fits have similar performance for both. Cut is around A_V > 0.7 for high and for low A_V < 0.2
- Compare the results of our fits to the literature functions
- Run multiple fits with different settings (epsilon, basis sets etc.)

# To Do

- Create script to rerun optimisation of galaxy-dependent parameters for a SR function


# Ideas

- A main problems with the new fits are
    1. We don't get A/Av=1 and Lv guaranteed
    2. Beyond Lv, the attenuation curve is approximately consant but galaxy-dependent. But the curves never get this bit right
- Solutions:
    1. We just enforce this manually afterwards for the functions we like the look of
    2. When re-optimising the functions afterwards, we could add a galaxy-dependent offset parameter. Alternatively, we could give A/Av at the final wavelength as an input feauture alongside the IOB parameters, so it could be used in the fit.