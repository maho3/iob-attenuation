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

Plot ratio of fluxes - not just the attenuation curve
- This is what Laura did in her paper

# To Do

- Add the ratio of fluxes plot to processing_fun and notebook
- Run fits for all galaxies using the literature models