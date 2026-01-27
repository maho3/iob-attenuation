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
- Create script to rerun optimisation of galaxy-dependent parameters for a SR function. Use the IOB parameters as an initial guess for the optimiser

# To Do

- Find requirements for SDSS catalogs for LtU in terms of DF/F
- 4-parameeter fit has a problem with degeneracy
    * A good result could be less degenerate parameters even if the fits are similar in quality
    * Re-parameterise the 4-parameter model to remove some degeneracies
- Send Laura an example function so we can
    1. Fix A/Av=1 at lv (by Sep.)
    2. Set the offset  (by Sep.)
    3. Run MCMC to see if our parameters are less degenerate (few examples by Sep.)
    4. Are these parameters more correlated with galaxy properties? (post Sep.)
- Optimise the curves which appear in PCA code but not IOB
- Do a fit with just first 4 IOB parameters
- Subsample the attenuation curve to remove the latger wavelengths



# Ideas/Comments

1. We don't get A/Av=1 and Lv guaranteed
    * We just enforce this manually afterwards for the functions we like the look of

2. Beyond Lv, the attenuation curve is approximately constant but galaxy-dependent. But the curves never get this bit right
    * When re-optimising the functions afterwards, we could add a galaxy-dependent offset parameter. 
    * Alternatively, we could give A/Av at the final wavelength as an input feauture alongside the IOB parameters, so it could be used in the fit.

3. We don't care so much about the high-wavelength part of the curve. This is partly solved by using MSE on Alamda/Av instead of on log of this
    * Could also sub-sample wavelengths at large lambda so we have fewer points out there

4. The functions at the knee of the Pareto front don't always have IOB3 or IOB4, yet these were needed to get a converged MSE in the IOB code
    * Always select functions with these in
    * Can we make the decoder of the IOB smaller so that they appear more simply in the expressions? Perhaps it is too complicated to add them for the gain they provide

# Discussion 17 November

* Need $d \exp(A_\lambda) / d \lambda$ well behaved (transmission function) - want derivative of $F$ to be good

# Discussion 22 January

* Could we ensure that the IOB parameters the bump uses are different from those the outer part uses?