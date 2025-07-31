import numpy as np
from utils import OperonArgs
from os.path import join as pjoin
import os
import argparse
import pandas as pd
import re


def get_data(ini_file):

    args = OperonArgs(ini_file)

    print('\nSplitting data into training, validation, and test sets...')
    print('\tFile:', args.input_file)

    # Read all galaxy data
    data = pd.read_csv(args.input_file, sep='\t',)
    print(data.head())

    # Get attenuation column names
    if f'logA_{int(args.lambda_V*1e4)}A' in data.keys():
        attenuation_cols = [col for col in data.columns if re.match(r'logA_\d+A', col)]
        if not np.all(data[f'logA_{int(args.lambda_V*1e4)}A'] == 0):
            print('\tWarning: logA_V column is not all zeros: normalising data')
            for col in attenuation_cols:
                data[col] = data[col] - data[f'logA_{int(args.lambda_V*1e4)}A']
            assert np.all(data[f'logA_{int(args.lambda_V*1e4)}A'] == 0), "Normalisation failed"
    elif f'A_{int(args.lambda_V*1e4)}A' in data.keys():
        attenuation_cols = [col for col in data.columns if re.match(r'A_\d+A', col)]
        if not np.all(data[f'A_{int(args.lambda_V*1e4)}A'] == 1):
            print('\tWarning: A_V column is not all ones: normalising data')
            for col in attenuation_cols:
                data[col] = data[col] / data[f'A_{int(args.lambda_V*1e4)}A']
            assert np.all(data[f'A_{int(args.lambda_V*1e4)}A'] == 1), "Normalisation failed"
    else:
        raise ValueError("Column with lambda_V not found in input file")

    # Extract the wavelength in angstroms
    # Mask these to range [lam_min, lam_max]
    lam_arr = np.array([int(re.search(r'_(\d+)A', col).group(1)) / 1e4 for col in attenuation_cols])
    sort_idx = np.argsort(lam_arr)
    lam_arr = lam_arr[sort_idx]
    mask = (lam_arr < args.lam_max) & (lam_arr > args.lam_min)
    lam_arr = lam_arr[mask]
    attenuation_cols = [attenuation_cols[i] for i in sort_idx if mask[i]]
    print('\tWavelengths:', lam_arr)
    lam_arr = lam_arr / args.lambda_V

    # Get all galaxy ids
    galaxy_ids = data['galaxy_id'].unique()
    print('\tNumber of unique galaxies:', len(galaxy_ids), 'of', len(data), 'attenuation curves')

    # Estimate number of unique IDs needed
    ntrain = int(args.ntrain / (args.ntrain + args.nval + args.ntest) * len(galaxy_ids))
    nval= int(args.nval / (args.ntrain + args.nval + args.ntest) * len(galaxy_ids))
    ntest = len(galaxy_ids) - nval - ntrain
    print(f'\tNumber of galaxies for training: {ntrain}, validation: {nval}, test: {ntest}')
    
    # Shuffle the galaxy ids and split into training, validation, and test sets
    np.random.seed(args.seed)
    np.random.shuffle(galaxy_ids)
    train_ids = galaxy_ids[:ntrain]
    val_ids = galaxy_ids[ntrain:ntrain+nval]
    test_ids = galaxy_ids[ntrain+nval:ntrain+nval+ntest]

    # Split data
    m = np.isin(data['galaxy_id'], train_ids)
    train_data = data[m]
    m = np.isin(data['galaxy_id'], val_ids)
    val_data = data[m]
    m = np.isin(data['galaxy_id'], test_ids)
    test_data = data[m]

    print('\tOriginal training data shape:', train_data.shape)
    print('\tOriginal validation data shape:', val_data.shape)
    print('\tOriginal test data shape:', test_data.shape)

    # Each galaxy can have more than one los, so we have too many points now
    # We again shuffle the data and reduce the number of objects
    train_data = train_data.iloc[np.random.permutation(train_data.shape[0])[:args.ntrain],:]
    val_data = val_data.iloc[np.random.permutation(val_data.shape[0])[:args.nval], :]
    test_data = test_data.iloc[np.random.permutation(test_data.shape[0])[:args.ntest], :]

    print('\tNumber of training curves:', train_data.shape[0])
    print('\tNumber of validation curves:', val_data.shape[0])
    print('\tNumber of test curves:', test_data.shape[0])

    # Get cols beginning with in_param and the rest is an integer
    in_cols = [col for col in data.columns if col.startswith(args.in_param.upper()) and col[len(args.in_param):].isdigit()]
    in_cols.sort()
    print(f'\tNumber of input parameters found: {len(in_cols)}')
    in_cols = in_cols[:args.npar]
    print(f'\tNumber of inputparameters used: {len(in_cols)}')

    # Make output directory
    dirname = pjoin(args.data_dir, f'{args.in_param}_data_{args.version_num}')
    os.makedirs(dirname, exist_ok=True)

    for name, data_set in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
        ngal = len(data_set)
        nlam = len(lam_arr)
        properties = data_set[in_cols].values
        props_repeated = np.repeat(properties, nlam, axis=0)
        wavelengths_tiled = np.tile(lam_arr, ngal).reshape(-1, 1)
        curves_flat = data_set[attenuation_cols].values.reshape(-1, 1)
        output_array = np.hstack((props_repeated, wavelengths_tiled, curves_flat))
        print(f'Saving {name.capitalize()} data of shape {output_array.shape} to {dirname} ...')
        if args.fit_log:
            header = ' '.join(in_cols + ['lam', 'log10A'])
        else:
            header = ' '.join(in_cols + ['lam', 'A'])
        outname = pjoin(dirname, f'{args.in_param}_{name}_data.txt')
        np.savetxt(outname, output_array, header=header, comments='')

        # Now save the galaxy ids and los in case we need them later
        output_array = data_set[['galaxy_id', 'los']].values
        outname = pjoin(dirname, f'{args.in_param}_{name}_galaxy_ids_los.txt')
        np.savetxt(outname, output_array, fmt='%d', header='\t'.join(['galaxy_id', 'los']), delimiter='\t')

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get data with a specified config file.")
    parser.add_argument("config_path", help="Path to the configuration file.")
    args = parser.parse_args()
    get_data(args.config_path)