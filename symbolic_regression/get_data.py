from posixpath import dirname
import numpy as np
from utils import OperonArgs
from os.path import join as pjoin
import os
import argparse
import pandas as pd
import re

import select_gals

def get_data(ini_file, overwrite=False):

    args = OperonArgs(ini_file)

    files_exist = (os.path.exists(args.selection.train_file)
                     and os.path.exists(args.selection.val_file)
                     and os.path.exists(args.selection.test_file))

    # First sub-sample the data into training, validation, and test sets    
    if (not files_exist) or overwrite:
        print('Selecting galaxies for training, validation, and test sets...')

        if args.selection.method == 'laura':
            df_train, df_val, df_test = select_gals.select_gals_laura(args.selection)
        elif args.selection.method == 'random':
            df_train, df_val, df_test = select_gals.select_gals_random(args.selection)
        else:
            raise ValueError(f"Selection method '{args.selection.method}' not recognised.")

        print('Number of selected galaxies:')
        print('\tTraining:', df_train['galaxy_id'].nunique())
        print('\tValidation:', df_val['galaxy_id'].nunique())
        print('\tTest:', df_test['galaxy_id'].nunique())

        # Verify no overlaps
        train_set = set(df_train['galaxy_id'].unique())
        val_set   = set(df_val['galaxy_id'].unique())
        test_set  = set(df_test['galaxy_id'].unique())
        assert train_set.isdisjoint(val_set), "Overlap between Train and Val sets!"
        assert train_set.isdisjoint(test_set), "Overlap between Train and Test sets!"
        assert val_set.isdisjoint(test_set),   "Overlap between Val and Test sets!"
        assert len(train_set) == df_train.shape[0], "Duplicate galaxy_ids in Train set!"
        assert len(val_set)   == df_val.shape[0],   "Duplicate galaxy_ids in Val set!"
        assert len(test_set)  == df_test.shape[0],  "Duplicate galaxy_ids in Test set!"

        # Save results to file
        print('Saving training data to:', args.selection.train_file)
        df_train.to_csv(args.selection.train_file, index=False)
        print('Saving validation data to:', args.selection.val_file)
        df_val.to_csv(args.selection.val_file, index=False)
        print('Saving test data to:', args.selection.test_file)
        df_test.to_csv(args.selection.test_file, index=False)

    # Load the data from files
    df_train = pd.read_csv(args.selection.train_file)
    df_val   = pd.read_csv(args.selection.val_file)
    df_test  = pd.read_csv(args.selection.test_file)
    
    # Now process the data for symbolic regression
    for name, df in zip(['train', 'val', 'test'], [df_train, df_val, df_test]):

        print(f'\nProcessing {name} data...')

        # Create output directory
        out_dirname = pjoin(args.out_data_dir, f'{args.in_param}_data_{args.version_num}')
        os.makedirs(out_dirname, exist_ok=True)
        outname = pjoin(out_dirname, f'{args.in_param}_{name}_data.txt')
        if (not overwrite) and os.path.exists(outname):
            print(f'{name.capitalize()} data file already exists at {outname}, skipping processing...')
            continue
        
        # Get attenuation column names
        log_Av_col = f'logA_{int(args.lambda_V*1e4)}A'
        A_v_col = f'A_{int(args.lambda_V*1e4)}A'
        to_norm = False
        if log_Av_col in df.keys():
            attenuation_cols = [col for col in df.columns if re.match(r'logA_\d+A', col)]
            if not np.all(df[log_Av_col] == 0):
                print(f'\tWarning: {log_Av_col} column is not all zeros: normalising data')
                to_norm = True
        elif A_v_col in df.keys():
            attenuation_cols = [col for col in df.columns if re.match(r'A_\d+A', col)]
            if not np.all(df[A_v_col] == 1):
                print(f'\tWarning: {A_v_col} column is not all ones: normalising data')
                to_norm = True
        else:
            raise ValueError("Column with lambda_V not found in input file")
        
        # Get wavelengths in angstroms and mask to [lam_min, lam_max]
        lam_arr = np.array([int(re.search(r'_(\d+)A', col).group(1)) / 1e4 for col in attenuation_cols])
        sort_idx = np.argsort(lam_arr)
        lam_arr = lam_arr[sort_idx]
        mask = (lam_arr < args.lam_max) & (lam_arr > args.lam_min)
        lam_arr = lam_arr[mask]
        attenuation_cols = [attenuation_cols[i] for i in sort_idx if mask[i]]
        lam_arr = lam_arr / args.lambda_V

        # Subsample wavelengths if specified
        if args.lam_trans is not None and args.f_subsample is not None:
            print(f'\tSubsampling wavelengths above {args.lam_trans} by a factor of {args.f_subsample}')
            lam_trans = args.lam_trans / args.lambda_V
            mask_below = lam_arr < lam_trans
            idx_above = np.nonzero(lam_arr >= lam_trans)[0]
            mask_above = np.zeros_like(mask_below)
            mask_above[idx_above[::args.f_subsample]] = True
            mask = mask_below | mask_above
            lam_arr = lam_arr[mask]
            attenuation_cols = [attenuation_cols[i] for i in range(len(attenuation_cols)) if mask[i]]
            print('\tSubsampled wavelengths:', lam_arr * args.lambda_V)
            print('\tNumber of wavelengths after subsampling:', len(lam_arr), 'from', len(mask))
        else:
            print('\tNo subsampling of wavelengths')

        # Get parameters used as input
        in_cols = [col for col in df.columns if col.startswith(args.in_param.upper()) and col[len(args.in_param):].isdigit()]
        in_cols.sort()
        print(f'\tNumber of input parameters found: {len(in_cols)}')
        in_cols = in_cols[:args.npar]
        print(f'\tNumber of input parameters used: {len(in_cols)}')

        ngal = len(df)
        nlam = len(lam_arr)
        print(f'\tProcessing {name} data with {ngal} galaxies and {nlam} wavelengths...')
        properties = df[in_cols].values
        props_repeated = np.repeat(properties, nlam, axis=0)
        galid_repearted = np.repeat(df['galaxy_id'].values, nlam).reshape(-1, 1)
        los_repeated = np.repeat(df['los'].values, nlam).reshape(-1, 1)
        wavelengths_tiled = np.tile(lam_arr, ngal).reshape(-1, 1)
        curves_flat = df[attenuation_cols].values.reshape(-1, 1)

        if to_norm:
            print('\tNormalising data...')
            if log_Av_col in df.keys():
                curves_flat = curves_flat - np.repeat(df[log_Av_col].values, nlam).reshape(-1, 1)
            elif A_v_col in df.keys():
                curves_flat = curves_flat / np.repeat(df[A_v_col].values, nlam).reshape(-1, 1)

        if log_Av_col in df.keys():
            print('\tConverting logA to A')
            curves_flat = 10.**curves_flat

        # Save output
        output_array = np.hstack((galid_repearted, los_repeated, props_repeated, wavelengths_tiled, curves_flat))
        print(f'\tSaving {name.capitalize()} data of shape {output_array.shape} to {out_dirname}')
        header = ' '.join(['galaxy_id', 'los'] + in_cols + ['lam', 'A'])
        np.savetxt(outname, output_array, header=header, comments='')

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get data with a specified config file.")
    parser.add_argument("config_path", help="Path to the configuration file.")
    args = parser.parse_args()
    get_data(args.config_path, True)