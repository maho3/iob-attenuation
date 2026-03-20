import os
import pandas as pd
import io
import re
import numpy as np
from functools import reduce
import argparse

from utils import OperonArgs

def get_data(ini_file, overwrite=False,):

    args = OperonArgs(ini_file)

    files_exist = (os.path.exists(args.train_file)
                     and os.path.exists(args.val_file)
                     and os.path.exists(args.test_file))
    
    if files_exist and (not overwrite):
        print('Data files already exist.')
        return

    # Set random seed for pandas
    np.random.seed(args.seed)

    all_galaxy_ids = []

    if isinstance(args.dust_mixture, str):
        all_dust_mixtures = [args.dust_mixture]
    else:
        all_dust_mixtures = ['MW', 'SMC']
    
    for dust_mixture in all_dust_mixtures:
        print('Loading data for dust mixture:', dust_mixture)
        fname = os.path.join(args.in_data_dir, f'merged_bestfit_galprops_{dust_mixture}.dat')
        df = pd.read_csv(fname, sep=r'\s+', usecols=["#galaxy_id"])
        galaxy_ids = df['#galaxy_id'].unique()
        all_galaxy_ids.append(galaxy_ids)
        print(f"\tNumber of unique galaxy_ids: {len(galaxy_ids)}")
        print(f"\tTotal number of rows:", df.shape[0])

    # Find galaxy IDs which are in all files
    common_galaxy_ids = reduce(np.intersect1d, all_galaxy_ids)

    print(f"Number of common galaxy_ids across all dust mixtures: {len(common_galaxy_ids)}")

    # If we have enough galaxies to have only one los for each one, then do this.
    # Othwerise, we will need multiple los for each galaxy. But we will make
    # sure no galaxy appears in more than one of the train/val/test sets.
    ntot = args.ntrain + args.nval + args.ntest
    if len(common_galaxy_ids) < ntot:
        print(f"Not enough common galaxy_ids to have only one los for each one")
        ntrain = int(args.ntrain / ntot * len(common_galaxy_ids))
        nval = int(args.nval / ntot * len(common_galaxy_ids))
        ntest = int(args.ntest / ntot * len(common_galaxy_ids))
    else:
        ntrain = args.ntrain
        nval = args.nval
        ntest = args.ntest
    gals_train = np.random.choice(common_galaxy_ids, size=ntrain, replace=False)
    remaining_gals = np.setdiff1d(common_galaxy_ids, gals_train)
    gals_val = np.random.choice(remaining_gals, size=nval, replace=False)
    remaining_gals = np.setdiff1d(remaining_gals, gals_val)
    gals_test = np.random.choice(remaining_gals, size=ntest, replace=False)
    print(f"Selected {len(gals_train)} galaxies for training, {len(gals_val)} for validation, and {len(gals_test)} for testing.")

    # Create dataframe with all the selected galaxy_ids, labelling the dust mixture for each one
    all_df = []
    for dust_mixture in all_dust_mixtures:
        fname = os.path.join(args.in_data_dir, f'merged_bestfit_galprops_{dust_mixture}.dat')

        # Only load the galaxy_id column to find the rows corresponding to the selected galaxy_ids
        df = pd.read_csv(fname, sep=r'\s+', usecols=["#galaxy_id"])

        # Find the rows corresponding to the selected galaxy_ids
        selected_galaxy_ids = np.concatenate([gals_train, gals_val, gals_test])
        selected_rows = df[df['#galaxy_id'].isin(selected_galaxy_ids)].index

        # Now load full data for these rows
        selected_set = set(selected_rows)
        out_lines = []
        with open(fname, "r") as fh:
            header = next(fh)
            for i, line in enumerate(fh, start=1):
                if i-1 in selected_set:
                    out_lines.append(line)
        data = "".join([header] + out_lines)
        df = pd.read_csv(io.StringIO(data), sep=r'\s+')

        # Add column for dust mixture
        df['dust_mixture'] = dust_mixture

        all_df.append(df)

    df = pd.concat(all_df, ignore_index=True)
    print(f"Number of rows in combined dataframe: {df.shape[0]}")

    # Remove any rows which contains NaN values in any column
    df.dropna(inplace=True)

    if len(common_galaxy_ids) < ntot:
        
        df_train = df[df['#galaxy_id'].isin(gals_train)]
        df_val = df[df['#galaxy_id'].isin(gals_val)]
        df_test = df[df['#galaxy_id'].isin(gals_test)]

        # Randomly sample
        df_train = df_train.sample(n=args.ntrain, random_state=args.seed)
        df_val = df_val.sample(n=args.nval, random_state=args.seed)
        df_test = df_test.sample(n=args.ntest, random_state=args.seed)

        # Put back together
        df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    else:
        print("Selecting one los for each galaxy.")

        # For each galaxy_id, randomly select one row/dust mixture and store the final row numbers
        selected_rows = (
            df.groupby('#galaxy_id', group_keys=False)
            .sample(n=1, random_state=args.seed)
            .index
        )
        df = df.loc[selected_rows]

    df.rename(columns={'#galaxy_id': 'galaxy_id'}, inplace=True)

    # Only keep the columns we need for fitting
    if 'gal_props' in args.in_param:
        ignore_pattern = r'^(galaxy_id|los|dust_mixture|Av|c\d+|B_2p|delta_2p|B_\d+|B_\d+s)$'
        cols_to_keep = [col for col in df.columns if not re.match(ignore_pattern, col)]
        cols_to_keep += [col for col in args.in_param if col != 'gal_props']

        # Remove duplicates while preserving order
        seen = set()
        cols_to_keep = [col for col in cols_to_keep if not (col in seen or seen.add(col))]

    else:
        cols_to_keep = args.in_param

    # Remove column if equal to target name
    cols_to_keep = [col for col in cols_to_keep if col != args.target_name]
    cols_to_keep = ['galaxy_id', 'los', 'dust_mixture'] + cols_to_keep + [args.target_name]
    df = df[cols_to_keep]

    df_train = df[df['galaxy_id'].isin(gals_train)]
    df_val = df[df['galaxy_id'].isin(gals_val)]
    df_test = df[df['galaxy_id'].isin(gals_test)]

    print(f"\nNumber of rows in training set: {df_train.shape[0]}")
    print(f"Number of rows in validation set: {df_val.shape[0]}")
    print(f"Number of rows in test set: {df_test.shape[0]}")

    print(f"\nSaving training set to {args.train_file}")
    df_train.to_csv(args.train_file, sep='\t', index=False)
    print(f"Saving validation set to {args.val_file}")
    df_val.to_csv(args.val_file, sep='\t', index=False)
    print(f"Saving test set to {args.test_file}")
    df_test.to_csv(args.test_file, sep='\t', index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get data with a specified config file.")
    parser.add_argument("config_path", help="Path to the configuration file.")
    args = parser.parse_args()
    get_data(args.config_path, overwrite=False)
    
