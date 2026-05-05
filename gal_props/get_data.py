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
    full_Av_name = 'A_5542A'

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
        all_dust_mixtures = args.dust_mixture
    
    for dust_mixture in all_dust_mixtures:
        print('Loading data for dust mixture:', dust_mixture)
        fname = os.path.join(args.in_data_dir, f'iob_codes_plus_Acurve_galprop_{dust_mixture}_all.dat')
        df = pd.read_csv(fname, sep=r'\s+', usecols=["galaxy_id", "los"])
        galaxy_ids = df['galaxy_id'].unique()
        all_galaxy_ids.append(galaxy_ids)
        los_mask = df['los'] < args.train_los_max
        print(f"\tNumber of unique galaxy_ids: {len(galaxy_ids)}")
        print(f"\tTotal number of rows: {df.shape[0]}")
        print(f"\tNumber of rows with los < {args.train_los_max}: {los_mask.sum()}")

    # Find galaxy IDs which are in all files
    common_galaxy_ids = reduce(np.intersect1d, all_galaxy_ids)

    print(f"Number of common galaxy_ids across all dust mixtures: {len(common_galaxy_ids)}")
    print(f"Training will use los < {args.train_los_max}; val/test will use los >= {args.train_los_max}")

    # Create dataframe with all the selected galaxy_ids, labelling the dust mixture for each one
    all_df = []
    print('\nLoading full data for selected galaxy_ids and applying min_av cut:')
    for dust_mixture in all_dust_mixtures:

        fname = os.path.join(args.in_data_dir, f'iob_codes_plus_Acurve_galprop_{dust_mixture}_all.dat')
        print(fname)

        # Only load the galaxy_id column to find the rows corresponding to the selected galaxy_ids
        df = pd.read_csv(fname, sep=r'\s+', usecols=["galaxy_id"])

        # Find the rows corresponding to the selected galaxy_ids
        selected_galaxy_ids = common_galaxy_ids
        selected_rows = df[df['galaxy_id'].isin(selected_galaxy_ids)].index

        # Now load full data for these rows
        selected_set = set(selected_rows)
        out_lines = []
        with open(fname, "r") as fh:
            header = next(fh)
            # Remove any part of header between square brackets, as this is not a valid column name and causes problems for pandas
            header = re.sub(r'\[.*?\]', '', header)
            for i, line in enumerate(fh, start=1):
                if i-1 in selected_set:
                    out_lines.append(line)
        data = "".join([header] + out_lines)
        df = pd.read_csv(io.StringIO(data), sep=r'\s+')

        # Remove columns which are in ignore_gal_props
        if args.ignore_gal_props is not None:
            cols_to_drop = [col for col in args.ignore_gal_props if col in df.columns]
            print(f"\tDropping columns: {cols_to_drop}")
            df.drop(columns=cols_to_drop, inplace=True)

        # Cut curves which do not meet min_av criterion
        old_len = df.shape[0]
        mask = np.isfinite(df[full_Av_name]) & (df[full_Av_name] >= args.min_av)
        df = df[mask]
        print(f"\t{dust_mixture}: Number of rows after applying min_av >= {args.min_av} cut: {df.shape[0]} (removed {old_len - df.shape[0]} rows)")

        # Add column for dust mixture
        df['dust_mixture'] = dust_mixture

        all_df.append(df)

    df = pd.concat(all_df, ignore_index=True)
    print(f"\nNumber of rows in combined dataframe: {df.shape[0]}")

    # Only keep the columns we need for fitting
    if 'gal_props' in args.in_param:
        ignore_pattern = r'^(galaxy_id|los|dust_mixture|Av|c\d+|B_2p|delta_2p|B_\d+|B_\d+s|A_\d+A|IOB\d+)$'
        cols_to_keep = [col for col in df.columns if not re.match(ignore_pattern, col)]
        cols_to_keep += [col for col in args.in_param if col != 'gal_props']

        # Remove duplicates while preserving order
        seen = set()
        cols_to_keep = [col for col in cols_to_keep if not (col in seen or seen.add(col))]

    else:
        cols_to_keep = args.in_param

    if 'Av' in cols_to_keep:
        cols_to_keep.remove('Av')
        cols_to_keep.append(full_Av_name)

    # We will get dust_mixture anyway, so let's remove it from cols_to_keep if it's there to avoid duplication
    if 'dust_mixture' in cols_to_keep:
        cols_to_keep.remove('dust_mixture')

    if args.target_name == 'Av':
        cols_to_keep = [col for col in cols_to_keep if col != full_Av_name]
        cols_to_keep = ['galaxy_id', 'los', 'dust_mixture'] + cols_to_keep + [full_Av_name]
    else:
        # Remove column if equal to target name
        cols_to_keep = [col for col in cols_to_keep if col != args.target_name]
        cols_to_keep = ['galaxy_id', 'los', 'dust_mixture'] + cols_to_keep + [args.target_name]
    df = df[cols_to_keep]

    # Remove any rows which contains NaN values in any column
    df.dropna(inplace=True)
    print(f"Number of rows after removing NaN values: {df.shape[0]}")

    # Now rename 'A_5542A' to 'Av' if it's in the dataframe
    if full_Av_name in df.columns:
        df.rename(columns={full_Av_name: 'Av'}, inplace=True)

    # Replace inclination by sin(inclination) to make it more physical 
    # and easier to fit, and rename column accordingly
    if 'inclination_deg' in df.columns:
        df['inclination_sin'] = np.sin(np.radians(df['inclination_deg']))
        df.drop(columns=['inclination_deg'], inplace=True)
        if 'inclination_deg' in cols_to_keep:
            i = cols_to_keep.index('inclination_deg')
            cols_to_keep[i] = 'inclination_sin'

    print('All dust:', df['dust_mixture'].unique())

    # Split by los value: rows with los < train_los_max are eligible for training;
    # rows with los >= train_los_max are eligible for validation and test.
    df_low = df[df['los'] < args.train_los_max]
    df_high = df[df['los'] >= args.train_los_max]

    los_per_gal_low = df_low.shape[0] / df_low['galaxy_id'].nunique()
    los_per_gal_high = df_high.shape[0] / df_high['galaxy_id'].nunique()
    print(f"\nAverage number of los values per galaxy in low-los set: {los_per_gal_low:.2f}")
    print(f"Average number of los values per galaxy in high-los set: {los_per_gal_high:.2f}")

    gals = df['galaxy_id'].unique()
    gals = np.random.permutation(gals)

    # Split the galaxy IDs by the ntrain:nval+ntest ratio, but upweight low-los by the ratio
    # of los_per_gal_high to los_per_gal_low, to try to get a more balanced number of rows in each set.
    train_ratio = (args.ntrain / (args.nval + args.ntest + args.ntrain)) * (los_per_gal_high / (los_per_gal_low + los_per_gal_high))
    n_train_gals = int(len(gals) * train_ratio)

    # Add more galaxies to the training set from the low-los set until we have enough rows in the training pool, 
    # up to the point where we have added all low-los galaxies. This is to try to get more training data while 
    # still ensuring that all training data comes from low-los galaxies.
    gals_low = df_low['galaxy_id'].unique()
    gals_low = np.random.permutation(gals_low)
    gals_train = gals_low[:n_train_gals]
    df_train_pool = df_low[df_low['galaxy_id'].isin(gals_train)]
    while df_train_pool.shape[0] < args.ntrain and df_train_pool.shape[0] < df_low.shape[0]:
        next_gal = gals_low[n_train_gals]
        gals_train = np.append(gals_train, next_gal)
        df_train_pool = df_low[df_low['galaxy_id'].isin(gals_train)]
        n_train_gals += 1

    # Remove gals_train from the full list of galaxies to get the remaining galaxies for val/test
    remaining_gals = [gal for gal in gals if gal not in gals_train]
    remaining_gals = np.random.permutation(remaining_gals)
    n_val_gals = int(len(remaining_gals) * (args.nval / (args.nval + args.ntest)))
    n_test_gals = len(remaining_gals) - n_val_gals
    gals_val = remaining_gals[:n_val_gals]
    gals_test = remaining_gals[n_val_gals:n_val_gals + n_test_gals]

    print(f"\nNumber of unique galaxy_ids allocated to training set: {len(gals_train)}")
    print(f"Number of unique galaxy_ids allocated to validation set: {len(gals_val)}")
    print(f"Number of unique galaxy_ids allocated to test set: {len(gals_test)}")

    df_train_pool = df_low[df_low['galaxy_id'].isin(gals_train)]
    df_val_pool = df_high[df_high['galaxy_id'].isin(gals_val)]
    df_test_pool = df_high[df_high['galaxy_id'].isin(gals_test)]

    df_train = df_train_pool.sample(n=min(args.ntrain, len(df_train_pool)), random_state=args.seed)
    df_val = df_val_pool.sample(n=min(args.nval, len(df_val_pool)), random_state=args.seed)
    df_test = df_test_pool.sample(n=min(args.ntest, len(df_test_pool)), random_state=args.seed)

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
    # parser = argparse.ArgumentParser(description="Get data with a specified config file.")
    # parser.add_argument("config_path", help="Path to the configuration file.")
    # args = parser.parse_args()
    # get_data(args.config_path, overwrite=True)
    all_config = os.listdir('conf')
    all_config = [f for f in all_config if not f.startswith('selection_') and f.endswith('.ini')]
    # all_config = ['Av_4.ini', 'B1_3.ini', 'B3_3.ini']
    # all_config = ['B0_12.ini']
    for config in all_config:
        get_data(os.path.join('conf', config), overwrite=False)
    
