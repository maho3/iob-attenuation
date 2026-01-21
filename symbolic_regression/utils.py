import configparser
import ast
import os
from os.path import join as pjoin

class OperonArgs(object):
    """
    Class to store information from ini file to be passed to operon
    
    This object has the following attributes:
    
        :data_dir (str): The directory where the data are stored
        :fit_dir (str): The directory where the fitting results are stored
        :version_num (int): The number of the version considered
        :in_param (str): The name of the input parameters used for fitting
        :seed (int): Seed to use for shuffling when creating the training examples
        :ntrain (int): The number of training examples to use
        :nval (int): The number of validation examples to use
        :ntest (int): The number of test examples to use
        :lambda_V (float): The wavelength at which the data is normalised
        :allowed_symbols (str): Comma-separated string of allowed operators in symbolic expressions
        :epsilon (float): The parameter determining within which tolerance two objective
            values are considered equal.
        :max_length (int): The maximum length of an expression allowed
        :time_limit (int): The maximum amount of time operon can search for [s]
        :objectives (list[str], default=['rmse', 'length']): The objectives for which operon
            will optimise during the equation search
        :max_evaluations (int, default=int(1e8)): The maximum number of evaluations operon 
            is allowed to take
        :generations (int, default=int(1e8)): The maximum number of generations operon 
            is allowed to have
    
    Args:
        :ini_file (str): The name of the file to be read by `configparser` containing the run's information
        :verbose (bool, default=True): Whether to print status
    """
    
    def __init__(self, ini_file, verbose=True):
        
        if verbose: 
            print(f"\nReading from configuration file: {ini_file}")
        if not os.path.isfile(ini_file):
            raise FileNotFoundError(f"Configuration file '{ini_file}' not found.")
        config = configparser.ConfigParser()
        config.read(ini_file)
        
        self.sr_data_dir = config['system']['sr_data_dir']
        self.fit_dir = config['system']['fit_dir']
        self.version_num = int(config['system']['version_num'])

        self.selection_conf = config['system']['selection_conf']
        self.selection = SelectionArgs(self.selection_conf, verbose=verbose)

        self.subtract_outer = bool(config['data']['subtract_outer'].strip().lower() == 'true') if 'subtract_outer' in config['data'] else False

        self.lambda_V = float(config['data']['lambda_v'])
        self.npar = int(config['data']['npar'])
        self.fit_log = bool(config['data']['fit_log'].strip().lower() == 'true')
        self.lam_max = float(config['data']['lam_max'])
        self.lam_min = float(config['data']['lam_min'])
        self.lam_bump_min = float(config['data'].get('lam_bump_min', self.lam_min))
        self.lam_bump_max = float(config['data'].get('lam_bump_max', self.lam_max))
        self.keep_region = config['data'].get('keep_region', 'all').strip().lower()
        if self.keep_region not in ['all', 'outer', 'bump']:
            raise ValueError("keep_region must be one of 'all', 'outer', or 'bump'")
        if self.lam_bump_min >= self.lam_bump_max:
            raise ValueError("lam_bump_min must be less than lam_bump_max")

        val = config['data'].get('lam_trans')
        self.lam_trans = float(val) if val is not None else None

        val = config['data'].get('f_subsample')
        self.f_subsample = int(val) if val is not None else None

        self.allowed_symbols = config['operon']['allowed_symbols']
        self.epsilon = float(config['operon']['epsilon'])
        self.max_length = int(config['operon']['max_length'])
        self.time_limit = int(float(config['operon']['time_limit']))
        self.objectives = list(ast.literal_eval(config['operon']['objectives']))
        self.max_evaluations = int(float(config['operon']['max_evaluations']))
        self.generations = int(float(config['operon']['generations']))

        # These files contain only those properties needed for symbolic regression
        self.train_sr_input_file = pjoin(self.sr_data_dir, f'train_sr_input_v{self.version_num}.csv')
        self.val_sr_input_file = pjoin(self.sr_data_dir, f'val_sr_input_v{self.version_num}.csv')
        self.test_sr_input_file = pjoin(self.sr_data_dir, f'test_sr_input_v{self.version_num}.csv')

    @property
    def ntrain(self):
        """Number of training examples from selection configuration"""
        return self.selection.ntrain
    
    @property
    def nval(self):
        """Number of validation examples from selection configuration"""
        return self.selection.nval
    
    @property
    def ntest(self):
        """Number of test examples from selection configuration"""
        return self.selection.ntest
    
    @property
    def in_data_dir(self):
        """Directory where the input data is stored"""
        return self.selection.in_data_dir
    
    @property
    def out_data_dir(self):
        """Directory where the output data is stored"""
        return self.selection.out_data_dir

    @property
    def in_param(self):
        """Input parameter for selection"""
        return self.selection.in_param


class SelectionArgs(object):
    """
    Class to store information from ini file to be passed to galaxy selection functions
    
    This object has the following attributes:
    
        :data_dir (str): The directory where the data are stored
        :version_num (int): The number of the version considered
        :in_file (str): The name of the input data file
        :exclude_file (str): The name of the file containing galaxy ids to exclude from selection
        :method (str): The method to use for galaxy selection
        :ntrain (int): The number of training galaxies to select
        :nval (int): The number of validation galaxies to select
        :ntest (int): The number of test galaxies to select
        :rng_seed (int): Seed to use for shuffling when selecting galaxies

    For the method "laura", the following additional attributes are defined:
        :min_per_bin_mult (int): Minimum number of galaxies per bin multiplier
        :nbins_max (int): Maximum number of bins
        :n_per_bin (int): Number of galaxies per bin
        :n_high (int): Number of high SFR galaxies to select
        :n_low (int): Number of low SFR galaxies to select
        :sfr_high_thresh (float): Threshold for high SFR galaxies
        :sfr_low_thresh (float): Threshold for low SFR galaxies
        :q_high (float): Quantile for high mass galaxies
        :q_low (float): Quantile for low mass galaxies

    Args:
        :ini_file (str): The name of the file to be read by `configparser` containing the run's information
        :verbose (bool, default=True): Whether to print status

    """

    def __init__(self, ini_file, verbose=True):
        
        if verbose: 
            print(f"\nReading from configuration file: {ini_file}")
        if not os.path.isfile(ini_file):
            raise FileNotFoundError(f"Configuration file '{ini_file}' not found.")
        config = configparser.ConfigParser()
        config.read(ini_file)
        
        self.in_data_dir = config['system']['in_data_dir']
        self.out_data_dir = config['system']['out_data_dir']
        self.version_num = int(config['system']['version_num'])
        self.in_file = pjoin(self.in_data_dir, config['system']['in_file'])
        self.exclude_file = pjoin(self.in_data_dir, config['system']['exclude_file'])

        self.in_param = config['data']['in_param']

        # These files contain all properties of the chosen galaxies
        self.train_file = pjoin(self.out_data_dir, f'train_data_v{self.version_num}.csv')
        self.val_file = pjoin(self.out_data_dir, f'val_data_v{self.version_num}.csv')
        self.test_file = pjoin(self.out_data_dir, f'test_data_v{self.version_num}.csv')


        self.method = config['selection']['method'].lower()
        self.ntrain = int(config['selection']['ntrain'])
        self.nval = int(config['selection']['nval'])
        self.ntest = int(config['selection']['ntest'])
        self.rng_seed = int(config['selection']['rng_seed'])

        self.remove_negative_curves = config['selection'].getboolean('remove_negative_curves')
        self.remove_peaked_curves = config['selection'].getboolean('remove_peaked_curves')
        self.peak_threshold = float(config['selection'].get('peak_threshold', 0.5))
        self.peak_lambda_min = float(config['selection'].get('peak_lambda_min', 5542))

        if self.method == 'laura':
            self.min_per_bin_mult = int(config['laura']['min_per_bin_mult'])
            self.nbins_max = int(config['laura']['nbins_max'])
            self.frac_high = float(config['laura']['frac_high'])
            self.frac_low = float(config['laura']['frac_low'])
            self.sfr_high_thresh = float(config['laura']['sfr_high_thresh'])
            self.sfr_low_thresh = float(config['laura']['sfr_low_thresh'])
            self.q_high = float(config['laura']['q_high'])
            self.q_low = float(config['laura']['q_low'])
            self.extreme_seed = int(config['laura']['extreme_seed'])

        if self.method not in ['laura', 'random']:
            raise ValueError(f"Selection method '{self.method}' not recognised. Available methods are 'laura' and 'random'.")
