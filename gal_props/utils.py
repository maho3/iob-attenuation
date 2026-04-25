import configparser
import ast
import os
from os.path import join as pjoin

class SelectionArgs(object):

    def __init__(self, ini_file, target_name, verbose=True):

        self.ini_file = ini_file
        self.target_name = target_name
        if verbose: 
            print(f"\nReading from selection file: {ini_file}")
        if not os.path.isfile(ini_file):
            raise FileNotFoundError(f"Selection file '{ini_file}' not found.")
        config = configparser.ConfigParser()
        config.read(ini_file)

        self.version_num = int(config['system']['version_num'])

        # Work out what params to get. It is either 'all' or a list of names
        self.in_param = config['data']['in_param']
        if ',' in self.in_param:
            self.in_param = [s.strip() for s in self.in_param.split(',')]
        else:
            self.in_param = [self.in_param.strip()]

        # Get dust mixture. Either a string or a list of strings.
        # If a list, we consider all the mixtures in the list for our target
        self.dust_mixture = config['data']['dust_mixture']
        if ',' in self.dust_mixture:
            self.dust_mixture = [s.strip().upper() for s in self.dust_mixture.split(',')]
            if 'STELLAR' in self.dust_mixture:
                self.dust_mixture[self.dust_mixture.index('STELLAR')] = 'stellar'

        # Get training, validation and test set sizes
        self.seed = int(config['data']['seed'])
        self.ntrain = int(config['data']['ntrain'])
        self.nval = int(config['data']['nval'])
        self.ntest = int(config['data']['ntest'])
        self.train_los_max = int(config['data']['train_los_max'])

        self.in_data_dir = config['system']['in_data_dir']
        self.out_data_dir = pjoin(config['system']['out_data_dir'], self.target_name)
        os.makedirs(self.out_data_dir, exist_ok=True)

        self.train_file = pjoin(self.out_data_dir, f'{self.target_name}_train_v{self.version_num}.txt')
        self.val_file = pjoin(self.out_data_dir, f'{self.target_name}_val_v{self.version_num}.txt')
        self.test_file = pjoin(self.out_data_dir, f'{self.target_name}_test_v{self.version_num}.txt')

class OperonArgs(object):

    def __init__(self, ini_file, verbose=True):
        
        self.ini_file = ini_file
        if verbose: 
            print(f"\nReading from configuration file: {ini_file}")
        if not os.path.isfile(ini_file):
            raise FileNotFoundError(f"Configuration file '{ini_file}' not found.")
        config = configparser.ConfigParser()
        config.read(ini_file)

        self.target_name = config['data']['target_name']
        self.selection_file = config['data']['selection_file']
        self.selection = SelectionArgs(self.selection_file, 
                                       self.target_name)
        self.seed = self.selection.seed
        self.dust_mixture = self.selection.dust_mixture
        self.ntrain = self.selection.ntrain
        self.nval = self.selection.nval
        self.ntest = self.selection.ntest
        self.train_los_max = self.selection.train_los_max
        self.in_param = self.selection.in_param

        # Get the file names for the training, validation and test sets
        self.fit_dir = config['system']['fit_dir']
        self.in_data_dir = self.selection.in_data_dir
        self.out_data_dir = self.selection.out_data_dir
        self.version_num = int(config['system']['version_num'])
        self.train_file = self.selection.train_file
        self.val_file = self.selection.val_file
        self.test_file = self.selection.test_file
        
        # Operon arguments
        self.allowed_symbols = config['operon']['allowed_symbols']
        self.epsilon = float(config['operon']['epsilon'])
        self.max_length = int(config['operon']['max_length'])
        self.time_limit = int(float(config['operon']['time_limit']))
        self.objectives = list(ast.literal_eval(config['operon']['objectives']))
        self.max_evaluations = int(float(config['operon']['max_evaluations']))
        self.generations = int(float(config['operon']['generations']))
        