import configparser
import ast

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
        
        if verbose: print(f"\nReading from configuration file: {ini_file}")
        config = configparser.ConfigParser()
        config.read(ini_file)
        
        self.input_file = config['system']['input_file']
        self.data_dir = config['system']['data_dir']
        self.fit_dir = config['system']['fit_dir']
        self.version_num = int(config['system']['version_num'])
        
        self.in_param = config['data']['in_param']
        self.seed = int(config['data']['seed'])
        self.ntrain = int(config['data']['ntrain'])
        self.nval = int(config['data']['nval'])
        self.ntest = int(config['data']['ntest'])
        self.lambda_V = float(config['data']['lambda_v'])
        
        self.allowed_symbols = config['operon']['allowed_symbols']
        self.epsilon = float(config['operon']['epsilon'])
        self.max_length = int(config['operon']['max_length'])
        self.time_limit = int(float(config['operon']['time_limit']))
        self.objectives = list(ast.literal_eval(config['operon']['objectives']))
        self.max_evaluations = int(float(config['operon']['max_evaluations']))
        self.generations = int(float(config['operon']['generations']))