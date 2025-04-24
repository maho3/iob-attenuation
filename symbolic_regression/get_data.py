import numpy as np
from utils import OperonArgs
from os.path import join as pjoin
import os

def format_output(mydata, wavelength, ipar, iwave, A_logged):

    n = mydata.shape[0]

    # Flatten attentuation curve
    data = mydata[:,np.array(iwave, dtype=int)]
    data = data.ravel()
    if not A_logged:
        data = np.log10(data)

    # Repeat the parameter values
    samples = np.repeat(mydata[:,np.array(ipar)], len(wavelength), axis=0)
    lam = np.tile(wavelength, n)

    output = np.column_stack((samples, lam, data))

    return output

def get_data(ini_file):

    args = OperonArgs(ini_file)

    print('\nSplitting data into training, validation, and test sets...')
    print('File:', args.input_file)

    # Read all galaxy ids
    with open(args.input_file, 'r') as f:
        header = f.readline().split()
    data = np.loadtxt(args.input_file, skiprows=1)

    # Check lambda_V from the file is as expected
    if f'logA_{int(args.lambda_V)}A' in header:
        assert np.all(data[:, header.index(f'logA_{int(args.lambda_V)}A')] == 0)
    elif f'A_{int(args.lambda_V)}A' in header:
        assert np.all(data[:, header.index(f'A_{int(args.lambda_V)}A')] == 1)
    else:
        raise ValueError("Column with lambda_V not found in input file")

    # Shuffle the galaxy ids
    galaxy_ids = np.unique(data[:, header.index('galaxy_id')])
    np.random.seed(args.seed)
    np.random.shuffle(galaxy_ids)

    # Split the galaxy ids into train, validation, and test sets
    n = len(galaxy_ids)

    print('Number of curves:', len(data))
    print('Number of unique galaxy ids:', n)

    # Estimate number of unique IDs needed
    ntrain = int(args.ntrain / (args.ntrain + args.nval + args.ntest) * len(galaxy_ids))
    nval= int(args.nval / (args.ntrain + args.nval + args.ntest) * len(galaxy_ids))
    ntest = len(galaxy_ids) - nval - ntrain

    train_ids = galaxy_ids[:ntrain]
    val_ids = galaxy_ids[ntrain:ntrain+nval]
    test_ids = galaxy_ids[ntrain+nval:ntrain+nval+ntest]

    # Split data
    m = np.isin(data[:, header.index('galaxy_id')], train_ids)
    train_data = data[m]
    m = np.isin(data[:, header.index('galaxy_id')], val_ids)
    val_data = data[m]
    m = np.isin(data[:, header.index('galaxy_id')], test_ids)
    test_data = data[m]

    # Each galaxy can have more than one los, so we have too many points now
    # We again shuffle the data and reduce the number of objects
    train_data = train_data[np.random.permutation(train_data.shape[0])[:args.ntrain], :]
    val_data = val_data[np.random.permutation(val_data.shape[0])[:args.nval], :]
    test_data = test_data[np.random.permutation(test_data.shape[0])[:args.ntest], :]
    
    print('Number of training curves:', train_data.shape[0])
    print('Number of validation curves:', val_data.shape[0])
    print('Number of test curves:', test_data.shape[0])

    # Convert each lambda into lambda / lambda_V
    iwave = [(i, float(h[2:-1])) for i, h in enumerate(header) if h.startswith('A_') and h.endswith('A') and h[2:-1].isdigit()] 
    A_logged = False
    if len(iwave) == 0:
        A_logged = True
        iwave = [(i, float(h[5:-1])) for i, h in enumerate(header) if h.startswith('logA_') and h.endswith('A') and h[5:-1].isdigit()] 
    assert len(iwave) > 0, "Could not read the wavelengths"
    wavelength = np.array([h[1] for h in iwave], dtype=float) / args.lambda_V
    iwave = np.array([h[0] for h in iwave], dtype=int)
    
    # Find the indices which contain the parameters of interest
    start = args.in_param.upper()
    ipar = [i for i, h in enumerate(header) if h.startswith(start) and h[len(start):].isdigit()]
    print(f'Number of {start} parameters found: {len(ipar)}')
    
    # Add to ipar the other parts of the header we need
    ipar = [header.index(h) for h in ['galaxy_id', 'los']] + ipar
    
    new_header = ' '.join([header[i] for i in ipar] + ['lam', 'log10A'])

    dirname = pjoin(args.data_dir, f'{args.in_param}_data_{args.version_num}')
    os.makedirs(dirname, exist_ok=True)

    for name, mydata in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
        output = format_output(mydata, wavelength, ipar, iwave, A_logged)
        outname = pjoin(dirname, f'{args.in_param}_{name}_data.txt')
        np.savetxt(outname, output, header=new_header, comments='')

    return 

if __name__ == "__main__":
    get_data('conf/iob_0.ini')