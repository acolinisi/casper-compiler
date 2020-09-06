import pandas as pd
import numpy as np
import argparse
import os

from trainer import train

parser = argparse.ArgumentParser(
        description='Train performance models from profiling data')
parser.add_argument('prof_data_file',
        help='Input data file with profiling measurements in CSV format')
parser.add_argument('model_dir',
        help='Name of output directory where to save models for each task')
parser.add_argument('--legacy-format', action='store_true',
        help='Treat input data as from legacy experiments (single kernel)')
parser.add_argument('--test-set-fraction', type=float, default=0.1,
        help='Fraction of datapoints to use as the test set')
parser.add_argument('--layers', type=int, default=2,
        help='Number of layers in the neural network')
parser.add_argument('--features', type=int, required=True,
        help='Number of features')
parser.add_argument('--steps', type=int, default=10000,
        help='Number of steps')
parser.add_argument('--tolerance', type=float, default=0.01,
        help='Tolerance for optimizer')
args = parser.parse_args()

header_setting = 0 if not args.legacy_format != 0 else None
prof_data = pd.read_csv(args.prof_data_file, header=header_setting)
print(prof_data)

if args.legacy_format:
    prof_data.columns = ['elapsed_s', 'h', 'w', 'p1', 'p2', 'p3', 'p4', 'dummy']
    task_col = pd.DataFrame(dict(task=['task'] * len(prof_data)))
    prof_data = pd.concat([task_col, prof_data], axis=1)
    print(prof_data)

for task, data in prof_data.groupby('task'):
    data = data[[c for c in data.columns if c != 'task']]
    data = np.array(data)

    # TODO: for non-legacy format, save variant params and join dataframes here
    if not args.legacy_format:
            sys.exit(0) # placeholder, let the build succeed

    task_model_dir = os.path.join(args.model_dir, task)
    os.makedirs(task_model_dir)

    train(task_model_dir, data,
            layers=args.layers, test_set_fraction=args.test_set_fraction,
            features=args.features, steps=args.steps, tolerance=args.tolerance)
