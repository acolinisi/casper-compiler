import pandas as pd
import numpy as np
import argparse
import os

from trainer import train

parser = argparse.ArgumentParser(
        description='Train performance models from profiling data')
parser.add_argument('samples_file',
        help='Input data file with sampled variants in CSV format')
parser.add_argument('measurements_file',
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
measurements = pd.read_csv(args.measurements_file, header=header_setting)
print(measurements)

samples = pd.read_csv(args.samples_file)
print(samples)

if args.legacy_format:
    measurements.columns = ['elapsed_s', 'h', 'w', 'p1', 'p2', 'p3', 'p4', 'dummy']
    task_col = pd.DataFrame(dict(task=['task'] * len(measurements)))
    measurements = pd.concat([task_col, measurements], axis=1)
    print(measurements)

for task, task_meas in measurements.groupby('task'):

    if not args.legacy_format:
            task_samples = samples[samples["task"] == task]

            # Create the data frame for the data to feed train(), by joining
            # the samples dataset (param values) with the measurements dataset
            # (elapsed times).

            # TODO: the following is extremely fragile. The interface into
            # trainer needs to be changed to take in a key-value map of params,
            # instead of relying on an ordered list of values, or at least take
            # the order (list of param names) as an argument as well.

            # First, pick any variant, just to get param count
            any_variant = task_meas.iloc[0]['variant']
            variant_params = \
                task_samples[task_samples['variant'] == any_variant]
            columns = ['elapsed_s'] + list(variant_params['param'])
            prof_df = pd.DataFrame(columns=columns)

            for i, row in task_meas.iterrows():
                variant_params = \
                        task_samples[task_samples['variant'] == row['variant']]
                row_dict = dict(elapsed_s=row['elapsed_s'])
                for j, prow in variant_params[['param', 'value']].iterrows():
                    row_dict[prow['param']] = prow['value']
                prof_df = pd.concat([prof_df, pd.DataFrame(row_dict, index=[0])],
                    axis=0, ignore_index=True)
    else: # legacy format (no joining, both samples and times are in one file)
        prof_df = measurements[[c for c in data.columns if c != 'task']]

    task_model_dir = os.path.join(args.model_dir, task)
    if not os.path.exists(task_model_dir) or not os.path.isdir(task_model_dir):
        os.makedirs(task_model_dir)

    print(prof_df)
    prof_data = np.array(prof_df)
    train(task_model_dir, prof_data,
            layers=args.layers, test_set_fraction=args.test_set_fraction,
            features=args.features, steps=args.steps, tolerance=args.tolerance)

    # Contract with CMake script (a file artifact to track generated directory)
    open(os.path.join(args.model_dir, "timestamp"), "w").close()
