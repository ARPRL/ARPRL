from argparse import ArgumentParser
import os

import pandas as pd
from cox import utils
import cox.store

from robustness.datasets import DATASETS
from robustness.defaults import check_and_fill_args
from robustness.tools import helpers
from robustness import defaults
from unsup_models import make_and_restore_model as make_and_restore_model_unsup
from train_unsup import train_model, eval_model
import time
import torch as ch
import numpy as np
import matplotlib.pyplot as plt


parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)
parser.add_argument('--no-store', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)

parser.add_argument('--task',
                    choices=['estimate-mi',
                             'train-model',
                             'train-encoder',
                             'train-privacy',
                             'train-privacy_classifier',
                             'train-all',
                             'train-classifier'])
parser.add_argument('--representation-type',
                    choices=['layer',
                             'neuron-asbatch',
                             'neuron-crossproduct'],
                    default='layer')
parser.add_argument('--estimator-loss', choices=['normal', 'worst'],
                    default='normal')
parser.add_argument('--classifier-loss', choices=['standard', 'robust', 'privacy', 'all'])
parser.add_argument('--robust_rounds', type=int, default=20)
parser.add_argument('--privacy_cla', type=int, default=20)
parser.add_argument('--privacy_rounds', type=int, default=20)
parser.add_argument('--utility_rounds', type=int, default=20)
parser.add_argument('--all_rounds', type=int, default=5)
parser.add_argument('--alpha', type=float, default=0.4)
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument('--lam', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--classifier-arch', choices=['mlp', 'linear', 'mi'])
parser.add_argument('--share-arch', action='store_true', default=False)

parser.add_argument('--va-mode', choices=['nce', 'fd', 'dv'], default='dv')
parser.add_argument('--va-fd-measure', default='JSD')
parser.add_argument('--va-hsize', type=int, default=2048)

args = parser.parse_args()

# torch.manual_seed(41)
# torch.cuda.manual_seed(41)
# np.random.seed(41)
# random.seed(41)
# torch.backends.cudnn.deterministic=True


def main(args):
    starting_time = time.time()
    datasets = get_toy_datasets()
    store = None if args.no_store else setup_store_with_metadata(args)

    if args.debug:
        args.workers = 0

    attacker_model, checkpoint = make_and_restore_model_unsup(args=args,
                                                              dataset=datasets)
    #attacker_model.load_state_dict(ch.load('D:/path/to/model/robust.atk'))
    if args.eval_only:
        eval_model(args, attacker_model, datasets,
                   checkpoint=checkpoint, store=store)
    else:
        train_model(args, attacker_model, datasets,
                    checkpoint=checkpoint, store=store)


    end_time = time.time()
    total_time = end_time - starting_time
    print('Total Time: ', total_time)
    train_data, test_data = datasets
    train_input = train_data[0][0]
    train_target = train_data[0][1]
    test_input = test_data[0][0]
    test_target = test_data[0][1]
    for j in range(5, 7):
        train_input = np.concatenate([train_input, train_data[j][0]], axis=0)
        train_target = np.concatenate([train_target, train_data[j][1]], axis=0)
    input = train_input
    target = train_target
    private_labels = get_private_label(input, target)
    input = ch.tensor(input, device='cuda:0', dtype=ch.float)
    target = ch.tensor(target, device='cuda:0', dtype=ch.long)
    private_labels = ch.tensor(private_labels, device='cuda:0')
    attacker_model.attacker.model.draw(input, private_labels, target)

def get_color(private_labels):
    color = []
    for i in private_labels:
        if i == 0:
            color.append('blue')
        else:
            color.append('red')
    return color
def get_toy_datasets():
    np.random.seed(42)
    group1_means = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    group1_sds = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    group2_means = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    group2_sds = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    #group1 = group1_means[0] + np.random.randn(5000) * group1_sds[0]
    #group2 = group2_means[0] + np.random.randn(5000) * group2_sds[0]
    group1 = np.random.uniform(0, 1, 500) - 0.5
    group2 = np.random.uniform(0, 1, 500) + 0.5
    for i in range(1):
        group1 = np.c_[group1, group1_means[i + 1] + np.random.randn(500) * group1_sds[i + 1]]
        group2 = np.c_[group2, group2_means[i + 1] + np.random.randn(500) * group2_sds[i + 1]]
    for i in range(len(group1)):
        dat = group1[i]
        dat[1] = ((0.25 - (dat[0] ** 2)) ** 0.5) * np.sign(dat[1])
        group1[i] = dat
        dat = group2[i]
        dat[1] = ((0.25 - ((dat[0]-1) ** 2)) ** 0.5) * np.sign(dat[1]-2)
        group2[i] = dat
    labels1 = get_private_label(group1)
    labels2 = get_private_label(group2)
    color1 = get_color(labels1)
    color2 = get_color(labels2)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(group1[:, 0], group1[:, 1], c=color1, marker='x', alpha=0.5)
    ax.scatter(group2[:, 0], group2[:, 1], c=color2, marker='.', alpha=0.5)
    #plt.show()
    group1_target = np.zeros(500)
    group2_target = np.ones(500)
    dataset = np.append(group1, group2, 0)
    target = np.append(group1_target, group2_target, 0)
    p = np.random.permutation(1000)
    dataset = dataset[p]
    target = target[p]
    train_dataset = dataset[0:900]
    train_target = target[0:900]
    test_dataset = dataset[900:1000]
    test_target = target[900:1000]
    train_data = []
    test_data = []
    for i in range(9):
        train_data.append([train_dataset[i * 100:(i + 1) * 100], train_target[i * 100:(i + 1) * 100]])
    for i in range(1):
        test_data.append([test_dataset[i * 100:(i + 1) * 100], test_target[i * 100:(i + 1) * 100]])
    datasets = (train_data, test_data)
    private_labels = get_private_label(train_dataset, train_target)
    print(np.mean(private_labels))

    return datasets

def get_private_label(input, target=None):
    private_labels = []
    for i in range(len(input)):
        #Y = 0 if target[i] == 0 else 2
        X = input[i]
        if X[1] > 0:
            private_labels.append(0)
        else:
            private_labels.append(1)
        # privacy = get_distance(X, Y)
        # private_label = 0 if privacy < 1.9 else 1
        # private_labels.append(private_label)
    return private_labels

def get_distance(p, q):
    s_sq_difference = 0
    for p_i, q_i in zip(p, q):
        s_sq_difference += (p_i - q_i) ** 2
    distance = s_sq_difference ** 0.5
    return distance

def setup_args(args):
    '''
    Fill the args object with reasonable defaults from
    :mod:`robustness.defaults`, and also perform a sanity check to make sure no
    args are missing.
    '''
    # override non-None values with optional config_path
    args.adv_train = (args.classifier_loss == 'robust') or \
                     (args.estimator_loss == 'worst')
    if args.config_path:
        args = cox.utils.override_json(args, args.config_path)

    ds_class = DATASETS[args.dataset]
    args = check_and_fill_args(args, defaults.CONFIG_ARGS, ds_class)

    if not args.eval_only:
        args = check_and_fill_args(args, defaults.TRAINING_ARGS, ds_class)

    if args.adv_train or args.adv_eval:
        args = check_and_fill_args(args, defaults.PGD_ARGS, ds_class)

    args = check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, ds_class)
    if args.eval_only: assert args.resume is not None, \
            "Must provide a resume path if only evaluating"
    return args


def setup_store_with_metadata(args):
    '''
    Sets up a store for training according to the arguments object. See the
    argparse object above for options.
    '''
    # Create the store
    store = cox.store.Store(args.out_dir, args.exp_name)
    args_dict = args.__dict__
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)
    return store


if __name__ == "__main__":
    args = cox.utils.Parameters(args.__dict__)
    args = setup_args(args)

    args.workers = 0 if args.debug else args.workers
    print(args)

    main(args)


