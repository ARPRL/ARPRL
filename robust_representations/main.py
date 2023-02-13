from argparse import ArgumentParser
import os

import pandas as pd
import torch
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
    #datasets = get_loan_datasets()
    datasets = get_income_datasets()
    #datasets = get_toy_datasets()
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
    test_input = test_data[0][0]
    test_target = test_data[0][1]
    for j in range(5, 7):
        test_input = np.concatenate([test_input, test_data[j][0]], axis=0)
        test_target = np.concatenate([test_target, test_data[j][1]], axis=0)
    input = test_input
    target = test_target
    private_labels = get_private_label(input, target)
    input = ch.tensor(input, device='cuda:0', dtype=ch.float)
    target = ch.tensor(target, device='cuda:0', dtype=ch.long)
    private_labels = ch.tensor(private_labels, device='cuda:0')
    attacker_model.attacker.model.draw(input, private_labels, target)

def get_loan_datasets():
    np.random.seed(42)
    dataset = pd.read_csv('mortgageloans.csv')
    print(dataset.nunique())
    data = dataset.to_numpy()
    input = data[:, 0:9]
    target = data[:, 9]
    p = np.random.permutation(len(target))
    input = input[p]
    target = target[p]
    train_data = []
    test_data = []
    for i in range(300):
        train_data.append([input[i * 100:(i + 1) * 100], target[i * 100:(i + 1) * 100]])
    for j in range(300, 370):
        test_data.append([input[(j * 100):min(len(target)-1, ((j + 1) * 100))],
                          target[(j * 100):min(len(target)-1, ((j + 1) * 100))]])
    datasets = (train_data, test_data)
    print('train private label: ', 1-np.mean(get_private_label(input[0:30000], target[0:30000])))
    print('test private label: ', 1-np.mean(get_private_label(input[30000:36982], target[30000:36982])))
    print('train targets: ', sum(target[0:30000] == 0)/30000, sum(target[0:30000] == 1)/30000, sum(target[0:30000] == 2)/30000)
    print('test targets: ', sum(target[30000:36982] == 0)/6982, sum(target[30000:36982] == 1)/6982, sum(target[30000:36982] == 2)/6982)
    return datasets

def get_income_datasets():
    np.random.seed(42)
    dataset = pd.read_csv('adult_income.csv')
    print(dataset.nunique())
    data = dataset.to_numpy()
    input = data[:, 0:13]
    input = np.delete(input, 2, axis=1)
    target = data[:, 13]
    private_labels = get_private_label(input, target)
    p = np.random.permutation(len(target))
    input = input[p]
    target = target[p]
    private_labels = private_labels[p]
    distinct_private_labels = 7 #5 for race 2 for gender
    inputs0 = []
    inputs1 = []
    # for i in range(distinct_private_labels):
    #     print('target=0, private label = ', i, ', data count: ', len((input[np.logical_and(target == 0, private_labels == i)])))
    #     print('target=1, private label = ', i, ', data count: ', len((input[np.logical_and(target == 1, private_labels == i)])))
    #     inputs0.append(input[np.logical_and(target == 0, private_labels == i)])
    #     inputs1.append(input[np.logical_and(target == 1, private_labels == i)])
    print(len(input[target == 0]), len(input[target == 1]))
    input0 = input[target == 0]
    input1 = input[target == 1]
    target0 = target[target == 0]
    target1 = target[target == 1]
    input = np.concatenate([input0[0:8000], input1], axis=0)
    # input = np.concatenate([input, input2], axis=0)
    # input = np.concatenate([input, input3[0:2000]], axis=0)
    target = np.concatenate([target0[0:8000], target1], axis=0)
    # target = np.concatenate([target, target2], axis=0)
    # target = np.concatenate([target, target3[0:2000]], axis=0)
    p = np.random.permutation(len(target))
    input = input[p]
    target = target[p]
    train_data = []
    test_data = []
    input = input.astype(float)
    print('train private label: ', sum(get_private_label(input[0:12600]) == 0)/12600, ' ',
          sum(get_private_label(input[0:12600]) == 1)/12600, ' ',
          sum(get_private_label(input[0:12600]) == 2)/12600, ' ',
          sum(get_private_label(input[0:12600]) == 3)/12600, ' ',
          sum(get_private_label(input[0:12600]) == 4)/12600, ' ',
          sum(get_private_label(input[0:12600]) == 5)/12600, ' ',
          sum(get_private_label(input[0:12600]) == 6)/12600)
    print('test private label: ', sum(get_private_label(input[12600:15841]) == 0)/3241, ' ',
          sum(get_private_label(input[12600:15841]) == 1)/3241, ' ',
          sum(get_private_label(input[12600:15841]) == 2)/3241, ' ',
          sum(get_private_label(input[12600:15841]) == 3)/3241, ' ',
          sum(get_private_label(input[12600:15841]) == 4)/3241, ' ',
          sum(get_private_label(input[12600:15841]) == 5)/3241, ' ',
          sum(get_private_label(input[12600:15841]) == 6)/3241)
    print('train targets: ', sum(target[0:12600] == 0)/12600, sum(target[0:12600] == 1)/12600)
    print('test targets: ', sum(target[12600:15841] == 0)/3241, sum(target[12600:15841] == 1)/3241)
    #norm_arr = np.sum(np.abs(input) ** 2, axis=1) ** (1. / 2)
    for i in range(12):
        input[:, i] = normalize(input[:, i], 0, 6)
    for i in range(126):
        train_data.append([input[i * 100:(i + 1) * 100], target[i * 100:(i + 1) * 100]])
    for j in range(126, 159):
        test_data.append([input[(j * 100):min(len(target)-1, ((j + 1) * 100))],
                          target[(j * 100):min(len(target)-1, ((j + 1) * 100))]])
    datasets = (train_data, test_data)
    return datasets

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr
def get_toy_datasets():
    np.random.seed(42)
    group1_means = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    group1_sds = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    group2_means = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    group2_sds = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    group1 = group1_means[0] + np.random.randn(5000) * group1_sds[0]
    group2 = group2_means[0] + np.random.randn(5000) * group2_sds[0]
    for i in range(14):
        group1 = np.c_[group1, group1_means[i + 1] + np.random.randn(5000) * group1_sds[i + 1]]
        group2 = np.c_[group2, group2_means[i + 1] + np.random.randn(5000) * group2_sds[i + 1]]
    group1_target = np.zeros(5000)
    group2_target = np.ones(5000)
    dataset = np.append(group1, group2, 0)
    target = np.append(group1_target, group2_target, 0)
    p = np.random.permutation(10000)
    dataset = dataset[p]
    target = target[p]
    train_dataset = dataset[0:9000]
    train_target = target[0:9000]
    test_dataset = dataset[9000:10000]
    test_target = target[9000:10000]
    train_data = []
    test_data = []
    for i in range(90):
        train_data.append([train_dataset[i * 100:(i + 1) * 100], train_target[i * 100:(i + 1) * 100]])
    for i in range(10):
        test_data.append([test_dataset[i * 100:(i + 1) * 100], test_target[i * 100:(i + 1) * 100]])
    datasets = (train_data, test_data)
    private_labels = get_private_label(train_dataset, train_target)
    print(np.mean(private_labels))
    return datasets

def get_private_label(input, target=None):
    private_labels = input[:, 3] #7 for gender 3 for marital in income change train_unsup as well
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


