'''Losses for training basic GANs.

Most of this was taken out of the f-GAN paper. WGAN (IPM-style) is also supported.

'''

import math

import torch
import torch.nn.functional as F

from functions.misc import log_sum_exp


def raise_measure_error(measure):
    supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    raise NotImplementedError(
        'Measure `{}` not supported. Supported: {}'.format(measure,
                                                           supported_measures))


def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.

    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)  # Note JSD will be shifted
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.

    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2  # Note JSD will be shifted
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples - 1.)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)

    if average:
        return Eq.mean()
    else:
        return Eq


def generator_loss(q_samples, measure, loss_type=None):
    """Computes the loss for the generator of a GAN.

    Args:
        q_samples: fake samples.
        measure: Measure to compute loss for.
        loss_type: Type of loss: basic `minimax` or `non-saturating`.

    """
    if not loss_type or loss_type == 'minimax':
        return get_negative_expectation(q_samples, measure)
    elif loss_type == 'non-saturating':
        return -get_positive_expectation(q_samples, measure)
    else:
        raise NotImplementedError(
            'Generator loss type `{}` not supported. '
            'Supported: [None, non-saturating, boundary-seek]')