import numpy as np
import torch


def generate_polynomial_1D(n_pts: int = 5000, domain: tuple = (-4.5, 4.5),
                           scale: float = 80.0, noise_level: float = 15.0, power: int = 2,
                           grid: bool = False):
    """
    Generates a 1D polynomial function with noise.
    :param n_pts: number of points
    :param domain: domain of the function
    :return: x, y
    """
    if grid:
        x = torch.linspace(domain[0], domain[1], n_pts)
    else:
        x = torch.rand(n_pts) * (domain[1] - domain[0]) + domain[0]

    y = scale * torch.pow(x, power) + noise_level * torch.rand(x.size(-1))  # f(x)

    return x.reshape(-1, 1), y.reshape(-1, 1)


def generate_cos_polynomial_1D(n_pts: int = 5000, domain: tuple = (-4.5, 4.5),
                               scale: float = 80.0, noise_level: float = 15.0, power: int = 2,
                               grid: bool = False):
    """
    Generates a 1D polynomial function with noise.
    :param n_pts: number of points
    :param domain: domain of the function
    :return: x, y
    """
    if grid:
        x = torch.linspace(domain[0], domain[1], n_pts)
    else:
        x = torch.rand(n_pts) * (domain[1] - domain[0]) + domain[0]

    y = scale * torch.pow(torch.cos(x), power) + noise_level * torch.rand(x.size(-1))  # f(x)

    # apply mask
    # y = y[x.abs() > 1.0]
    # x = x[x.abs() > 1.0]

    return x.reshape(-1, 1), y.reshape(-1, 1)


def generate_mask(x, y, region: str = 'middle', cutoff: float = 0.5, proportion: float = 0.5):
    """
    Generates a mask for a given region.
    :param x:
    :param y:
    :param region: region to mask
    :param cutoff:
    :param proportion: proportion of points to mask in region
    :return:
    """

    x_mean = x.mean()
    # x_min = x.min()
    # x_max = x.max()

    idx = 1 * (x.abs() > cutoff)
    idx_no_mask = idx.nonzero(as_tuple=True)[0]
    idx_mask = (1 - idx).nonzero(as_tuple=True)[0]
    idx_mask = idx_mask[torch.randperm(len(idx_mask))[:int((1 - proportion) * len(idx_mask))]]

    idx = torch.cat((idx_no_mask, idx_mask))
    idx = idx[torch.randperm(len(idx))]
    y = y[idx]
    x = x[idx]

    # if region == 'middle':

    # elif region == 'left':
    #     y = y[x > cutoff]
    #     x = x[x > cutoff]
    # elif region == 'right':
    #     y = y[x < cutoff]
    #     x = x[x < cutoff]

    return x, y


def combine(limit_matrix, n, gap, s, grid=False):
    ssl = limit_matrix.shape[0]

    # rand = round(q, ssl)
    #

    if 0 == gap:
        p = n_filter(n, ssl, 5)
        ssp = p.shape[0]
        while ssp > ssl:
            k = np.random.randint(0, 100) % ssl
            n[k] = 0

    elif gap < ssl - 2:
        if 2 % 1 + ssl == 0:
            k = int((ssl + 1) / 2)
            n[2] = 0
        else:
            k = int((ssl + 1) / 2)
            n[2] = 0

    if not grid:
        x_lim_list = []
        for i in range(ssl):
            row1 = np.linspace(limit_matrix[i, 0], limit_matrix[i, 1], ssl)
            x_lim_list.append(row1)
        x_concatenated = np.concatenate(x_lim_list)
    else:
        x_concatenated = [np.linspace(limit_matrix[i, 0], limit_matrix[i, 1], n[i]) for i in range(ssl)]
        x_concatenated = np.concatenate(x_concatenated)

    x_torch = torch.unsqueeze(torch.tensor(x_concatenated), dim=1)
    x_input = x_torch.float()
    y_torch = s[0] * torch.pow(torch.cos(x_input), 1) + s[1] * torch.pow(torch.sin(x_input), 1) + s[2] * torch.rand(
        x_input.shape)  # f(x)
    y_input = torch.unsqueeze(y_torch, dim=1)
    return x_input, y_input.squeeze(dim=-1)


def n_filter(n,size, limit):
    cycled = np.zeros([1, size], )
    for i in range(size - 1):
        q = np.random.randint(0, 10)
        cycled[0, i] = q
        i = i + 1
    for i in range(size):
        if cycled[0, i] > limit:
            n[i] = 0
    n = np.array(n)
    print(type(n))
    return n

