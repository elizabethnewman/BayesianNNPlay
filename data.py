
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
    #
    # elif region == 'left':
    #     y = y[x > cutoff]
    #     x = x[x > cutoff]
    # elif region == 'right':
    #     y = y[x < cutoff]
    #     x = x[x < cutoff]

    return x, y


