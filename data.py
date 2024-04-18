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
                               scale: float = 80.0, noise_level: float = 15.0, power: int = 3,
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

def generate_sinosoidal_polynomial_1D(n_pts: int = 5000, domain: tuple = (-4.5, 4.5),
                               scale: float = 80.0, noise_level: float = 15.0, power: int = 1,
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

    y = scale * (torch.pow(torch.cos(x), power) + torch.pow(torch.sin(x), power)) + noise_level * torch.rand(x.size(-1))  # f(x)

    # apply mask
    # y = y[x.abs() > 1.0]
    # x = x[x.abs() > 1.0]

    return x.reshape(-1, 1), y.reshape(-1, 1)


def generate_mask(x, y, ep, region: str = 'middle', cutoff: float = 0.5, proportion: float = 0.5):
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
    x_min = x.min()
    x_max = x.max()

    idx = 1 * (x.abs() > cutoff)
    idx_no_mask = idx.nonzero(as_tuple=True)[0]
    idx_mask = (1 - idx).nonzero(as_tuple=True)[0]
    idx_mask = idx_mask[torch.randperm(len(idx_mask))[:int((1 - proportion) * len(idx_mask))]]

    idx = torch.cat((idx_no_mask, idx_mask))
    idx = idx[torch.randperm(len(idx))]
    y = y[idx]
    x = x[idx]

    if region == 'middle':
        x_max = x_mean + ep
        x_min = x_mean - ep

        y = y[x > x_min]
        y = y[x < x_max]

        x = y[x > x_min]
        x = y[x < x_max]


    elif region == 'left':
         y = y[x > cutoff]

         x = x[x > cutoff]

    elif region == 'right':
         y = y[x < cutoff]
         x = x[x < cutoff]

    elif region == 'between':
        y = y[x > x_min]
        y = y[x < x_max]

        x = y[x > x_min]
        x = y[x < x_max]

    # Assuming tensor has shape (batch_size, num_elements)
    num_el = x.shape[0]


    # Reshape the tensor to have each element as a new row
    new_shape = (num_el, 1)

    xn = torch.reshape(x, new_shape)
    yn = torch.reshape(y,new_shape)
    return xn, yn


def combine(n, gap, domain, grid=False):
    # obtains shape of x

    if 0 != gap:
        if grid == True:
            domain = tuple(-4.5, 4.5)

    # discrete or continuous domain
    if grid:
        x = torch.linspace(domain[0], domain[1], n)
    else:
        x = torch.rand(n) * (domain[1] - domain[0]) + domain[0]

    # if testing, filtering data can be done here
    #if test == True:
        #sift = any_filter(n, domain, limit)
    # To split data

    x_input = x_torch.float()
    y_torch = s[0] * torch.pow(torch.cos(x_input), 1) + s[1] * torch.pow(torch.sin(x_input), 1) + s[2] * torch.rand(
        x_input.shape)  # f(x)
    y_input = torch.unsqueeze(y_torch, dim=1)
    return x_input, y_input.squeeze(dim=-1)


def any_filter(any, size, threshold, value, limit):
    if limit == False:
        cycled = np.zeros([1, size], )
        for i in range(size - 1):
            q = np.random.randint(0, 10)
            cycled[0, i] = q
            i = i + 1
        for i in range(size):
            if cycled[0, i] > threshold:
                any[i] = 0
    any = np.array(any)
    cycled = np.zeros([1, size])
    for i in range(size):
        if cycled[0, i] > limit:
            any[i] = value

    return any


def prepare(x, y):
    x_torch = torch.unsqueeze(torch.tensor(x), dim=1)
    x = torch.unsqueeze(x_torch, dim=1)
    y_torch = torch.unsqueeze(torch.tensor(y), dim=1)
    y = torch.unsqueeze(y_torch, dim=1)
    return x, y
