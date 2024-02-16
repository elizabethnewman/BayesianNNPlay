import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import os
from copy import deepcopy
from data import generate_polynomial_1D, generate_cos_polynomial_1D, generate_mask, combine
from utils import setup_parser, get_logger, makedirs, number_network_weights


# setup argument parser
parser = setup_parser()
args = parser.parse_args()

# create a naming convention for saving results
# sStartTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
file_details = 'kl_weight_%0.2f_final_sigma_%0.2f_%s' % (args.kl_weight, args.final_sigma, args.data)


# path to save results
sPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experiments', file_details)

# logger
makedirs(sPath)
logger = get_logger(logpath=os.path.join(sPath, 'results.log'), filepath=os.path.abspath(__file__),
                    saving=True, mode="w")

logger.info(f'Bayesian Neural Network')
logger.info(f'args: {args}')

torch.manual_seed(args.seed)


#%% generate data
if args.data == 'cos':
    x, y = generate_cos_polynomial_1D(n_pts=args.n_train, domain=args.domain,
                                      scale=args.scale, noise_level=args.noise_level, power=args.power, grid=False)

elif args.data == 'poly':
    x, y = generate_polynomial_1D(n_pts=args.n_train, domain=args.domain,
                                  scale=args.scale, noise_level=args.noise_level, power=args.power, grid=False)
elif args.data == 'combine':
    import numpy as np
    x_matrix = np.array([[0, 1], [2, 3], [4, 5]])  # Lower Bound
    size_limits = x_matrix.shape
    size = [10, 0, 10]
    n = [50, 10, 4]
    x, y = combine(x_matrix, n, size)

else:
    raise ValueError(f'Unknown data type: {args.data}')

# mask data (currently only supports for masking in middle of domain)
if args.mask:
    x, y = generate_mask(x, y, cutoff=args.cutoff, proportion=args.propotion)


plt.figure()
plt.scatter(x, y, label='training points')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig(os.path.join(sPath, 'training_data.png'))
# plt.show()


#%% create network

model = nn.ModuleList()

w_old = x.shape[1]
for i, w in enumerate(args.width):
    if len(args.prior_mu) < len(args.width):
        prior_mu = args.prior_mu[0]
    else:
        prior_mu = args.prior_mu[i]

    if len(args.prior_sigma) < len(args.width):
        prior_sigma = args.prior_sigma[0]
    else:
        prior_sigma = args.prior_sigma[i]

    model.append(bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=w_old, out_features=w))
    model.append(nn.ReLU())

    # update w_old
    w_old = deepcopy(w)

# final layer
model.append(bnn.BayesLinear(prior_mu=args.final_mu, prior_sigma=args.final_sigma,
                             in_features=w_old, out_features=y.shape[1]))

model = nn.Sequential(*model)

mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction=args.kl_type, last_layer_only=args.last_layer_only)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

#%% setup logger information
logger.info("---------------------- Network ----------------------------")
logger.info(model)
logger.info("Number of trainable parameters: {}".format(number_network_weights(model)))
logger.info("--------------------------------------------------")
logger.info(str(optimizer))
logger.info(str(scheduler))
logger.info("dtype={:} device={:}".format(x.dtype, x.device))
logger.info("epochs={:} ".format(args.max_epochs))
logger.info("saveLocation = {:}".format(sPath))
logger.info("--------------------------------------------------\n")


#%% train


# helper functions
def freeze(model):
    for layer in model:
        if hasattr(layer, 'freeze'):
            layer.freeze()


def unfreeze(model):
    for layer in model:
        if hasattr(layer, 'unfreeze'):
            layer.unfreeze()


def evaluate(model):
    with torch.no_grad():
        model.eval()
        freeze(model)
        pred = model(x)
        mse = mse_loss(pred, y)
        kl = kl_loss(model)
    return mse.item(), kl.item()


# store results
results = {'headers': ('epoch', 'f', 'mse', 'kl'),
           'frmt': '{:<15d}{:<15.2e}{:<15.2e}{:<15.2e}',
           'values': []}

# initial evaluation
mse, kl = evaluate(model)
results['values'].append([-1,  mse + args.kl_weight * kl, mse, kl])

logger.info((len(results['headers']) * '{:<15s}').format(*results['headers']))
logger.info(results['frmt'].format(*results['values'][-1]))

t0 = time.perf_counter()

for i in range(args.max_epochs):
    unfreeze(model)
    for step in range(args.num_samples):
        pre = model(x)
        mse = mse_loss(pre, y)
        kl = kl_loss(model)
        cost = mse + args.kl_weight * kl

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    # adjust learning rate
    scheduler.step()

    # evaluate and store results
    mse, kl = evaluate(model)
    results['values'].append([i, mse + args.kl_weight * kl, mse, kl])
    logger.info(results['frmt'].format(*results['values'][-1]))

t1 = time.perf_counter()

if torch.cuda.is_available():
    torch.cuda.synchronize()

logger.info('Total Training Time: {:.2f} seconds'.format(t1 - t0))

results['args'] = args
torch.save(model.state_dict(), os.path.join(sPath, 'model.pth'))

pd.DataFrame.to_csv(pd.DataFrame(results['values'], columns=results['headers']), os.path.join(sPath, 'results.csv'))



#%% plot results

if args.data == 'cos':
    x_grid, y_grid = generate_cos_polynomial_1D(n_pts=args.n_test, domain=args.domain,
                                                scale=args.scale, noise_level=args.noise_level, power=args.power, grid=True)

elif args.data == 'poly':
    x_grid, y_grid = generate_polynomial_1D(n_pts=args.n_train, domain=args.domain,
                                            scale=args.scale, noise_level=args.noise_level, power=args.power, grid=True)
elif args.data == 'combine':
    x_grid, y_grid = combine(x_matrix, n, size, grid=True)


unfreeze(model)

num_draws = 100

plt.figure()
for i in range(num_draws):
    y_predict = model(x_grid).detach()
    plt.plot(x_grid, y_predict, 'k-', linewidth=2, alpha=0.1)

plt.scatter(x_grid, y_grid, color='r', s=10, label='training points', zorder=100)

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
plt.savefig(os.path.join(sPath, 'approximation.png'))
