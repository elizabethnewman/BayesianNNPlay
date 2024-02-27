import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn

# Initialization

power_x = 2

n = 5000  # Number of Nodes
x1 = -4.5  # Lower Bound
x2 = 4.5  # Upper Bounda
k = 1   # 0 - sum, 1 - mean
kl_type = 0
s = 80     # S
ss = 15
if k == 1:
    kl_type = 'mean'
elif k == 0:
    kl_type = 'sum'

x = torch.linspace(x1, x2, n)
y = s*torch.pow(torch.cos(x),power_x) + ss*torch.rand(x.size(-1))  # f(x)

x = torch.unsqueeze(x, dim=1)
y = torch.unsqueeze(y, dim=1)

plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

model = nn.Sequential(
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.5, in_features=1, out_features=40),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=40, out_features=60),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=60, out_features=1),
)

mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction=kl_type, last_layer_only=False)
kl_weight = 0.4

optimizer = optim.Adam(model.parameters(), lr=.01)

i = 15
KL_store = []
MSE_store = []
for i in range(10):

    for step in range(300):
        pre = model(x)
        mse = mse_loss(pre, y)
        kl = kl_loss(model)
        cost = mse + kl_weight * kl

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
    KL_store.append(kl.item())
    MSE_store.append(mse.item())
KL_AVG = sum(KL_store)/i
MSE_AVG = sum(MSE_store)/i
print('- MSE : %s, KL : %s' % (KL_AVG, MSE_AVG))

x_test = torch.linspace(x1, x2, n)
y_test = y  # Restricted to be the same size as x initialization

# Solving would help to make testing different functions f(x) easier

x_test = torch.unsqueeze(x_test, dim=1)
y_test = torch.unsqueeze(y_test, dim=1)

# Display Results
from results import model_attempts

model_attempts(x_test,y_test,model)

