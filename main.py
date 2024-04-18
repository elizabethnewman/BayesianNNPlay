import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn

# Initialization




model = nn.Sequential(
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.5, in_features=1, out_features=40),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=40, out_features=60),
    nn.ReLU(),
    bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=60, out_features=2),
)

mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction=kl_type, last_layer_only=False)
kl_weight = 0.7

optimizer = optim.Adam(model.parameters(), lr=.01)

i = 15
KL_store = []
MSE_store = []
for i in range(10):
    for step in range(300):
        pre = model(x)
        mse = mse_loss(pre,y)
        kl = kl_loss(model)
        cost = mse + kl_weight * kl

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
    KL_store.append(kl.item())
    MSE_store.append(mse.item())
KL_AVG = sum(KL_store)/i
MSE_AVG = sum(MSE_store)/i

