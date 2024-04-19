import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from master import evaluate, model, sPath
from data import comb, generate_mask


def train_plotter(sPath, num_draws, x_grid, y_grid):
    plt.scatter(x_grid, y_grid, color='r', s=1, label='training points', zorder=100)

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.legend()
    #plt.savefig(os.path.join(sPath, 'approximation.png'))
    plt.close()


def cost_plotter(sPath):
    kl = []
    mse = []
    epochs = []  # Use a list to collect all epochs
    cost = []
    # Read the data
    with open('experiments/cos/results.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header
        next(csv_reader)  # Skip the first data row

        for row in csv_reader:
            epoch_value = float(row[0])
            kl_value = float(row[4])
            mse_value = float(row[3])
            cost_value = float(row[2])

            epochs.append(epoch_value)
            kl.append(kl_value)
            mse.append(mse_value)
            cost.append(cost_value)
    # Convert lists to numpy arrays for better handling in plots
    p_epochs = np.array(epochs)
    p_kl = np.array(kl)
    p_mse = np.array(mse)
    p_cost = np.array(cost)



    # Plotting

    fig, ax = plt.subplots()
    ax.plot(p_epochs, p_cost, label='Cost (lambda * KL + MSE)', linewidth=2.0)
    ax.plot(p_epochs, p_kl, label='KL Contribution', linestyle='--')
    ax.plot(p_epochs, p_mse, label='MSE Value', linestyle='-.')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Values')
    ax.legend()

    plt.savefig(os.path.join(sPath, 'costs.png'))
    plt.close(fig)

    return


# Example usage:


def heatmap_plotter(x,y,mapsize,f,sPath):
    harvest = np.zeros((mapsize, mapsize))

    store = comb(x, 3)

    xn,yn = generate_mask(x, y, .1, 'middle', .05, .05)

    print(xn,yn)
    for i in range(mapsize):
        mse, kl = evaluate(model)
        #harvest[i, :] = kl
        harvest[i, :] = np.full(harvest.shape[1], kl)
        kl_weight = f + .1

    print(harvest)
    fig, ax = plt.subplots()
    im = ax.imshow(harvest)
    ax.set_title("KL-Variance Comparison")
    fig.tight_layout()
    plt.savefig(os.path.join(sPath, 'harvest.png'))
    plt.close



cost_plotter(sPath)
#i want to relate the idea of kl for each model but im not sure what to
# plot it against because I only have an evaluation of cost. So what
# Do i just compare final KL tweaked value and the final value of cost.
# What can i vary such that this becomes increasingly relevant. I can do this
# With plot holes to see how lack of data is imbedded into the solution of
# So I make it not left or not right but that wont really change anything
# Then can we sift and not completely remove. change the sifting on a side
# Now that becomes something interesting
# We can then repair it fully and compare to how it views sink holes


# DATA PREPARATION:

# CONFIRM MASK WORKS
# FIX COMBINE

    #
# CREATE CSV READER FOR result.csv


# VISUALIZATION PREPARATION:

# CONFIRM COST PLOT ALONG EVALUATIONS WITH BOTH LOSSES (KIND OF)
# CREATE DATA DEPENDENCIES ON VISUALS.PY
    # there should be a way for visual modules to augment data modules
    # ARGS: args.cutoff, args.proportion, args.gap, args.domain
# RUN MODULE THROUGH VISUALS TO CREATE HEATMAP
    # CHANGE DENISNTY OF POINTS AGAINST FREQUENCY OF SINK HOLES
    # COLOR SHOULD REPRESENT LAST EVALUATION OF COST

