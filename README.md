# BayesianNNPlay
A tool to explore Bayesian Neural Networks


To run the master script (without a csv file)
```console
python master.py
```

To run for 100 epochs, we can do

```console
python master.py --max_epochs 100
```

We can also change the network architecture!

```console
python master.py --width 10 20 30
```

and many more things!


To run from a csv file (e.g., like test.csv), we can do the following:

```console
python run_experiments.py test.csv
```

and we should also be able to run in parallel (to be added).
