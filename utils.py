import logging
import os
import argparse


def setup_parser():
    parser = argparse.ArgumentParser()

    # reproducibility
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # data
    parser.add_argument('--n_train', type=int, default=1000, help='number of training points')
    parser.add_argument('--n_val', type=int, default=1000, help='number of validation points')
    parser.add_argument('--n_test', type=int, default=1000, help='number of test points')
    parser.add_argument('--data', type=str, default='cos', help='type of data')
    parser.add_argument('--domain', type=float, nargs=2, default=[-4.5, 4.5], help='domain of function')
    parser.add_argument('--scale', type=float, default=80.0, help='scale of function')
    parser.add_argument('--noise_level', type=float, default=15.0, help='noise added to function')
    parser.add_argument('--power', type=int, default=2, help='power of function')
    parser.add_argument('--mask', action='store_true', help='mask function')
    parser.add_argument('--cutoff', type=float, default=1.0, help='cutoff for mask')
    parser.add_argument('--proportion', type=float, default=0.9,
                        help='proportion of points to mask (0 = do not mask, 1 = mask all)')
    parser.add_argument('--gap', type=int, default=0)
    # training
    parser.add_argument('--max_epochs', type=int, default=10, help='maximum number of epochs')
    parser.add_argument('--num_samples', type=int, default=300, help='number of samples drawn per epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gamma', type=float, default=1, help='decay rate for scheduler')
    parser.add_argument('--step_size', type=float, default=100, help='step size for scheduler')

    # network
    parser.add_argument('--width', type=int, nargs="+", default=[40, 60], help='width of network')
    parser.add_argument('--prior_mu', type=float, nargs="+", default=[0.0, 0.0], help='prior mean')
    parser.add_argument('--prior_sigma', type=float, nargs="+", default=[0.5, 0.5], help='prior standard deviation')

    parser.add_argument('--final_mu', type=float, default=0.0, help='final layer prior mean')
    parser.add_argument('--final_sigma', type=float, default=0.25, help='final layer prior standard deviation')

    # loss
    parser.add_argument('--kl_type', type=str, default='mean', help='type of KL divergence; options are mean, sum, none')
    parser.add_argument('--last_layer_only', action='store_true', help='only compute KL divergence on last layer')
    parser.add_argument('--kl_weight', type=float, default=1.0, help='weight of KL divergence')

    return parser


# Directly modified from the OT-flow Github Depository

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False, mode="a"):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode=mode)
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def makedirs(dirname):
    """
    make the directory folder structure
    :param dirname: string path
    :return: void
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def number_network_weights(net):
    n = 0
    for p in net.parameters():
        n += p.numel()

    return n