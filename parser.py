import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Bayesian VAE', add_help=False)
    parser.add_argument('--dataset', default='MNIST', type=str, choices=['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN'],
                        help='dataset to use')
    parser.add_argument('--batch_size', default=10000, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=5000, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', default=1e-1, type=float,
                        help='weight decay')
    parser.add_argument('--z_dim', default=64, type=int,
                        help='z dimension')
    parser.add_argument('--like_dim', default=256, type=int,
                        help='likelihood dimension')
    parser.add_argument('--nn_widths', default=[1, 64, 128, 256], type=list,
                        help='neural network widths')
    parser.add_argument('--n_obs', default=10000, type=int,
                        help='number of observations')
    parser.add_argument('--mc_samples', default=100, type=int,
                        help='number of monte carlo samples')
    parser.add_argument('--seed', default=315, type=int,
                        help='seed')
    return parser

args, unknown = get_args_parser().parse_known_args()