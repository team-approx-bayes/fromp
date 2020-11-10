from train_toydata import train_model
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--num_tasks', type=int, default=5, help='number of tasks for continual learning')
parser.add_argument('--batch_size', type=int, default=20, help='number of data points in a batch')
parser.add_argument('--hidden_size', type=int, default=20, help='network hidden layer size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=50, help='number of training epochs')
parser.add_argument('--num_points', type=int, default=20, help='number of inducing points for each task')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--select_method', type=str, default='lambda_descend',
                    help='method to select memorable points, can be: {random, lambda_descend, lambda_ascend}')
parser.add_argument('--tau', type=float, default=1,
                    help='hyperparameter tau (scaled by a factor N), should be scaled with num_points')

args = parser.parse_args()

def main(args):

    use_cuda = False

    train_model(args=args, use_cuda=use_cuda)


if __name__ == '__main__':

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print('FROMP on toy data, seed %d' % (args.seed))

    main(args)
