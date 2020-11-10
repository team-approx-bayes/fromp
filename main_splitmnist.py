from train_splitmnist import train_splitmnist
import argparse
import torch
import numpy as np
import copy


parser = argparse.ArgumentParser()
parser.add_argument('--num_tasks', type=int, default=5, help='number of tasks for continual learning')
parser.add_argument('--batch_size', type=int, default=128, help='number of data points in a batch')
parser.add_argument('--hidden_size', type=int, default=256, help='network hidden layer size (2 hidden layers)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=15, help='number of training epochs')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--num_runs', type=int, default=1, help='how many random seed runs to average over')
parser.add_argument('--num_points', type=int, default=40, help='number of memorable points for each task')
parser.add_argument('--select_method', type=str, default='lambda_descend',
                    help='method to select memorable points, can be: {random, lambda_descend, lambda_ascend}')
parser.add_argument('--tau', type=float, default=10,
                    help='hyperparameter tau (scaled by a factor N), should be scaled with num_points')
args = parser.parse_args()


def main(args):

    use_cuda = True if torch.cuda.is_available() else False

    acc = train_splitmnist(num_tasks=args.num_tasks, batch_size=args.batch_size, hidden_size=args.hidden_size,
                           lr=args.lr, num_epochs=args.num_epochs, num_points=args.num_points,
                           select_method=args.select_method, use_cuda=use_cuda, tau=args.tau)
    return acc


if __name__ == '__main__':

    acc_list = []
    args_list = []

    for i in range(args.num_runs):
        # Set random seed
        np.random.seed(args.seed+i)
        torch.manual_seed(args.seed+i)
        print('\nSplit MNIST, seed', args.seed+i)

        # Run FROMP
        acc = main(args)
        acc_list.append(acc)
        args_list.append(copy.copy(args))

        # Save results
        save_results = False
        if save_results:
            save_path = 'results/'
            torch.save({
                'args_list': args_list,
                'accs_list': acc_list,
            }, save_path + 'splitmnist_seed_%d.tar' % (args.seed))

    # Print average final accuracy and standard deviation
    print('Mean accuracy', np.mean([np.mean(x[-1]) for x in acc_list]))
    print('Mean std', np.std([np.mean(x[-1]) for x in acc_list]))
