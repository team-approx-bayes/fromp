import torch
import torch.nn as nn
from models import CifarNet
from datasets import SplitCIFAR100
from torchvision import datasets
from torch.utils.data.dataloader import DataLoader
from utils import select_memorable_points, update_fisher, random_memorable_points
from opt_fromp import opt_fromp
import logging
from torchvision import transforms


def train(model, dataloaders, memorable_points, criterion, optimizer, label_sets, task_id=0,
          num_epochs=100, use_cuda=False):
    trainloader, testloader = dataloaders
    label_set_cur = label_sets[task_id]

    # Train
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in trainloader:
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            if isinstance(optimizer, opt_fromp):
                # Closure on current task's data
                def closure():
                    optimizer.zero_grad()
                    logits = model.forward(inputs, label_set_cur)
                    loss = criterion(logits, labels)
                    return loss, logits

                # Closure on memorable past data
                def closure_memorable_points(tid):
                    memorable_data_t = memorable_points[tid][0]
                    label_set_t = label_sets[tid]
                    if use_cuda:
                        memorable_data_t = memorable_data_t.cuda()
                    optimizer.zero_grad()
                    logits = model.forward(memorable_data_t, label_set_t)
                    return logits

                # Optimiser step
                loss, logits = optimizer.step(closure, closure_memorable_points, task_id)

    # Test
    model.eval()
    print('Begin testing...')
    test_accuracy = []
    for tid, testdata in enumerate(testloader):
        total = 0
        correct = 0
        for inputs, labels in testdata:
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            label_set_t = label_sets[tid]
            logits = model.forward(inputs, label_set_t)
            predict_label = torch.argmax(logits, dim=-1)
            total += inputs.shape[0]
            if use_cuda:
                correct += torch.sum(predict_label == labels).cpu().item()
            else:
                correct += torch.sum(predict_label == labels).item()
        test_accuracy.append(correct / total)

    return test_accuracy


def train_cifar(num_tasks, batch_size, lr, num_epochs, num_points, select_method='lambda_descend',
                use_cuda=True, tau=10):

    # Log console output to 'clog.txt'
    logging.basicConfig(filename='clog.txt')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Data generator
    num_classes_per_task = 10
    out_dim = num_tasks*num_classes_per_task
    data_transforms = transforms.ToTensor()
    cifar10_train = datasets.CIFAR10('datasets/',
                                     train=True, transform=data_transforms, download=True)
    cifar10_test = datasets.CIFAR10('datasets/',
                                    train=False, transform=data_transforms, download=True)
    cifar100_train = datasets.CIFAR100('datasets/',
                                       train=True, transform=data_transforms, download=True)
    cifar100_test = datasets.CIFAR100('datasets/',
                                      train=False, transform=data_transforms, download=True)
    data_gen = SplitCIFAR100(cifar100_train, cifar100_test)

    # Model
    model = CifarNet(3, out_dim)

    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        criterion.cuda()
        model.cuda()

    # Optimiser
    opt = opt_fromp(model, lr=lr, prior_prec=1e-4, grad_clip_norm=100, tau=tau)

    # Train on tasks
    memorable_points = []
    testloaders = []
    label_sets = []  # To record the labels for each task.
    acc_list = []
    for tid in range(num_tasks):

        # If not first task, need to calculate and store regularisation-term-related quantities
        if tid > 0:
            def closure(task_id):
                memorable_points_t = memorable_points[task_id][0]
                label_set_t = label_sets[task_id]
                if use_cuda:
                    memorable_points_t = memorable_points_t.cuda()
                opt.zero_grad()
                logits = model.forward(memorable_points_t, label_set_t)
                return logits
            opt.init_task(closure, tid, eps=1e-6)

        # Data generator for this task
        if tid == 0:
            itrain, itest = cifar10_train, cifar10_test
            ilabel_set = list(range(10))
        else:
            itrain, itest, ilabel_set = data_gen.next_task()
        label_sets.append(ilabel_set)
        itrainloader = DataLoader(dataset=itrain, batch_size=batch_size, shuffle=True, num_workers=3)
        itestloader = DataLoader(dataset=itest, batch_size=batch_size, shuffle=False, num_workers=3)
        memorableloader = DataLoader(dataset=itrain, batch_size=6, shuffle=False, num_workers=3)
        testloaders.append(itestloader)
        iloaders = [itrainloader, testloaders]

        # Train and test
        acc = train(model, iloaders, memorable_points, criterion, opt, label_sets, task_id=tid,
                    num_epochs=num_epochs, use_cuda=use_cuda)

        # Select memorable past datapoints
        if select_method == 'random':
            i_memorable_points = random_memorable_points(itrain, num_points=num_points, num_classes=num_classes_per_task)
        elif select_method == 'lambda_descend':
            i_memorable_points = select_memorable_points(memorableloader, model, num_points=num_points, num_classes=num_classes_per_task,
                                                    use_cuda=use_cuda, label_set=ilabel_set, descending=True)
        elif select_method == 'lambda_ascend':
            i_memorable_points = select_memorable_points(memorableloader, model, num_points=num_points, num_classes=num_classes_per_task,
                                                    use_cuda=use_cuda, label_set=ilabel_set, descending=False)
        else:
            raise Exception('Invalid memorable points selection method.')

        memorable_points.append(i_memorable_points)

        # Update covariance (\Sigma)
        update_fisher(memorableloader, model, opt, use_cuda=use_cuda, label_set=ilabel_set)

        print(acc)
        print('Mean accuracy after task %d: %f'%(tid+1, sum(acc)/len(acc)))
        logger.info('After learn task: %d'%(tid+1))
        logger.info(acc)
        logger.info('Mean accuracy is: %f'%(sum(acc)/len(acc)))
        acc_list.append(acc)

    return acc_list
