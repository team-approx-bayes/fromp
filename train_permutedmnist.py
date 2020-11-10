import torch
import torch.nn as nn
from models import MLP
from datasets import PermutedMnistGenerator
from torch.utils.data.dataloader import DataLoader
from utils import select_memorable_points, update_fisher, random_memorable_points
from opt_fromp import opt_fromp
import logging


def train(model, dataloaders, memorable_points, criterion, optimizer, task_id=0, num_epochs=10, use_cuda=True):
    trainloader, testloader = dataloaders

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
                    logits = model.forward(inputs)
                    loss = criterion(logits, labels)
                    return loss, logits

                # Closure on memorable past data
                def closure_memorable_points(task_id):
                    memorable_points_t = memorable_points[task_id][0]
                    if use_cuda:
                        memorable_points_t = memorable_points_t.cuda()
                    optimizer.zero_grad()
                    logits = model.forward(memorable_points_t)
                    return logits

                # Optimiser step
                loss, logits = optimizer.step(closure, closure_memorable_points, task_id)

    # Test
    model.eval()
    print('Begin testing...')
    test_accuracy = []
    for testdata in testloader:
        total = 0
        correct = 0
        for inputs, labels in testdata:
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            logits = model.forward(inputs)
            predict_label = torch.argmax(logits, dim=-1)
            total += inputs.shape[0]
            if use_cuda:
                correct += torch.sum(predict_label == labels).cpu().item()
            else:
                correct += torch.sum(predict_label == labels).item()
        test_accuracy.append(correct / total)

    return test_accuracy


def train_permutedmnist(num_tasks, batch_size, hidden_size, lr, num_epochs, num_points,
                        select_method='lambda_descend', use_cuda=True, tau=0.5):

    # Log console output to 'pmlog.txt'
    logging.basicConfig(filename='pmlog.txt')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Data generator
    datagen = PermutedMnistGenerator(max_iter=num_tasks)

    # Model
    num_classes = 10
    layer_size = [784, hidden_size, hidden_size, num_classes]
    model = MLP(layer_size, act='relu')

    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        criterion.cuda()
        model.cuda()

    # Optimiser
    opt = opt_fromp(model, lr=lr, prior_prec=1e-5, grad_clip_norm=0.01, tau=tau)

    # Train on tasks
    memorable_points = []
    testloaders = []
    acc_list = []
    for tid in range(num_tasks):

        # If not first task, need to calculate and store regularisation-term-related quantities
        if tid > 0:
            def closure(task_id):
                memorable_points_t = memorable_points[task_id][0]
                if use_cuda:
                    memorable_points_t = memorable_points_t.cuda()
                opt.zero_grad()
                logits = model.forward(memorable_points_t)
                return logits
            opt.init_task(closure, tid, eps=1e-5)

        # Data generator for this task
        itrain, itest = datagen.next_task()
        itrainloader = DataLoader(dataset=itrain, batch_size=batch_size, shuffle=True, num_workers=3)
        itestloader = DataLoader(dataset=itest, batch_size=batch_size, shuffle=False, num_workers=3)
        memorableloader = DataLoader(dataset=itrain, batch_size=batch_size, shuffle=False, num_workers=3)
        testloaders.append(itestloader)
        iloaders = [itrainloader, testloaders]

        # Train and test
        acc = train(model, iloaders, memorable_points, criterion, opt, task_id=tid, num_epochs=num_epochs,
                    use_cuda=use_cuda)

        # Select memorable past datapoints
        if select_method == 'random':
            i_memorable_points = random_memorable_points(itrain, num_points=num_points, num_classes=num_classes)
        elif select_method == 'lambda_descend':
            i_memorable_points = select_memorable_points(memorableloader, model, num_points=num_points, num_classes=num_classes,
                                                    use_cuda=use_cuda, descending=True)
        elif select_method == 'lambda_ascend':
            i_memorable_points = select_memorable_points(memorableloader, model, num_points=num_points, num_classes=num_classes,
                                                    use_cuda=use_cuda, descending=False)
        else:
            raise Exception('Invalid memorable points selection method.')

        memorable_points.append(i_memorable_points)

        # Update covariance (\Sigma)
        update_fisher(memorableloader, model, opt, use_cuda=use_cuda)

        print(acc)
        print('Mean accuracy after task %d: %f'%(tid+1, sum(acc)/len(acc)))
        logger.info('After learn task: %d'%(tid+1))
        logger.info(acc)
        acc_list.append(acc)

    return acc_list
