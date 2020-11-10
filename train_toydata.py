import torch
import torch.nn as nn
from models import MLP
from opt_fromp import opt_fromp
from datasets import ToydataGenerator
from torch.utils.data.dataloader import DataLoader
from utils import select_memorable_points, update_fisher, random_memorable_points
import numpy as np
import time
import matplotlib.pyplot as plt


def train(model, dataloaders, memorable_points, criterion, optimizer, task_id=0, num_epochs=25, use_cuda=False):

    trainloader, testloader = dataloaders

    model.train()
    for epoch in range(num_epochs):
        running_train_loss = 0
        count = 0
        for inputs, labels in trainloader:
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # Continual learning optimiser
            if isinstance(optimizer, opt_fromp):
                def closure():
                    optimizer.zero_grad()
                    logits = model.forward(inputs)
                    loss = criterion(torch.squeeze(logits, dim=-1), labels)
                    return loss, logits
                def closure_memorable_points(task_id):
                    memorable_points_t = memorable_points[task_id]
                    if use_cuda:
                        memorable_points_t = memorable_points_t.cuda()
                    optimizer.zero_grad()
                    logits = model.forward(memorable_points_t)
                    return logits
                loss, logits = optimizer.step(closure, closure_memorable_points, task_id)

            if use_cuda:
                loss_val = loss.detach().cpu().item()
            else:
                loss_val = loss.detach().item()
            running_train_loss += loss_val
            count += 1
        if epoch == 0 or epoch == num_epochs-1:
            print('Epoch[%d]: Train loss: %f' %(epoch, running_train_loss/count))

    # Run on test data (a 2D grid of points for plotting)
    full_outputs = []
    model.eval()
    print('Begin test.')
    for inputs, _ in testloader:
        if use_cuda:
            inputs = inputs.cuda()
        outputs = model(inputs)
        full_outputs.append(outputs)
    full_outputs = torch.cat(full_outputs, dim=0)
    full_outputs = torch.sigmoid(full_outputs)

    return full_outputs


def train_model(args, use_cuda=False):
    start_time = time.time()

    # Read values from args
    num_tasks = args.num_tasks
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    lr = args.lr
    num_epochs = args.num_epochs
    num_points = args.num_points
    coreset_select_method = args.select_method

    # Some parameters
    dataset_generation_test = False
    dataset_num_samples = 2000

    # Colours for plotting
    color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    # Load / Generate toy data
    datagen = ToydataGenerator(max_iter=num_tasks, num_samples=dataset_num_samples)

    plt.figure()
    datagen.reset()
    total_loaders = []
    criterion_cl = nn.CrossEntropyLoss()

    # Create model
    layer_size = [2, hidden_size, hidden_size, 2]
    model = MLP(layer_size, act='sigmoid')
    if use_cuda:
        model = model.cuda()

    # Optimiser
    opt = opt_fromp(model, lr=lr, prior_prec=1e-4, grad_clip_norm=None, tau=args.tau)

    memorable_points = None
    inducing_targets = None

    for tid in range(num_tasks):
        # If not first task, need to calculate and store regularisation-term-related quantities
        if tid > 0:
            def closure(task_id):
                memorable_points_t = memorable_points[task_id]
                if use_cuda:
                    memorable_points_t = memorable_points_t.cuda()
                opt.zero_grad()
                logits = model.forward(memorable_points_t)
                return logits
            opt.init_task(closure, tid, eps=1e-3)

        # Data generator for this task
        itrain, itest = datagen.next_task()
        itrainloader = DataLoader(dataset=itrain, batch_size=batch_size, shuffle=True, num_workers=8)
        itestloader = DataLoader(dataset=itest, batch_size=batch_size, shuffle=False, num_workers=8)
        inducingloader = DataLoader(dataset=itrain, batch_size=batch_size, shuffle=False, num_workers=8)
        iloaders = [itrainloader, itestloader]

        if tid == 0:
            total_loaders = [itrainloader]
        else:
            total_loaders.append(itrainloader)

        # Train and test
        cl_outputs = train(model, iloaders, memorable_points, criterion_cl, opt, task_id=tid, num_epochs=num_epochs,
                           use_cuda=use_cuda)

        # Select memorable past datapoints
        if coreset_select_method == 'random':
            i_memorable_points, i_inducing_targets = random_memorable_points(
                itrain, num_points=num_points, num_classes=2)
        else:
            i_memorable_points, i_inducing_targets = select_memorable_points(
                inducingloader, model, num_points=num_points, use_cuda=use_cuda)

        # Add memory points to set
        if tid > 0:
            memorable_points.append(i_memorable_points)
            inducing_targets.append(i_inducing_targets)
        else:
            memorable_points = [i_memorable_points]
            inducing_targets = [i_inducing_targets]

        # Update covariance (\Sigma)
        update_fisher(inducingloader, model, opt, use_cuda=use_cuda)

        # Plot visualisation (2D figure)
        cl_outputs, _ = torch.max(cl_outputs, dim=-1)
        cl_show = 2*cl_outputs - 1

        cl_show = cl_show.detach()
        if use_cuda:
            cl_show = cl_show.cpu()
        cl_show = cl_show.numpy()
        cl_show = cl_show.reshape(datagen.test_shape)

        plt.figure()
        axs = plt.subplot(111)
        axs.title.set_text('FROMP')
        if not dataset_generation_test:
            plt.imshow(cl_show, cmap='gray',
                       extent=(datagen.x_min, datagen.x_max, datagen.y_min, datagen.y_max), origin='lower')
        for l in range(tid+1):
            idx = np.where(datagen.y == l)
            plt.scatter(datagen.X[idx][:,0], datagen.X[idx][:,1], c=color[l], s=0.03)
            idx = np.where(datagen.y == l+datagen.offset)
            plt.scatter(datagen.X[idx][:,0], datagen.X[idx][:,1], c=color[l+datagen.offset], s=0.03)
            if not dataset_generation_test:
                plt.scatter(memorable_points[l][:,0], memorable_points[l][:, 1], c='m', s=0.4, marker='x')

        plt.show()

        # Calculate and print train accuracy and negative log likelihood
        with torch.no_grad():
            if not dataset_generation_test:
                model.eval()
                N = len(itrain)

                metric_task_id = 0
                nll_loss_avg = 0
                accuracy_avg = 0
                for metric_loader in total_loaders:
                    nll_loss = 0
                    correct = 0
                    for inputs, labels in metric_loader:
                        if use_cuda:
                            inputs, labels = inputs.cuda(), labels.cuda()

                        logits = model.forward(inputs)

                        nll_loss += nn.functional.cross_entropy(torch.squeeze(logits, dim=-1), labels) * float(inputs.shape[0])

                        # Calculate predicted classes
                        pred = logits.data.max(1, keepdim=True)[1]

                        # Count number of correctly predicted datapoints
                        correct += pred.eq(labels.data.view_as(pred)).sum()

                    nll_loss /= N
                    accuracy = float(correct) / float(N) * 100.

                    print('Task {}, Train accuracy: {:.2f}%, Train Loglik: {:.4f}'.format(
                        metric_task_id, accuracy, nll_loss))

                    metric_task_id += 1
                    nll_loss_avg += nll_loss
                    accuracy_avg += accuracy

                print('Avg train accuracy: {:.2f}%, Avg train Loglik: {:.4f}'.format(
                    accuracy_avg/metric_task_id, nll_loss_avg/metric_task_id))

    print('Time taken: ', time.time()-start_time)
