import torch
import torch.nn.functional as F


# We only calculate the diagonal elements of the hessian
def logistic_hessian(f):
    f = f[:, :]
    pi = torch.sigmoid(f)
    return pi*(1-pi)


def softmax_hessian(f):
    s = F.softmax(f, dim=-1)
    return s - s*s


# Calculate the full softmax hessian
def full_softmax_hessian(f):
    s = F.softmax(f, dim=-1)
    e = torch.eye(s.shape[-1], dtype=s.dtype, device=s.device)
    return s[:, :, None]*e[None, :, :] - s[:, :, None]*s[:, None, :]


# Select memorable points ordered by their lambda values (descending=True picks most important points)
def select_memorable_points(dataloader, model, num_points=10, num_classes=2,
                            use_cuda=False, label_set=None, descending=True):
    memorable_points = {}
    scores = {}
    num_points_per_class = int(num_points/num_classes)
    for i, dt in enumerate(dataloader):
        data, target = dt
        if use_cuda:
            data_in = data.cuda()
        else:
            data_in = data
        if label_set == None:
            f = model.forward(data_in)
        else:
            f = model.forward(data_in, label_set)
        if f.shape[-1] > 1:
            lamb = softmax_hessian(f)
            if use_cuda:
                lamb = lamb.cpu()
            lamb = torch.sum(lamb, dim=-1)
            lamb = lamb.detach()
        else:
            lamb = logistic_hessian(f)
            if use_cuda:
                lamb = lamb.cpu()
            lamb = torch.squeeze(lamb, dim=-1)
            lamb = lamb.detach()
        for cid in range(num_classes):
            p_c = data[target == cid]
            if len(p_c) > 0:
                s_c = lamb[target == cid]
                if len(s_c) > 0:
                    if cid not in memorable_points:
                        memorable_points[cid] = p_c
                        scores[cid] = s_c
                    else:
                        memorable_points[cid] = torch.cat([memorable_points[cid], p_c], dim=0)
                        scores[cid] = torch.cat([scores[cid], s_c], dim=0)
                        if len(memorable_points[cid]) > num_points_per_class:
                            _, indices = scores[cid].sort(descending=descending)
                            memorable_points[cid] = memorable_points[cid][indices[:num_points_per_class]]
                            scores[cid] = scores[cid][indices[:num_points_per_class]]
    r_points = []
    r_labels = []
    for cid in range(num_classes):
        r_points.append(memorable_points[cid])
        r_labels.append(torch.ones(memorable_points[cid].shape[0], dtype=torch.long,
                                   device=memorable_points[cid].device)*cid)
    return [torch.cat(r_points, dim=0), torch.cat(r_labels, dim=0)]


# Randomly select some points as memory
def random_memorable_points(dataset, num_points, num_classes):
    memorable_points = {}
    num_points_per_class = int(num_points/num_classes)
    exact_num_points = num_points_per_class*num_classes
    idx_list = torch.randperm(len(dataset))
    select_points_num = 0
    for idx in range(len(idx_list)):
        data, label = dataset[idx_list[idx]]
        cid = label.item() if isinstance(label, torch.Tensor) else label
        if cid in memorable_points:
            if len(memorable_points[cid]) < num_points_per_class:
                memorable_points[cid].append(data)
                select_points_num += 1
        else:
            memorable_points[cid] = [data]
            select_points_num += 1
        if select_points_num >= exact_num_points:
            break
    r_points = []
    r_labels = []
    for cid in range(num_classes):
        r_points.append(torch.stack(memorable_points[cid], dim=0))
        r_labels.append(torch.ones(len(memorable_points[cid]), dtype=torch.long,
                                   device=r_points[cid].device)*cid)
    return [torch.cat(r_points, dim=0), torch.cat(r_labels, dim=0)]


# Update the fisher matrix after training on a task
def update_fisher(dataloader, model, opt, label_set=None, use_cuda=False):
    model.eval()
    for data, label in dataloader:
        if use_cuda:
            data = data.cuda()
        def closure():
            opt.zero_grad()
            if label_set == None:
                logits = model.forward(data)
            else:
                logits = model.forward(data, label_set)
            return logits
        opt.update_fisher(closure)


def save(opt, memorable_points, path):
    torch.save({
        'mu': opt.state['mu'],
        'fisher': opt.state['fisher'],
        'memorable_points': memorable_points
    }, path)


def load(opt, path):
    checkpoint = torch.load(path)
    opt.state['mu'] = checkpoint['mu']
    opt.state['fisher'] = checkpoint['fisher']
    return checkpoint['memorable_points']


def softmax_predictive_accuracy(logits_list, y, ret_loss = False):
    probs_list = [F.log_softmax(logits, dim=1) for logits in logits_list]
    probs_tensor = torch.stack(probs_list, dim = 2)
    probs = torch.mean(probs_tensor, dim=2)
    if ret_loss:
        loss = F.nll_loss(probs, y, reduction='sum').item()
    _, pred_class = torch.max(probs, 1)
    correct = pred_class.eq(y.view_as(pred_class)).sum().item()
    if ret_loss:
        return correct, loss
    return correct
