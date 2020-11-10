import math
import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn as nn
import torch.nn.functional as f
from utils import logistic_hessian, full_softmax_hessian


def update_input(self, input, output):
    self.input = input[0].data
    self.output = output


def _check_param_device(param, old_param_device):
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # check if in same gpu
            warn = (param.get_device() != old_param_device)
        else:  # check if in cpu
            warn = (old_param_device != -1)
        if warn:
            raise typeerror('found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device


def parameters_to_matrix(parameters):
    param_device = None
    mat = []
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        m = param.shape[0]
        mat.append(param.view(m, -1))
    return torch.cat(mat, dim=-1)


def parameters_grads_to_vector(parameters):
    param_device = None
    vec = []
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        if param.grad is None:
            raise valueerror('gradient not available')
        vec.append(param.grad.data.view(-1))
    return torch.cat(vec, dim=-1)


# Optimizer that is torch.optim.adam with extra regularisation terms for FROMP (Pan et al., 2020)
# grad_clip_norm: What value to clip the norm of the gradient to during training
class opt_fromp(Optimizer):
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, prior_prec=1e-3, grad_clip_norm=1, tau=1,
                 amsgrad=False):
        if not 0.0 <= lr:
            raise valueerror("invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise valueerror("invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise valueerror("invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise valueerror("invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= prior_prec:
            raise valueerror("invalid prior precision: {}".format(prior_prec))
        if grad_clip_norm is not None and not 0.0 <= grad_clip_norm:
            raise valueerror("invalid gradient clip norm: {}".format(grad_clip_norm))
        if not 0.0 <= tau:
            raise valueerror("invalid tau: {}".format(tau))
        defaults = dict(lr=lr, betas=betas, eps=eps, prior_prec=prior_prec, grad_clip_norm=grad_clip_norm,
                        tau=tau, amsgrad=amsgrad)
        super(opt_fromp, self).__init__(model.parameters(), defaults)

        self.model = model
        self.train_modules = []
        self.set_train_modules(model)
        for module in self.train_modules:
            module.register_forward_hook(update_input)

        parameters = self.param_groups[0]['params']

        p = parameters_to_vector(parameters)
        self.state['mu'] = p.clone().detach()
        self.state['mu_previous'] = p.clone().detach()
        self.state['fisher'] = torch.zeros_like(self.state['mu'])
        self.state['step'] = 0
        self.state['exp_avg'] = torch.zeros_like(self.state['mu'])
        self.state['exp_avg_sq'] = torch.zeros_like(self.state['mu'])
        if amsgrad:
            self.state['max_exp_avg_sq'] = torch.zeros_like(self.state['mu'])


    # Load zeros into model
    def load_zeros(self):
        zeros = torch.zeros_like(self.state['mu'])
        vector_to_parameters(zeros, self.param_groups[0]['params'])


    def update_mu(self):
        parameters = self.param_groups[0]['params']
        p = parameters_to_vector(parameters)
        self.state['mu'] = p.clone().detach()

    # Calculate values (memorable_logits, hkh_l) for regularisation term (all but the first task)
    def init_task(self, closure, task_id, eps=1e-5):
        self.state['exp_avg'] = torch.zeros_like(self.state['mu'])
        self.state['exp_avg_sq'] = torch.zeros_like(self.state['mu'])
        self.state['step'] = 0
        self.state['kernel_inv'] = []
        self.state['memorable_logits'] = []
        fisher = self.state['fisher']
        prior_prec = self.param_groups[0]['prior_prec']
        mu = self.state['mu']
        self.state['mu_previous'] = mu.clone().detach()
        parameters = self.param_groups[0]['params']
        vector_to_parameters(mu, parameters)
        covariance = 1. / (fisher + prior_prec)

        # Calculate kernel = J \Sigma J^T for all memory points, and store via cholesky decomposition
        self.model.eval()
        for i in range(task_id):
            preds = closure(i)
            num_fun = preds.shape[-1]
            if num_fun == 1:
                preds = torch.sigmoid(preds)
            else:
                preds = torch.softmax(preds, dim=-1)
            self.state['memorable_logits'].append(preds.detach())
            lc = []
            for module in self.train_modules:
                lc.append(module.output)
            kernel_inv = []
            for fi in range(num_fun):
                loss = preds[:, fi].sum()
                retain_graph = True if fi < num_fun - 1 else None
                grad = self.cac_grad(loss, lc, retain_graph=retain_graph)
                kernel = torch.einsum('ij,j,pj->ip', grad, covariance, grad) + \
                    torch.eye(grad.shape[0], dtype=grad.dtype, device=grad.device)*eps
                kernel_inv.append(torch.cholesky_inverse(torch.cholesky(kernel)))

            self.state['kernel_inv'].append(kernel_inv)


    # For calculating Jacobians in PyTorch
    def set_train_modules(self, module):
        if len(list(module.children())) == 0:
            if len(list(module.parameters())) != 0:
                self.train_modules.append(module)
        else:
            for child in list(module.children()):
                self.set_train_modules(child)


    # Update the hyperparameter tau
    def update_tau(self, tau):
        self.defaults['tau'] = tau


    # Calculate the gradient (part of calculating Jacobian) of the parameters lc wrt loss
    def cac_grad(self, loss, lc, retain_graph=None):
        linear_grad = torch.autograd.grad(loss, lc, retain_graph=retain_graph)
        grad = []
        for i, module in enumerate(self.train_modules):
            g = linear_grad[i]
            a = module.input.clone().detach()
            m = a.shape[0]

            if isinstance(module, nn.Linear):
                grad.append(torch.einsum('ij,ik->ijk', g, a))
                if module.bias is not None:
                    grad.append(g)

            if isinstance(module, nn.Conv2d):
                a = f.unfold(a, kernel_size=module.kernel_size, dilation=module.dilation, padding=module.padding,
                                stride=module.stride)
                _, k, hw = a.shape
                _, c, _, _ = g.shape
                g = g.view(m, c, -1)
                grad.append(torch.einsum('ijl,ikl->ijk', g, a))
                if module.bias is not None:
                    a = torch.ones((m, 1, hw), device=a.device)
                    grad.append(torch.einsum('ijl,ikl->ijk', g, a))

            if isinstance(module, nn.BatchNorm1d):
                grad.append(torch.mul(g, a))
                if module.bias is not None:
                    grad.append(g)

            if isinstance(module, nn.BatchNorm2d):
                grad.append(torch.einsum('ijkl->ij', torch.mul(g, a)))
                if module.bias is not None:
                    grad.append(torch.einsum('ijkl->ij', g))

        grad_m = parameters_to_matrix(grad)
        return grad_m.detach()


    # Calculate the Jacobian matrix
    def cac_jacobian(self, output, lc):
        if output.dim() > 2:
            raise valueerror('the dimension of output must be smaller than 3.')
        elif output.dim() == 2:
            num_fun = output.shape[1]
        grad = []
        for i in range(num_fun):
            retain_graph = None if i == num_fun - 1 else True
            loss = output[:, i].sum()
            g = self.cac_grad(loss, lc, retain_graph=retain_graph)
            grad.append(g)
        result = torch.zeros((grad[0].shape[0], grad[0].shape[1], num_fun),
                             dtype=grad[0].dtype, device=grad[0].device)
        for i in range(num_fun):
            result[:, :, i] = grad[i]
        return result


    # After training on a new task, update the fisher matrix estimate
    def update_fisher(self, closure):
        fisher = self.state['fisher']
        preds = closure()
        lc = []
        for module in self.train_modules:
            lc.append(module.output)
        jac = self.cac_jacobian(preds, lc)
        if preds.shape[-1] == 1:
            hes = logistic_hessian(preds).detach()
            hes = hes[:, :, None]
        else:
            hes = full_softmax_hessian(preds).detach()
        jhj = torch.einsum('ijd,idp,ijp->j', jac, hes, jac)
        fisher.add_(jhj)


    # Iteration step for this optimiser
    # Added extra regularisation terms to torch.optim.adam
    def step(self, closure_data, closure_memorable_points, task_id):
        defaults = self.defaults
        lr = self.param_groups[0]['lr']
        beta1, beta2 = self.param_groups[0]['betas']
        amsgrad = self.param_groups[0]['amsgrad']
        parameters = self.param_groups[0]['params']
        mu = self.state['mu']

        #vector_to_parameters(mu, parameters)
        self.model.train()

        # Normal loss term over current task's data
        vector_to_parameters(mu, parameters)
        loss_cur, preds_cur = closure_data()
        loss_cur.backward(retain_graph=True)
        grad = parameters_grads_to_vector(parameters).detach()
        grad.mul_(1/defaults['tau'])


        # The loss term corresponding to memorable points
        if task_id > 0:
            self.model.eval()
            kernel_inv = self.state['kernel_inv']
            memorable_logits = self.state['memorable_logits']
            grad_t_sum = torch.zeros_like(grad)
            for t in range(task_id):
                preds_t = closure_memorable_points(t)
                num_fun = preds_t.shape[-1]
                if num_fun == 1:
                    preds_t = torch.sigmoid(preds_t)
                else:
                    preds_t = torch.softmax(preds_t, dim=-1)
                lc = []
                for module in self.train_modules:
                    lc.append(module.output)
                for fi in range(num_fun):
                    # \Lambda * Jacobian
                    loss_jac_t = preds_t[:, fi].sum()
                    retain_graph = True if fi < num_fun - 1 else None
                    jac_t = self.cac_grad(loss_jac_t, lc, retain_graph=retain_graph)

                    # m_t - m_{t-1}
                    logits_t = preds_t[:, fi].detach()
                    delta_logits = logits_t - memorable_logits[t][:,fi]

                    # K_{t-1}^{-1}
                    kernel_inv_t = kernel_inv[t][fi]

                    # Uncomment the following line for L2 variants of algorithms
                    # kernel_inv_t = torch.eye(kernel_inv_t.shape[0], device=kernel_inv_t.device)

                    # Calculate K_{t-1}^{-1} (m_t - m_{t-1})
                    kinvf_t = torch.squeeze(torch.matmul(kernel_inv_t, delta_logits[:,None]), dim=-1)

                    grad_t = torch.einsum('ij,i->j', jac_t, kinvf_t)
                    grad_t_sum.add_(grad_t)

            grad.add_(grad_t_sum)

        # Do gradient norm clipping
        clip_norm = self.defaults['grad_clip_norm']
        if clip_norm is not None:
            grad_norm = torch.norm(grad)
            grad_norm = 1.0 if grad_norm < clip_norm else grad_norm/clip_norm
            grad.div_(grad_norm)

        # Adam update equations
        exp_avg, exp_avg_sq = self.state['exp_avg'], self.state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = self.state['max_exp_avg_sq']
        self.state['step'] += 1
        exp_avg.mul_(beta1).add_(1-beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1-beta2, grad, grad)
        if amsgrad:
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = max_exp_avg_sq.sqrt().add_(self.param_groups[0]['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(self.param_groups[0]['eps'])

        bias_correction1 = 1 - beta1 ** self.state['step']
        bias_correction2 = 1 - beta2 ** self.state['step']
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        mu.addcdiv_(-step_size, exp_avg, denom)
        vector_to_parameters(mu, parameters)

        return loss_cur, preds_cur
