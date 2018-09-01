#!/usr/bin/env python
# Poincare Embedding


import torch as th
from torch.optim.optimizer import Optimizer, required

spten_t = th.sparse.FloatTensor


def poincare_grad(p, d_p):
    """
    """
    if d_p.is_sparse:
        p_sqnorm = th.sum(
            p.data[d_p._indices()[0].squeeze()] ** 2, dim=1,
            keepdim=True
        ).expand_as(d_p._values())
        n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
        d_p = spten_t(d_p._indices(), n_vals, d_p.size())
    else:
        p_sqnorm = th.sum(p.data ** 2, dim=-1, keepdim=True)
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
    return d_p


def euclidean_grad(p, d_p):
    return d_p


def euclidean_retraction(p, d_p, lr):
    p.data.add_(-lr, d_p)


class RiemannianSGD(Optimizer):
    """Riemannian stochastic gradient descent.
    """

    def __init__(self, params, lr=required, rgrad=required, retraction=required):
        defaults = dict(lr=lr, rgrad=rgrad, retraction=retraction)
        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None):
        """
        """
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if lr is None:
                    lr = group['lr']
                d_p = group['rgrad'](p, d_p)
                group['retraction'](p, d_p, lr)

        return loss
