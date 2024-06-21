import numpy as np
import torch
import torch.nn as nn

def normalize(V):

    d = V.shape[0]
    norms = torch.norm(V, 2, dim=1)
    V[:, :] = V / norms.view(d, 1)

    return norms

def H(v):

    d = v.shape[0]
    I = torch.eye(d, d, device=v.device)
    v = v.reshape(d, 1)
    return I - 2 * v @ v.T

def Q(V):
    d = V.shape[0]
    norms = torch.norm(V.T, 2, dim=1)
    V = V / norms.view(1, d)
    M = torch.eye(d, d, device=V.device)
    for i in range(d):
        M = M @ H(V[:, i:i + 1])
    return M

#Fast Orthogonal Parameterization
def fasthpp(V, stop_recursion=1):
    d = V.shape[0]
    norms = torch.norm(V.T, 2, dim=1)
    V = V / norms.view(1, d)
    Y_ = V.clone().T
    W_ = -2 * Y_.clone()

    # Only works for powers of two.
    assert (d & (d - 1)) == 0 and d != 0, "d should be power of two.  "

    # Step 1: compute (Y, W) s by merging!
    k = 1
    for i, c in enumerate(range(int(np.log2(d)))):

        k_2 = k
        k *= 2

        m1_ = Y_.view(d // k_2, k_2, d)[0::2] @ torch.transpose(W_.view(d // k_2, k_2, d)[1::2], 1, 2)
        m2_ = torch.transpose(W_.view(d // k_2, k_2, d)[0::2], 1, 2) @ m1_

        W_ = W_.view(d // k_2, k_2, d)
        W_[1::2] += torch.transpose(m2_, 1, 2)
        W_ = W_.view(d, d)

        if stop_recursion is not None and c == stop_recursion: break
    X = torch.eye(d, d, device=V.device)
    # Step 2:
    if stop_recursion is None:
        return X + W_.T @ (Y_ @ X)
    else:
        # For each (W,Y) pair multiply with
        for i in range(d // k - 1, -1, -1):
            X = X + W_[i * k: (i + 1) * k].T @ (Y_[i * k: (i + 1) * k] @ X)
        return X

#Exemplery Usage: parameterize a NN layer
class OrthogonalWight(torch.nn.Module):
    #w<h
    def __init__(self, ci, co, k):
        super(OrthogonalWight, self).__init__()

        self.U = torch.nn.Parameter(torch.zeros((ci, ci)).normal_(0, 0.05))
        self.V = torch.nn.Parameter(torch.zeros((co * k * k, co * k * k)).normal_(0, 0.05))

        self.ci = ci
        self.co = co
        self.k = k

    def forward(self):
        with torch.no_grad():
            U = Q(self.U)
            V = Q(self.V)
            if self.ci < self.co * self.k * self.k:
                weight = nn.Parameter(U.mm(V[:self.ci, :]).view(1, self.ci, self.co, self.k, self.k))
            else:
                weight = nn.Parameter(U[:, :self.co * self.k * self.k].mm(V).view(1, self.ci, self.co,
                                                                                  self.k, self.k))

        return weight

class OrthogonalWightMLP(torch.nn.Module):
    #w<h
    def __init__(self, ci, co):
        super(OrthogonalWightMLP, self).__init__()

        self.U = torch.nn.Parameter(torch.zeros((ci, ci)).normal_(0, 0.05))
        self.V = torch.nn.Parameter(torch.zeros((co, co)).normal_(0, 0.05))

        self.ci = ci
        self.co = co

    def forward(self):

        U = Q(self.U)
        V = Q(self.V)
        if self.ci < self.co:
            weight = nn.Parameter(U.mm(V[:self.ci, :]))
        else:
            weight = nn.Parameter(U[:, :self.co].mm(V))

        return weight

# #Orthogonality Check
# d = 4
# V = torch.nn.Parameter(torch.zeros((d, d)).normal_(0, 0.05))
# print(fasthpp(V).mm(fasthpp(V).T))
#
# conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# ci, co, k, _ = conv.weight.shape
# print(ci, co, k)
# ortho = OrthogonalWight(ci, co, k)
# conv.weight = ortho()
#
# print(conv.weight.view(ci,co*k*k).T.mm(conv.weight.view(ci,co*k*k)))
#
# print(conv.weight.view(ci,co*k*k).mm(conv.weight.view(ci,co*k*k).T).diag().sum())



