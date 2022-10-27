import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import math
from QVSinit import *

argslayer = get_arguments()
if argslayer.ftype == 'googlenet':
    Fdim = 1024  # for googlenet features
elif argslayer.ftype == 'color' and argslayer.metric == 'OVP':
    Fdim = 861  # for seqdpp features OVP
elif argslayer.ftype == 'color' and argslayer.metric == 'Youtube':
    Fdim = 1581  # for seqdpp features Youtube
else:
    logger.info("Please specify dataset (OVP / Youtube) and feature type (googlenet / color).")

Fsize = 32
scalar = 1
segnum = 13
np.set_printoptions(threshold=np.inf)

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv3d') != -1:
        m.weight.data.copy_(torch.abs(m.weight))
        m.bias.data.copy_(torch.abs(m.bias))

    elif classname.find('Linear') != -1:
        m.weight.data.copy_(torch.abs(m.weight))
        m.bias.data.copy_(torch.abs(m.bias))

def Vsets3(InputMat,InputMatS):

    return(torch.norm(InputMat[:,None]-InputMatS,dim=2))

class layers(nn.Module):
    """father of the layer classes"""

    def __init__(self, in_dim, out_dim, bias=True):

        super(layers, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight = Parameter(torch.Tensor(in_dim, out_dim))
        if bias:
            self.bias = Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(0, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(0, stdv)

class Div(layers):

    def __init__(self, in_dim=scalar, out_dim=scalar):
        super(Div, self).__init__(in_dim, out_dim)

    def forward(self, seq, qvs_idx, sum_idx):

        inseq = seq
        seqlen = seq.shape[0]

        sum_idxs = sum_idx.squeeze().nonzero().squeeze()
        if sum_idxs.dim()==0:
            sum_idxs.unsqueeze_(0)

        seqLdist = Vsets3(inseq, inseq)
        seqLdistmax = seqLdist.max()
        seqL = seqLdistmax - seqLdist


        seqLz = seqL * sum_idx * sum_idx.transpose(0,1)
        divtmp = torch.det(seqLz[sum_idxs, :][:, sum_idxs])
        divtmp2 = divtmp.unsqueeze(0)
        logger.debug(divtmp.dim())

        if self.bias is not None:
            return (divtmp2.matmul(self.weight) + self.bias)
        else:
            return (divtmp2.matmul(self.weight))

class Cov(layers):

    def __init__(self, in_dim=scalar, out_dim=scalar):
        super(Cov, self).__init__(in_dim, out_dim)

    def forward(self, seq, qvs_idx, sum_idx):

        inseq = seq

        sum_idxs = sum_idx.squeeze().nonzero().squeeze()
        if sum_idxs.numel()==0:
            simcov4 = torch.tensor(0.).unsqueeze(0).cuda()
            if self.bias is not None:
                return (simcov4.matmul(self.weight) + self.bias)
            else:
                return (simcov4.matmul(self.weight))
        inseqS = inseq * sum_idx
        inseqQ = inseq * qvs_idx

        SeqDistS = Vsets3(inseqQ, inseqS)


        norm = SeqDistS.mean()

        simcov4tmp = SeqDistS[:, sum_idxs]

        if simcov4tmp.dim() == 1:
            simcov4tmp.unsqueeze_(1)

        simcov4 = torch.min(simcov4tmp, 1)[0].unsqueeze(1)
        simcov4[simcov4 > norm] = norm
        simcov4 = 1 - simcov4 / norm

        if self.bias is not None:

            return (simcov4.matmul(self.weight) + self.bias)
        else:

            return (simcov4.matmul(self.weight))


class Cov_2(layers):

    def __init__(self, in_dim=scalar, out_dim=scalar):
        super(Cov_2, self).__init__(in_dim, out_dim)

    def forward(self, seq, qvs_idx, sum_idx):

        inseq = seq

        sum_idxs = sum_idx.squeeze().nonzero().squeeze()
        inseqS = inseq * sum_idx
        inseqQ = inseq * qvs_idx

        SeqDistS = Vsets3(inseqQ, inseqS)


        norm = SeqDistS.mean()

        simcov4tmp = SeqDistS[:, sum_idxs]

        if simcov4tmp.dim() == 1:
            simcov4tmp.unsqueeze_(1)

        simcov4 = torch.mean(simcov4tmp, 1).unsqueeze(1)
        simcov4[simcov4 > norm] = norm
        simcov4 = 1 - simcov4 / norm

        if self.bias is not None:

            return (simcov4.matmul(self.weight) + self.bias)
        else:

            return (simcov4.matmul(self.weight))

