
import torch.nn as nn
import torch
import numpy as np
import dynamics
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

def get_W_gt(labels):
    n=labels.shape[0]
    W = torch.zeros(n, n)
    for i, yi in enumerate(labels):
        for j, yj in enumerate(labels[i+1:],i+1):
              W[i, j] = W[j, i] = yi==yj    
    W = W.cuda()
    W.requires_grad=False
    return W

def get_sim_pairs(labels):
  n= labels.shape[0]
  W = np.ones((n,n), dtype=(int,2))*-1
  for i, yi in enumerate(labels):
      for j, yj in enumerate(labels[i+1:],i+1):
          if(yi <= yj):
              W[i, j] = W[j, i] = (yi,yj)
          else:
              W[i, j] = W[j, i] = (yj,yi)
  return W

def thresh_func(x,thresh,tau):
    index_0=x<thresh-tau
    index_1=x>=thresh+tau/2
    index_between=torch.logical_and(x>=thresh-tau,x < thresh+tau/2)
    m=(thresh-tau)
    x[index_between]=(x[index_between].float()-m[index_between].float())/(1.5*tau)
    x[index_0]=0
    x[index_1]=1
    return x

def dcorr(x,y,device):
    
    x=torch.reshape(x,(1,1024))
    n=x.shape[1]
    xx=torch.matmul(torch.ones(x.shape,device=device).t(),x)
    yy=torch.matmul(torch.ones(x.shape,device=device).t(),x).t()
    dists_x=torch.abs(xx-yy)


    x=torch.reshape(y,(1,1024))
    xx=torch.matmul(torch.ones(x.shape,device=device).t(),x)
    yy=torch.matmul(torch.ones(x.shape,device=device).t(),x).t()
    dists_y=torch.abs(xx-yy)

    dists=dists_x
    row_means=torch.mean(dists, dim=1)
    col_means=torch.mean(dists, dim=0)
    row_means=row_means.unsqueeze(1)
    col_means=col_means.unsqueeze(0)
    full_mean=torch.mean(dists)
    res_x=dists-row_means-col_means+full_mean


    dists=dists_y
    row_means=torch.mean(dists, dim=1)
    col_means=torch.mean(dists, dim=0)
    row_means=row_means.unsqueeze(1)
    col_means=col_means.unsqueeze(0)
    full_mean=torch.mean(dists)
    res_y=dists-row_means-col_means+full_mean

    dcov2_xy = torch.sum(res_x * res_y)/float(n * n)

    dcov2_xx = torch.sum(res_x * res_x)/float(n * n)

    dcov2_yy = torch.sum(res_y * res_y)/float(n * n)

    dcorr = torch.sqrt(dcov2_xy)/torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

    return dcorr





class GTG(nn.Module):
    def __init__(self, total_classes, tol=-1., max_iter=5, sim='correlation',sim_model=None, set_negative='hard', mode='replicator', device='cuda:0'):
        super(GTG, self).__init__()
        self.m = total_classes
        self.tol = tol
        self.max_iter = max_iter
        self.mode = mode
        self.sim = sim
        self.set_negative = set_negative
        self.device = device
        self.thresh=.1
        self.sim_model=sim_model
      

    def _init_probs_uniform(self, labs, L, U):
        """ Initialized the probabilities of GTG from uniform distribution """
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = 1. / self.m
        ps[L, labs] = 1.

        # check if probs sum up to 1.
        assert torch.allclose(ps.sum(dim=1), torch.ones(n))
        return ps

    def _init_probs_prior(self, probs, labs, L, U):
        """ Initiallized probabilities from the softmax layer of the CNN """
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = probs[U, :]
        ps[L, labs] = 1.
        assert torch.allclose(ps.sum(dim=1), torch.ones(n).cuda())
        return ps

    def _init_probs_prior_only_classes(self, probs, labs, L, U, classes_to_use):
        """ Different version of the previous version when it considers only classes in the minibatch,
            surprisingly it works worse than the version that considers all classes """
        n = len(L) + len(U)
        ps = torch.zeros(n, self.m).to(self.device)
        ps[U, :] = probs[torch.meshgrid(torch.tensor(U), torch.from_numpy(classes_to_use))]
        ps[L, labs] = 1.
        ps /= ps.sum(dim=ps.dim() - 1).unsqueeze(ps.dim() - 1)
        return ps

    def set_negative_to_zero(self, W):
        return F.relu(W)

    def set_negative_to_zero_soft(self, W):
        """ It shifts the negative probabilities towards the positive regime """
        n = W.shape[0]
        minimum = torch.min(W)
        W = W - minimum
        W = W * (torch.ones((n, n)).to(self.device) - torch.eye(n).to(self.device))
        return W

    def _get_D(self, x):
        n = x.shape[0]
        D = torch.zeros(n, n)
        for i, xi in enumerate(x):
            for j, xj in enumerate(x[(i+1 ):], i+1):
                  D[i, j] = D[j, i] = torch.dist(xi,xj,2)
        return D
    
    def _get_cos_D(self, x):
        n = x.shape[0]
        D_cos = torch.zeros(n, n)
        cos=nn.CosineSimilarity(dim=0)
        for i, xi in enumerate(x):
            for j, xj in enumerate(x[(i+1 ):], i+1):
                  D_cos[i, j] = D_cos[j, i] = cos(xi,xj)
        return D_cos
      



    def _get_W(self, x, quantiles=None,Y=None,get_D=False,get_cos_D=False, get_both=False):

        
        
        if self.sim == 'correlation':
            x = (x - x.mean(dim=1).unsqueeze(1))
            norms_1 = x.norm(p=2,dim=1)
            W = torch.mm(x, x.t())/ torch.ger(norms_1, norms_1)



        elif self.sim == 'correlation_scaled':
            x = (x - x.mean(dim=1).unsqueeze(1))
            norms_1 = x.norm(p=2,dim=1)

            W = torch.mm(x, x.t()) / torch.ger(norms_1, norms_1)
            scales=last_linear_weights[Y,:][:,Y]
            bias=proxie_bias[Y,:][:,Y]
            bias=(bias+bias.t())/2
            scales=(scales+scales.t())/2

            W=W*scales+bias
  
        elif self.sim == 'd_correlation':
            n = x.shape[0]
            W = torch.zeros(n, n)
            D = torch.zeros(n, n)
            for i, xi in enumerate(x):
                for j, xj in enumerate(last_linear_weights[(i+1 ):], i+1):
                      W[i, j] = W[j, i] = dcorr(xi,xj,device='cuda:0')



        
        elif self.sim == 'cosine_extreme':

            A,B=torch.meshgrid(quantiles,quantiles)
            threshs=torch.min(A,B)

            x=F.normalize(x,p=2,dim=1)
            W = torch.mm(x, x.t())
            W = thresh_func(W,threshs,self.thresh)


        elif self.sim == 'cosine':

            x=F.normalize(x,p=2,dim=1)
            W = torch.mm(x, x.t())





        elif self.sim == 'learnt':
            n = x.shape[0]
            W = torch.zeros(n, n)
            D = torch.zeros(n, n)
            for i, xi in enumerate(x):
                for j, xj in enumerate(x[(i+1 ):], i+1):
                      W[i, j] = W[j, i] = self.sim_model(xi, xj)
  
        elif self.sim == 'hybrid':
            n = x.shape[0]
            y = (x - x.mean(dim=0).unsqueeze(0))
            W_learn = torch.zeros(n, n)
            D = torch.zeros(n, n)
            for i, xi in enumerate(y):
                for j, xj in enumerate(y[(i+1 ):], i+1):
                      W_learn[i, j] = W_learn[j, i] = self.sim_model(xi, xj)
            W_learn=W_learn.to('cuda:0')


            x = (x - x.mean(dim=1).unsqueeze(1))
            norms = x.norm(dim=1)
            W_corr = torch.mm(x, x.t()) / torch.ger(norms, norms)
            W=W_corr+W_learn
            W = W.cuda()
        if self.set_negative == 'hard':
            W = self.set_negative_to_zero(W.cuda())

        else:
            W = self.set_negative_to_zero_soft(W)
        if(get_D):
            return W, self._get_D(z)
        if(get_cos_D):
            return W, self._get_cos_D(z)
        if(get_both):
            return W, self._get_cos_D(z), self._get_D(z)
        else:
            return W
      


    def forward(self, fc7, num_points, labs, L, U,probs=None, quantiles=None, Y=None,classes_to_use=None):
        W = self._get_W(fc7,quantiles, Y)
        if type(probs) is type(None):
            ps = self._init_probs(labs, L, U).cuda()
        else:
            if type(classes_to_use) is type(None):
                ps = probs
                ps = self._init_probs_prior(ps, labs, L, U)
            else:
                ps = probs
                ps = self._init_probs_prior(ps, labs, L, U)
        ps = dynamics.dynamics(W, ps, self.tol, self.max_iter, self.mode)
        return ps, W