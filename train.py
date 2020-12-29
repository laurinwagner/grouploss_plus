#from apex import amp
import logging, imp
import random
import os
import sys
import warnings
import argparse


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from  matplotlib.lines import Line2D

import gtg as it_process
from gtg import get_W_gt
from gtg import get_sim_pairs
import net
import data_utility
from data_utility import plot_grad_flow, plot_grad_flow_v2, plot_losses
import utils
from RAdam import RAdam

import argparse
import random
import similarity
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate


def get_sims(model,loader,q=.5, thresh=.12):
    gtg = it_process.GTG(args.nb_classes, max_iter=args.num_iter_gtg, sim='cosine', set_negative=args.set_negative, device=device).to(device)

    similarity_count={}
    for i in range(args.nb_classes):
        for j in range(i,args.nb_classes):
            similarity_count[(i,j)]=[]


    with torch.no_grad():
      for x, Y in loader:
            probs, fc7 = model(x.to(device))
            keys= get_sim_pairs(Y)
            values= gtg._get_W(fc7)
            for i in range(keys.shape[0]):
                for j in range(i+1, keys.shape[0]):
                    similarity_count[tuple(keys[i,j])].append(values[i,j].cpu().numpy())

    data_train_mean=np.zeros((args.nb_classes,args.nb_classes))
    #data_train_var=np.zeros((args.nb_classes,args.nb_classes))
    for i in range(args.nb_classes):
        item=similarity_count[(i,i)]
        if(len(item)>0):
            #data_train_var[i,i]=np.var(item)
            data_train_mean[i,i]=np.quantile(item,q)-.06


    return np.diag(data_train_mean)


def calc_anchors(model, dataloader, net_type):
    fc7s, L, means = [], [], []
    model.eval()

    with torch.no_grad():
        i=0
        for X, Y in dataloader:
            _, fc7 = model(X.cuda())
            fc7=F.normalize(fc7,p=2,dim=1)
            fc7s.append(fc7.cpu())
            L.append(Y)
        fc7, Y = torch.cat(fc7s), torch.cat(L)
        labels=torch.unique(Y, sorted=True)
        for label in labels:
            means.append(torch.mean(fc7[Y==label,:],dim=0,keepdim=True))
        means= torch.cat(means)
        means= F.normalize(means, p=2,dim=1)
        means.requires_grad=False
        model.train()
    return torch.squeeze(fc7), torch.squeeze(Y), torch.squeeze(means)


# param_groups = [
#     {'params': list(set(model.parameters()).difference(set(model.last_linear.parameters())))},
#     {'params': model.last_linear.parameters(), 'lr':float(args.lr) * 1},
# ]
# if args.loss == 'Proxy_Anchor':
#     param_groups.append({'params': criterion.proxies, 'lr':float(args.lr) * 100})

# def initialize_optimizer(model=None, lr_net=0.0002, anchor_speedup=1):
#     opt = RAdam([
#                 {'params': list(set(model.parameters()).difference(set(model.last_linear.parameters())))},
#                 {'params': list(set(model.last_linear.parameters())),'lr':lr_net*anchor_speedup},
#                   ],lr=lr_net ,weight_decay=args.weight_decay)
#     return opt

def initialize_optimizer(model=None, lr_net=0.0002, anchor_speedup=1):
    opt = RAdam([
                {'params': list(set(model.parameters()))}
                  ],lr=lr_net ,weight_decay=args.weight_decay)
    return opt


def rnd(lower, higher):
    exp = random.randint(-higher, -lower)
    base = 0.9 * random.random() + 0.1
    return base * 10 ** exp
  


warnings.filterwarnings("ignore")


class Hyperparameters():
    def __init__(self, dataset_name='cub'):
        self.dataset_name = dataset_name
        if dataset_name == 'cub':
            self.dataset_path = '../datasets/CUB_200_2011'
        elif dataset_name == 'cars':
            self.dataset_path = '../datasets/CARS'  #'../../datasets/CARS'

        elif dataset_name == 'Inshop':
            self.dataset_path = '../datasets/Inshop_Clothes'  #'../../datasets/CARS'
        else:
            self.dataset_path = '../datasets/Stanford'
        self.num_classes = {'cub': 100, 'cars': 98, 'Stanford': 11318,'Inshop': 3997}
        self.num_classes_iteration = {'cub': 6, 'cars':6 , 'Stanford': 10}
        self.num_elemens_class = {'cub': 9, 'cars': 7, 'Stanford': 6}
        self.get_num_labeled_class = {'cub': 2, 'cars': 3, 'Stanford': 2}
        self.learning_rate = 0.0002 #0.0002
        self.weight_decay = {'cub': 6.059722614369727e-06, 'cars': 4.863656728256105e-07, 'Stanford': 5.2724883734490575e-12}
        self.softmax_temperature = {'cub': 24, 'cars': 79, 'Stanford': 54}
        self.Beta = {'cub': 0.004, 'cars': 0.004, 'Stanford': 0.0005, 'Inshop': .002}
        self.Leak = {'cub': .75, 'cars': .0, 'Stanford': 1, 'Inshop': 1}
        self.Lambda = {'cub': .5, 'cars': .5, 'Stanford': .6,'Inshop': .57}

    def get_path(self):
        return self.dataset_path

    def get_number_classes(self):
        return self.num_classes[self.dataset_name]

    def get_number_classes_iteration(self):
        return self.num_classes_iteration[self.dataset_name]

    def get_number_elements_class(self):
        return self.num_elemens_class[self.dataset_name]

    def get_number_labeled_elements_class(self):
        return self.get_num_labeled_class[self.dataset_name]

    def get_learning_rate(self):
        return self.learning_rate

    def get_weight_decay(self):
        return self.weight_decay[self.dataset_name]

    def get_epochs(self):
        return 75

    def get_num_gtg_iterations(self):
        return 1

    def get_softmax_temperature(self):
        return self.softmax_temperature[self.dataset_name]

    def get_Beta(self):
        return self.Beta[self.dataset_name]

    def get_Lambda(self):
        return self.Lambda[self.dataset_name]

    def get_Leak(self):
        return self.Leak[self.dataset_name]


parser = argparse.ArgumentParser(description='Training inception V2' +
                                             ' (BNInception) on CUB-200-2011 (cub), CARS 196 (cars) and Stanford Online Products (Stanford) with The Group Loss as described in ' +
                                             '`The Group Loss for Deep Metric Learning.`')
dataset_name = 'Inshop'  # cub, cars or Stanford
parser.add_argument('--dataset_name', default=dataset_name, type=str, help='The name of the dataset')
hyperparams = Hyperparameters(dataset_name)
parser.add_argument('--cub-root', default=hyperparams.get_path(), help='Path to dataset folder')
parser.add_argument('--cub-is-extracted', action='store_true',
                    default=True, help='If `images.tgz` was already extracted, do not extract it again.' +
                                       ' Otherwise use extracted data.')
parser.add_argument('--embedding-size', default=1024, type=int, dest='sz_embedding', help='The embedding size')
parser.add_argument('--lr-net', default=0.0002, type=float, help='The learning rate')
parser.add_argument('--weight-decay', default=5e-14, type=float, help='The l2 regularization strength')  #hyperparams.get_weight_decay()
parser.add_argument('--nb_epochs', default=hyperparams.get_epochs(), type=int, help='Number of training epochs.')
parser.add_argument('--nb_workers', default=4, type=int, help='Number of workers for dataloader.')
parser.add_argument('--net_type', default='bn_inception', type=str, choices=['bn_inception', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
                                                                            'resnet18', 'resnet34', 'resenet50', 'resnet101', 'resnet152'],
                                                                            help='The type of net we want to use')
parser.add_argument('--sim_type', default='correlation', type=str, help='type of similarity we want to use')
####################################################################SAMPLING PARAMS##################################################################################################
parser.add_argument('--nb_classes', default=hyperparams.get_number_classes(), type=int,
                    help='Number of first [0, N] classes used for training and ' +
                         'next [N, N * 2] classes used for evaluating with max(N) = 100.')
parser.add_argument('--num_classes_iter', default=8, type=int,
                    help='Number of classes in the minibatch')
parser.add_argument('--num_elements_class', default=5, type=int,
                    help='Number of samples per each class')
parser.add_argument('--num_labeled_points_class', default=1, type=int,
                    help='Number of labeled samples per each class')
parser.add_argument('--superclass', default=0, type=int,
                    help='use superclass sampling')
##########################################################################INFERENCE PARAMS##############################################################################################
parser.add_argument('--Beta', default=hyperparams.get_Beta(), type=float,
                    help='Beta for beta-normalization on inference')
parser.add_argument('--Lambda', default=hyperparams.get_Lambda(), type=float,
                    help='controlling the mixture between max and average pooled features. lambda*(max)+(1-lambda)*avg')
parser.add_argument('--Leak', default=hyperparams.get_Leak(), type=int,
                    help='controlling the leakyness of final relu on inference')
parser.add_argument('--rerank', default=0, type=int,
                    help='use reranking during inference')
########################################################################################################################################################################
parser.add_argument('--set_negative', default='hard', type=str,
                    help='type of threshold we want to do'
                         'hard - put all negative similarities to 0'
                         'soft - subtract minus (-) negative from each entry')
parser.add_argument('--num_iter_gtg', default=1, type=int, help='Number of iterations we want to do for GTG')# hyperparams.get_num_gtg_iterations()
parser.add_argument('--embed', default=0, type=int, help='boolean controling if we want to do embedding or not')
parser.add_argument('--scaling_loss', default=1, type=int, dest='scaling_loss', help='Scaling parameter for the loss')
parser.add_argument('--scaling_w_loss', default=1, type=int, help='Scaling parameter for the loss')
parser.add_argument('--temperature', default=40, help='Temperature parameter for the softmax')
parser.add_argument('--temperature_2', default=1, help='Temperature parameter for the softmax')
parser.add_argument('--decrease_learning_rate', default=10, type=float,
                    help='Number to divide the learnign rate with')
parser.add_argument('--id', default=1, type=int,
                    help='id, in case you run multiple independent nets, for example if you want an ensemble of nets')
parser.add_argument('--is_apex', default=0, type=int,
                    help='if 1 use apex to do mixed precision training')
parser.add_argument('--thresh', default=.001, type=float,
                    help='threshold for similarity computation')


args = parser.parse_args()
print(args)
file_name = args.dataset_name + '_net_type:' + args.net_type + '_lr:' + str(args.lr_net) + '_weight_dec:' + str(args.weight_decay) + '_num_iter_gtg:' + str(args.num_iter_gtg)
plot_gradient_flow=False #if true we plot gradient flow for model
batch_size = args.num_classes_iter * args.num_elements_class

iterations=None
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# create folders where we save the trained nets and we put the results
save_folder_nets = 'save_trained_nets'
save_folder_results = 'save_results'
save_folder_graphics = 'save_graphics'
if not os.path.exists(save_folder_nets):
    os.makedirs(save_folder_nets)
if not os.path.exists(save_folder_results):
    os.makedirs(save_folder_results)
if not os.path.exists(save_folder_graphics):
    os.makedirs(save_folder_graphics)


# load models and optimizer
model = net.load_net(dataset=args.dataset_name, net_type=args.net_type, nb_classes=args.nb_classes, embed=args.embed, sz_embedding=args.sz_embedding)
model = model.to(device)
opt= initialize_optimizer(model, args.lr_net)

#continue training some model
#path_model='/content/gdrive/MyDrive/Models/inshop_best2.pth'
#model.load_state_dict(torch.load(os.path.join(path_model)),strict=True)

# set inference config
model.Lambda=hyperparams.get_Lambda()
model.Beta=hyperparams.get_Beta()
model.Leak=hyperparams.get_Leak()

#initialize iterative procedure
gtg = it_process.GTG(args.nb_classes, max_iter=args.num_iter_gtg, sim=args.sim_type, set_negative=args.set_negative, device=device).to(device)

# define the losses
criterion_NLL_group = nn.NLLLoss().to(device)
criterion_NLL_class = nn.NLLLoss().to(device)
criterion_MSE = nn.MSELoss().to(device)

#do training in mixed precision
if args.is_apex:
    model, opt = amp.initialize(model, opt, opt_level="O3")

# create loaders
dict_superclass=None
if(args.superclass):
    dict_superclass=np.load('dict_superclass_stanford.npy',allow_pickle=True).item()

if(dataset_name=='Inshop'):
    dl_tr,dl_query, dl_gallery,dl_tr_ev = data_utility.create_loaders_Inshop(args.cub_root, args.nb_classes, args.cub_is_extracted,
                                                                 args.nb_workers, args.num_classes_iter,args.num_elements_class,
                                                                 batch_size, iterations_for_epoch=iterations)

else:
    dl_tr, dl_ev, dl_ev_2, dl_finetune,dl_tr_ev, dl_tr_superclass = data_utility.create_loaders(args.cub_root, args.nb_classes, args.cub_is_extracted,
                                                                 args.nb_workers, args.num_classes_iter, args.num_elements_class,
                                                                 batch_size, iterations_for_epoch=iterations, dict_superclass=superclass)

# train and evaluate the net
best_accuracy = 0
scores = []
loss_class_agg=[]
loss_group_agg=[]
epoch=1
for e in range(1,args.nb_epochs +1 ):
  if e == 25:
      model.load_state_dict(torch.load(os.path.join(save_folder_nets, file_name + '.pth')))
      for g in opt.param_groups:
            g['lr'] = g['lr'] / (args.decrease_learning_rate)
  elif e == 50:
      model.load_state_dict(torch.load(os.path.join(save_folder_nets, file_name + '.pth')))
      for g in opt.param_groups:
            g['lr'] = g['lr'] / 2
      args.sim_type='cosine_extreme'
      args.num_labeled_points_class=0
      args.temperature=30
      gtg = it_process.GTG(args.nb_classes, max_iter=args.num_iter_gtg, sim=args.sim_type, set_negative=args.set_negative, device=device).to(device)
      gtg.thresh=args.thresh
      dl_tr, dl_ev, dl_ev_2, dl_finetune,dl_tr_ev, dl_tr_superclass = data_utility.create_loaders(args.cub_root, args.nb_classes, args.cub_is_extracted,
                                                                 args.nb_workers, 30, 3,
                                                                 90, iterations_for_epoch=iterations,dict_superclass=dict_superclass)
      if(dataset_name=='Stanford'):
          dl_tr = dl_tr_superclass
  elif e == 65:
      model.load_state_dict(torch.load(os.path.join(save_folder_nets, file_name + '.pth')))
      args.sim_type='cosine'
      gtg = it_process.GTG(args.nb_classes, max_iter=args.num_iter_gtg, sim=args.sim_type, set_negative=args.set_negative, device=device).to(device)
      for g in opt.param_groups:
          g['lr'] = 0

  if(args.sim_type=='cosine_extreme'):
      quantiles=torch.from_numpy(get_sims(model,dl_tr_ev,args.thresh)).to(device)





  i = 1
  running_class_loss = 0.0 
  running_group_loss = 0.0

  for x, Y in dl_tr:
      opt.zero_grad()
      Y = Y.to(device)
      probs,fc7 = model(x.to(device))
      if(args.sim_type=='cosine_extreme'):
        quantiles_batch=quantiles[Y]
      else:
        quantiles_batch=None
      labs, L, U = data_utility.get_labeled_and_unlabeled_points(labels=Y,
                                                                num_points_per_class=args.num_labeled_points_class,
                                                                num_classes=args.nb_classes)

      # compute softmax  and softmax with temperature
      probs_for_class = F.softmax(probs/ args.temperature_2)
      probs_for_gtg = F.softmax(probs/ args.temperature)

      # do GTG (iterative process)
      probs_for_gtg, W = gtg(fc7, fc7.shape[0], labs, L, U, probs_for_gtg, quantiles_batch, Y)
      probs_for_class = torch.log(probs_for_class+1e-14)
      probs_for_gtg = torch.log(probs_for_gtg + 1e-14)

      # compute the losses
      loss_group = criterion_NLL_group(probs_for_gtg, Y)
      loss_class = criterion_NLL_class(probs_for_class, Y)
      loss=loss_class + loss_group
      i = i+1

      #generate data to plot loss curves
      running_group_loss += loss_group.item()
      running_class_loss += loss_class.item()

      # check possible net divergence
      if torch.isnan(loss):
          print("We have NaN numbers, closing")
          print("\n\n\n")
          sys.exit(0)

      # backprop
      if args.is_apex:
          with amp.scale_loss(loss, opt) as scaled_loss:
              scaled_loss.backward()
      else:
          loss.backward()
          if(plot_gradient_flow):
            if(i%10==0):
                plot_grad_flow_v2(model.named_parameters(),'model_gradients_from_grouploss',e)

      #torch.nn.utils.clip_grad_value_(model.parameters(), 10)
      opt.step()

  loss_group_agg.append(running_group_loss/i)
  loss_class_agg.append(running_class_loss/i)
  #print(torch.sum(model.last_linear.weight[model.last_linear.weight.data<0]))

#compute recall and NMI at the end of each epoch (for Stanford NMI takes forever so skip it)
  with torch.no_grad():
      logging.info("**Evaluating...**")
      if dataset_name == 'Inshop':
                recall = utils.evaluate_Inshop(model, dl_query, dl_gallery,rerank=args.rerank)
                nmi=-1
      else:
        print('evaluating')
        nmi, recall = utils.evaluate(model, dl_ev, args.nb_classes,net_type=args.net_type, dataroot=args.dataset_name,rerank=args.rerank)
        print( 'nmi:'+ str(nmi) + '  recall:' + str(recall)) 
      scores.append((nmi, recall))
      model.current_epoch = e
      print(recall[0])
      if recall[0] >= best_accuracy:
          best_accuracy = recall[0]
          torch.save(model.state_dict(), os.path.join(save_folder_nets, file_name + '.pth'))

  #plot the losses
  plot_losses(loss_class_agg, loss_group_agg,path_to_save=save_folder_graphics,file_name='losses')


with open(os.path.join(save_folder_results, file_name + '.txt'), 'a+') as fp:
    fp.write(file_name + "\n")
    fp.write(str(args))
    fp.write('\n')
    fp.write('\n'.join('%s %s' % x for x in scores))
    fp.write("\n\n\n")