import dataset
from dataset.Inshop import Inshop_Dataset
import torch
from collections import defaultdict
from combine_sampler import CombineSampler, CombineSamplerAdvanced, CombineSamplerSuperclass, CombineSamplerSuperclass2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from  matplotlib.lines import Line2D
import similarity
import sys

def create_loaders(data_root, num_classes, is_extracted, num_workers, num_classes_iter, num_elements_class, size_batch, dict_superclass=None, iterations_for_epoch=None):
    Dataset = dataset.Birds(
        root=data_root,
        labels=list(range(0, num_classes)),
        is_extracted=is_extracted,
        transform=dataset.utils.make_transform())

    Dataset_2 = dataset.Birds(
        root=data_root,
        labels=list(range(num_classes, 2*num_classes)),
        is_extracted=is_extracted,
        transform=dataset.utils.make_transform())

    ddict = defaultdict(list)
    for idx, label in enumerate(Dataset.ys):
        ddict[label].append(idx)

    list_of_indices_for_each_class = []
    for key in ddict:
        list_of_indices_for_each_class.append(ddict[key])
  
    ddict_2 = defaultdict(list)
    for idx, label in enumerate(Dataset_2.ys):
        ddict_2[label].append(idx)

    list_of_indices_for_each_class_2 = []
    for key in ddict_2:
        list_of_indices_for_each_class_2.append(ddict_2[key])
      
    dl_ev_2= torch.utils.data.DataLoader(
        Dataset_2,
        batch_size=size_batch,
        shuffle=False,
        sampler=CombineSampler(list_of_indices_for_each_class_2, num_classes_iter, num_elements_class),
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    dl_tr = torch.utils.data.DataLoader(
        Dataset,
        batch_size=size_batch,
        shuffle=False,
        sampler=CombineSampler(list_of_indices_for_each_class, num_classes_iter, num_elements_class),
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    dl_tr_superclass=None
    if(dict_superclass is not None):
        dl_tr_superclass = torch.utils.data.DataLoader(
            Dataset,
            batch_size=size_batch,
            shuffle=False,
            sampler=CombineSamplerSuperclass(list_of_indices_for_each_class, num_classes_iter, num_elements_class, dict_superclass, iterations_for_epoch),
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True
        )


    

    if data_root == 'Stanford':
        class_end = 2 * num_classes - 2
    else:
        class_end = 2 * num_classes

    dl_ev = torch.utils.data.DataLoader(
        dataset.Birds(
            root=data_root,
            labels=list(range(num_classes, class_end)),
            is_extracted=is_extracted,
            transform=dataset.utils.make_transform(is_train=False)
        ),
            batch_size=50,
            shuffle=False,
            num_workers=1,
            pin_memory=True
    )

    dl_finetune = torch.utils.data.DataLoader(
        dataset.Birds(
            root=data_root,
            labels=list(range(num_classes)),
            is_extracted=is_extracted,
            transform=dataset.utils.make_transform(is_train=False)
        ),
        batch_size=size_batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    dl_train_evaluate = torch.utils.data.DataLoader(
        dataset.Birds(
            root=data_root,
            labels=list(range(num_classes)),
            is_extracted=is_extracted,
            transform=dataset.utils.make_transform(is_train=False)
        ),
        batch_size=150,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    return dl_tr, dl_ev, dl_ev_2, dl_finetune, dl_train_evaluate, dl_tr_superclass

def create_loaders_Inshop(data_root, num_classes, is_extracted, num_workers, num_classes_iter, num_elements_class, size_batch, dict_class_distances=None, iterations_for_epoch=None):
    Dataset = Inshop_Dataset(
        root = data_root,
        mode = 'train',
        transform = dataset.utils.make_transform())

    ddict = defaultdict(list)
    for idx, label in enumerate(Dataset.ys):
        ddict[label].append(idx)

    list_of_indices_for_each_class = []
    for key in ddict:
        list_of_indices_for_each_class.append(ddict[key])
  
    dl_tr = torch.utils.data.DataLoader(
        Dataset,
        batch_size = 180,
        sampler=CombineSampler(list_of_indices_for_each_class, num_classes_iter, num_elements_class),
        shuffle = False,
        num_workers = num_workers,
        drop_last = True,
        pin_memory = True
    )

    dl_tr_ev = torch.utils.data.DataLoader(
    Dataset,
    batch_size=150,
    shuffle=False,
    num_workers=1,
    pin_memory=True
    )   

    query_dataset = Inshop_Dataset(
            root = data_root,
            mode = 'query',
            transform = dataset.utils.make_transform(is_train=False
    ))
    
    dl_query = torch.utils.data.DataLoader(
        query_dataset,
        batch_size = 180,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True
    )

    gallery_dataset = Inshop_Dataset(
            root = data_root,
            mode = 'gallery',
            transform = dataset.utils.make_transform(
                is_train = False
    ))
    
    dl_gallery = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size = 180,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True
    )
    return dl_tr, dl_query, dl_gallery,dl_tr_ev



def create_loaders_finetune(data_root, num_classes, is_extracted, num_workers, size_batch):

    if data_root == 'Stanford':
        class_end = 2 * num_classes - 2
    else:
        class_end = 2 * num_classes

    dl_ev = torch.utils.data.DataLoader(
        dataset.Birds(
            root=data_root,
            labels=list(range(num_classes, class_end)),
            is_extracted=is_extracted,
            transform=dataset.utils.make_transform(is_train=False)
        ),
            batch_size=150,
            shuffle=False,
            num_workers=1,
            pin_memory=True
    )

    dl_finetune = torch.utils.data.DataLoader(
        dataset.Birds(
            root=data_root,
            labels=list(range(num_classes)),
            is_extracted=is_extracted,
            transform=dataset.utils.make_transform(is_train=False)
        ),
        batch_size=size_batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


    return dl_ev, dl_finetune




def create_loaders_new(data_root, num_classes, is_extracted, dict_class_distances, num_classes_iter=0, num_elements_class=0, iterations_for_epoch=200):
    Dataset = dataset.Birds(
       root=data_root,
       labels=list(range(0, num_classes)),
       is_extracted=is_extracted,
       transform=dataset.utils.make_transform())
    ddict = defaultdict(list)
    for idx, label in enumerate(Dataset.ys):
       ddict[label].append(idx)
    list_of_indices_for_each_class = []
    for key in ddict:
       list_of_indices_for_each_class.append(ddict[key])
    # with open('list_of_indices_for_each_class_stanford.pickle', 'wb') as handle:
    #     pickle.dump(list_of_indices_for_each_class, handle, protocol=pickle.HIGHEST_PROTOCOL)
    list_of_indices_for_each_class = pickle.load(open('list_of_indices_for_each_class_stanford.pickle', 'rb'))
    dl_tr = torch.utils.data.DataLoader(
       Dataset,
       batch_size=num_classes_iter * num_elements_class,
       shuffle=False,
       sampler=CombineSamplerSuperclass2(list_of_indices_for_each_class, num_classes_iter, num_elements_class, dict_class_distances, iterations_for_epoch=iterations_for_epoch),
       num_workers=8,
       drop_last=True,
       pin_memory=True
    )
    return dl_tr


def get_labeled_and_unlabeled_points(labels, num_points_per_class, num_classes=100):
    labs, L, U = [], [], []
    labs_buffer = np.zeros(num_classes)
    num_points = labels.shape[0]
    for i in range(num_points):
        if labs_buffer[labels[i]] == num_points_per_class:
            U.append(i)
        else:
            L.append(i)
            labs.append(labels[i])
            labs_buffer[labels[i]] += 1
    return labs, L, U


def get_W_gt(labels):
    n=labels.shape[0]
    W = torch.zeros(n, n)
    for i, yi in enumerate(labels):
        for j, yj in enumerate(labels[i+1:],i+1):
              W[i, j] = W[j, i] = yi==yj    
    W = W.cuda()
    W.requires_grad=False
    return W


def debug_info(gtg, model):
    for name, param in gtg.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                # print(name, param.grad.data.norm(2))
                print(name, torch.mean(param.grad.data))
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                # print(name, param.grad.data.norm(2))
                print(name, torch.mean(param.grad.data))
    print("\n\n\n")


def plot_grad_flow(named_parameters,name):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    #plt.tight_layout()
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig('save_graphics/'+str(name))
    


def plot_grad_flow_v2(named_parameters,name, e):
    plt.figure(int(e))
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.25, color="r")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient per layer")
    plt.title("Gradient flow")
    #plt.tight_layout()
    plt.grid(True)
    plt.savefig('save_graphics/'+str(name)+'_epoch:'+str(e))


def plot_losses(loss_class, loss_group, path_to_save='save_graphics/', file_name='model'):
    plt.figure(100)
    plt.plot(range(len(loss_class)),loss_class, c="r", alpha=1, label="classification loss")
    plt.plot(range(len(loss_group)),loss_group, c="b", alpha=1, label="group loss")
    plt.grid(True)
    plt.title('plot of losses')
    plt.xlabel("average loss per epoch")
    plt.ylabel("loss")
    plt.legend(loc='upper right')
    plt.savefig(path_to_save+file_name+'_loss_curve')
    plt.clf()
    return None



def calc_anchors(model, dataloader, net_type):
    fc7s, L = [], []
    with torch.no_grad():
        for X, Y in dataloader:
            _, fc7 = model(X.cuda())
            fc7s.append(fc7.cpu())
            L.append(Y)
    fc7, Y = torch.cat(fc7s), torch.cat(L)
    print(fc7.shape)
    print(Y.shape)
    return torch.squeeze(fc7), torch.squeeze(Y)