import torch
import os
import net
import torch.nn as nn


def load_net(dataset, net_type, nb_classes, embed=False, sz_embedding=512):

    if net_type == 'bn_inception':
        model = net.bn_inception(pretrained=True)
        model.last_linear = nn.Linear(1024, nb_classes)
        #model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

        if embed:
            model = net.Inception_embed(model, 1024, sz_embedding, num_classes=nb_classes)
            model.load_state_dict(torch.load(os.path.join('net', 'finetuned_cub_embedded_512_10_.pth')))

    elif net_type == 'resnet18':
        model = net.resnet18(pretrained=True)
        model.fc = nn.Linear(512, nb_classes)
        model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

    elif net_type == 'resnet34':
        model = net.resnet34(pretrained=True)
        model.fc = nn.Linear(512, nb_classes)
        model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

    elif net_type == 'resnet50':
        model = net.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, nb_classes)
        model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

    elif net_type == 'resnet101':
        model = net.resnet101(pretrained=True)
        model.fc = nn.Linear(2048, nb_classes)
        model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

    elif net_type == 'resnet152':
        model = net.resnet152(pretrained=True)
        model.fc = nn.Linear(2048, nb_classes)
        model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))

    elif net_type == 'densenet121':
        model = net.densenet121(pretrained=True)
        model.classifier = nn.Linear(1024, nb_classes)
        model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))
        # model.load_state_dict(torch.load('sop_resnet_new/Stanford_paramRes_16bit_densenet121_0.0002_5.2724883734490575e-12_10_6_2.pth'))

    elif net_type == 'densenet161':
        model = net.densenet161(pretrained=True)
        model.classifier = nn.Linear(2208, nb_classes)
        model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))
        # model.load_state_dict(torch.load('sop_resnet_new/Stanford_paramRes_16bit_densenet161_0.0002_5.2724883734490575e-12_10_6_2.pth'))

    elif net_type == 'densenet169':
        model = net.densenet169(pretrained=True)
        model.classifier = nn.Linear(1664, nb_classes)
        model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))
        # model.load_state_dict(torch.load('sop_resnet_new/Stanford_paramRes_16bit_densenet169_0.0002_5.2724883734490575e-12_10_6_2.pth'))

    elif net_type == 'densenet201':
        model = net.densenet201(pretrained=True)
        model.classifier = nn.Linear(1920, nb_classes)
        model.load_state_dict(torch.load('net/finetuned_' + dataset + '_' + net_type + '.pth'))
        # model.load_state_dict(torch.load('sop_resnet_new/Stanford_paramRes_16bit_densenet201_0.0002_5.2724883734490575e-12_10_6_2.pth'))
    return model

