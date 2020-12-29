import torch
import torch.nn as nn

class learned_similarity_1(nn.Module):
    def __init__(self, in_size=1024):
        super(learned_similarity_1, self).__init__()
        self.lin = nn.Linear(in_size, 512)
        self.lin2= nn.Linear(512,256)
        self.lin3= nn.Linear(256,1)
        self.relu1= nn.ReLU(False)
        self.relu2=nn.ReLU(False)
        self.relu3=nn.ReLU(False)
        self.current_epoch=0
        self.sigmoid = nn.Sigmoid()

    def forward(self, xi, xj):
        out = torch.mul(xi/torch.norm(xi,p=2), xj/torch.norm(xj,p=2))
        #out= torch.cat((xi, xj), 0)
        #out= torch.exp(torch.abs(xi-xj))
        out = self.lin(out)
        out= self.relu1(out)
        out= self.lin2(out)
        out= self.relu2(out)
        out= self.lin3(out)
        out = self.sigmoid(out)
        return out


class learned_similarity_2(nn.Module):
    def __init__(self, in_size=1024):
        super(learned_similarity_2, self).__init__()
        self.lin = nn.Linear(in_size, 256)
        self.lin2 = nn.Linear(256, 1)
        self.relu1= nn.ReLU(False)
        self.relu2=nn.ReLU(False)
        self.current_epoch=0
        self.sigmoid = nn.Sigmoid()

    def forward(self, xi, xj):
        out = torch.mul(xi/torch.norm(xi,p=2), xj/torch.norm(xj,p=2))
        out = self.lin(out)
        out= self.relu1(out)
        out= self.lin2(out)
        out = self.sigmoid(out) 
        return out

class learned_similarity_3(nn.Module):
    def __init__(self, in_size=1024):
        super(learned_similarity_3, self).__init__()
        self.lin = nn.Linear(in_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU(False)
        self.current_epoch=0

    def forward(self, xi, xj):
        out = torch.mul(xi/torch.norm(xi,p=2), xj/torch.norm(xj,p=2))
        out = self.lin(out)
        out= self.sigmoid(out)
        return out

class learned_similarity_4(nn.Module):
    def __init__(self, in_size=2048):
        super(learned_similarity_4, self).__init__()
        self.lin = nn.Linear(in_size, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3= nn.Linear(512,200)
        self.lin4= nn.Linear(200,1)
        self.relu1= nn.ReLU(False)
        self.relu2=nn.ReLU(False)
        self.relu3=nn.ReLU(False)
        self.current_epoch=0
        self.sigmoid = nn.Sigmoid()

    def forward(self, xi, xj):
        out= torch.cat((xi, xj), 0)
        #out= torch.exp(torch.abs(xi-xj))
        out = self.lin(out)
        out= self.relu1(out)
        out= self.lin2(out)
        out= self.relu2(out)
        out= self.lin3(out)
        out=self.relu3(out)
        out=self.lin4(out)
        out = self.sigmoid(out)
        return out




class learned_similarity_5(nn.Module):
    def __init__(self, in_size=1024):
        super(learned_similarity_5, self).__init__()
        self.lin = nn.Linear(in_size, 1024)
        self.lin2 = nn.Linear(1024, 512)
        self.lin3= nn.Linear(512,200)
        self.lin4= nn.Linear(200,1)
        self.relu1= nn.ReLU(False)
        self.relu2=nn.ReLU(False)
        self.relu3=nn.ReLU(False)
        self.current_epoch=0
        self.tanh = nn.Tanh()

    def forward(self, xi, xj):
        out = torch.abs(xi - xj)
        #out= torch.cat((xi, xj), 0)
        #out= torch.exp(torch.abs(xi-xj))
        out = self.lin(out)
        out= self.relu1(out)
        out= self.lin2(out)
        out= self.relu2(out)
        out= self.lin3(out)
        out=self.relu3(out)
        out=self.lin4(out)
        out = self.tanh(out)
        return out


class learned_similarity_6(nn.Module):
    def __init__(self, in_size=1024):
        super(learned_similarity_6, self).__init__()
        self.lin = nn.Linear(in_size, 256)
        self.lin2 = nn.Linear(256, 1)
        self.relu1= nn.ReLU(False)
        self.current_epoch=0
        self.tanh = nn.Tanh()

    def forward(self, xi, xj):
        out = torch.abs(xi - xj)/torch.dist(xi,xj,2)
        out = self.lin(out)
        out= self.relu1(out)
        out= self.lin2(out)
        out = self.tanh(out) 
        return out

class learned_similarity_7(nn.Module):
    def __init__(self, in_size=1024):
        super(learned_similarity_7, self).__init__()
        self.lin = nn.Linear(in_size, 1)
        self.tanh = nn.Tanh()
        self.current_epoch=0

    def forward(self, xi, xj):
        out = torch.abs(xi - xj)
        out = self.lin(out)
        out= self.tanh(out)
        return out

class learned_similarity_8(nn.Module):
    def __init__(self, in_size=1024):
        super(learned_similarity_8, self).__init__()
        self.lin = nn.Linear(1, 1)
        self.lin2=nn.Linear(1,1)
        self.tanh=nn.Tanh()
        self.sigmoid=nn.Sigmoid()

    def forward(self, xi, xj):
        out = torch.reshape(torch.dist(xi,xj,2),(1,1))
        out=self.lin2(out)
        out=self.tanh(out)
        out =self.lin(out)
        out = self.sigmoid(out)
        return out

class learned_similarity_9(nn.Module):
    def __init__(self, in_size=1024):
        super(learned_similarity_9, self).__init__()
        self.lin = nn.Linear(in_size, 1)
        torch.nn.init.ones_(self.lin.weight)
        torch.nn.init.zeros_(self.lin.bias)
        self.sigmoid = nn.Sigmoid()
        self.relu1= nn.ReLU(False)
        self.current_epoch=0

    def forward(self, xi, xj):
        out = torch.mul(xi,xj)
        out = self.lin(out)
        out = self.relu1(out)
        out= torch.sqrt(torch.sqrt(out))
        return out

class learned_similarity_10(nn.Module):
    def __init__(self, in_size=1024):
        super(learned_similarity_10, self).__init__()
        self.lin = nn.Linear(98,in_size)

    def forward(self, xi, xj):
        xi=(xi-torch.mean(xi))/torch.norm(xi-torch.mean(xi), p=2)
        xj=(xj-torch.mean(xj))/torch.norm(xj-torch.mean(xj), p=2)
        out = torch.mul(xi,xj)
        out = self.lin(out)
        return out


