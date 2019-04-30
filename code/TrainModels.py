import numpy as np
import os

import torch 
import torch.nn as nn
from torch.autograd import Variable 
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

from qpth.qp import SpQPFunction, QPFunction
import cvxpy as cp

from DataGenerator import * 

np.random.seed(7)

# return train,validation and test 
def data_loader(data,target):
    n,d=data.shape
    nTrain=int(0.8*n)
    nVal=int(0.1*n)
    nTest=n-nTrain-nVal
    train_data=data[0:nTrain]
    train_target=target[0:nTrain]

    val_data=data[nTrain:nTrain+nVal]
    val_target=target[nTrain:nTrain+nVal]

    test_data=data[nTrain+nVal:]
    test_target=target[nTrain+nVal:]

    return train_data, train_target, val_data, val_target, test_data, test_target

def data_trans(init,final):
    data=torch.autograd.Variable(torch.from_numpy(init))
    target=torch.autograd.Variable(torch.from_numpy(final))
    data=data.float()
    target=target.float()
    
    return data, target

def train_model(data,target,model,loss_fn,optimizer,nEpoch,bach_size=20):
    train_data, train_target, val_data, val_target,test_data, test_target=data_loader(data,target)
    nBach=int(len(train_data)/bach_size)

    for n in range(nEpoch):
        count=0
        model.train()
        for t in range(nBach):
            batch_data=train_data[t*bach_size:(t+1)*bach_size]
            batch_target=train_target[t*bach_size:(t+1)*bach_size]
            y_pred = model(batch_data)
#             loss = loss_fn(y_pred, batch_target)+0.01*(torch.sum(batch_data[:,[4,5,6,7]]**2,dim=1)-torch.sum(y_pred[:,[4,5,6,7]]**2,dim=1))**2
            loss=loss_fn(y_pred,batch_target)
            count+=loss
            optimizer.zero_grad()
#             loss.mean().backward()
            loss.backward()
            optimizer.step()
        
        model.eval()
        y_test = model(test_data)
        test_loss = loss_fn(y_test, test_target)

    print('episode: {}, train loss: {}, test loss: {}'.format(n,count.mean()/nBach,test_loss.item()))
        
    
    return model

# input (8): (x1p, y1p, r1p, m1p, vx1p, vx2p, vy1p, vy2p)
# output (8): (x1p, x2p, y1p, y2p, vx1p, vx2p, vy1p, vy2p)
class ball_ball_update_net(nn.Module):
    def __init__(self):
        super(ball_ball_update_net,self).__init__()
        
        self.fc1=nn.Linear(8,2000)
        self.fc2=nn.Linear(2000,2000)
        self.fc3=nn.Linear(2000,8)
       
    
    def forward(self,x):
        x=F.relu(self.fc1(x)) 
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        # x=e_con(x)
        return x

# input (8): (x1p, y1p, r1p, m1p, vx1p, vx2p, vy1p, vy2p)
# output (8): (x1p, x2p, y1p, y2p, vx1p, vx2p, vy1p, vy2p)
class ball_ball_update_opt_net(nn.Module):
    def __init__(self, nFeatures=8, nHidden=500, nCls=8, neq=2, Qpenalty=10, eps=1e-4):
        super(ball_ball_update_opt_net,self).__init__()

        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.nCls = nCls
        
        self.fc0=nn.Linear(nFeatures,nFeatures)
        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2=nn.Linear(nHidden,nHidden)
        self.fc3=nn.Linear(nHidden,nHidden)
        self.fc4 = nn.Linear(nHidden, nCls)
        self.fc5=nn.Linear(nFeatures,nCls)
        
        self.Q = Parameter(torch.eye(nFeatures).double()) # will change
        self.G = Variable(torch.zeros(nFeatures,nFeatures).double())
        self.h = Variable(torch.zeros(nFeatures).double())
#         self.A = Parameter(torch.rand(neq,nHidden).double()) 
        # momentum conservation: Az=b
#         self.A=Variable(torch.zeros(2,nFeatures).double())
#         self.b = Parameter(torch.ones(self.A.size(0)).double())
#         self.p=Parameter(-torch.ones(nFeatures).double())
        self.neq = neq

    def forward(self, x):
        nBatch = x.size(0)

        # QP-FC
        x = x.view(nBatch, -1)
#         print(x.size())
        x = self.fc0(x)

        Q = self.Q.unsqueeze(0).expand(nBatch, self.Q.size(0), self.Q.size(1))
        p = -x.view(nBatch,-1)
#         p=self.p.unsqueeze(0).expand(nBatch,self.p.size(0))
        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0))
       
#         A=torch.cat([torch.tensor([[0,0,0,0,0,0,0,0,x[i,6],x[i,7],0,0],[0,0,0,0,0,0,0,0,0,0,x[i,6],x[i,7]] for i in range(nBatch)])],0)
        A=torch.cat(([torch.tensor([[0,0,0,0,x[i,3],1,0,0],[0,0,0,0,0,0,x[i,3],1]]).unsqueeze(0).double() for i in range(nBatch)]),0).double()
        # momentum: m1*vx1+m2*vx2=m1*vx1'+m2*vx2'
       
#         A = self.A.unsqueeze(0).expand(nBatch, self.A.size(0), self.A.size(1))
        
#         b = self.b.unsqueeze(0).expand(nBatch, self.b.size(0))
#         b=torch.stack((x[:,5]+x[:,6],x[:,7]+x[:,8]),dim=1).double()
		# b=[m1*vx1+m2*vx2, m1*vy1+m2*vy2]
        b=torch.stack((x[:,3]*x[:,4]+x[:,5],x[:,3]*x[:,6]+x[:,7]),dim=1).double()
        x = QPFunction(verbose=False)(Q, p.double(), G, h, A, b).float()
        
        x = F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
       
        return x

# input (5): (x1, y1, r1, vx1, vy1)
# output (5): one hot encode of collision wall
class ball_wall_detect_net(nn.Module):
    def __init__(self):
        super(ball_wall_detect_net,self).__init__()
        
        self.fc1=nn.Linear(5,100)
        self.fc2=nn.Linear(100,100)
        self.fc3=nn.Linear(100,5)
    
    def forward(self,x):
        x=F.relu(self.fc1(x)) 
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        # x=F.softmax(self.fc3(x),dim=1)

        return x

# input: (x1p, y1p, r1p, vx1p, vy1p)
# output: 0: no collision, 1: collision 
class ball_ball_detect_net(nn.Module):
    def __init__(self):
        super(ball_ball_detect_net,self).__init__()
        
        self.fc1=nn.Linear(5,200)
        self.fc2=nn.Linear(200,200)
        self.fc3=nn.Linear(200,1)
    
    def forward(self,x):
        x=F.relu(self.fc1(x)) 
        x=F.relu(self.fc2(x))
        x=torch.sigmoid(self.fc3(x))

        return x

if __name__ == '__main__':

	"""
	# train ball ball update model
	initials,finals=ball_ball_update_data(n_sample=10000,dt=1)
	print(initials.shape,finals.shape)
	data,target=data_trans(initials,finals)
	model=ball_ball_update_net()
	loss_fn = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(),lr=3e-3, weight_decay=4e-4)
	model=train_model(data,target,model,loss_fn,optimizer,nEpoch=500,bach_size=100)
	torch.save(model.state_dict(), './saved_models/model_bb_update')
	print('model ball-ball update saved')
	"""

	
	# train ball ball update opt model
	initials,finals=ball_ball_update_data(n_sample=20000,dt=1)
	print(initials.shape,finals.shape)
	data,target=data_trans(initials,finals)
	model=ball_ball_update_opt_net()
	loss_fn = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(),lr=3e-3, weight_decay=4e-4)
	model=train_model(data,target,model,loss_fn,optimizer,nEpoch=300,bach_size=100)
	torch.save(model.state_dict(), './saved_models/model_bb_opt_update')
	print('model ball-ball opt update saved')

	
	"""
	# train ball wall detection model
	initials,finals=ball_wall_detect_data(dt=1,width=1,n_sample=10000)
	# finals=np.argmax(finals,axis=1)
	print(initials.shape,finals.shape)
	data,target=data_trans(initials,finals)
	model=ball_wall_detect_net()
	loss_fn = nn.MultiLabelSoftMarginLoss()
	optimizer = optim.Adam(model.parameters(),lr=3e-3, weight_decay=4e-4)
	train_model(data,target,model,loss_fn,optimizer,nEpoch=500,bach_size=200)
	torch.save(model.state_dict(), './saved_models/model_bw_detect')
	print(model(data[1:3]))
	print('model ball-wall detect saved')
	"""

	"""
	# train ball ball detection model
	initials,finals=ball_ball_detect_data(dt=1,width=1,n_sample=10000)
	print(initials.shape,finals.shape)
	data,target=data_trans(initials,finals)
	model=ball_ball_detect_net()
	loss_fn = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(),lr=3e-3, weight_decay=4e-4)
	train_model(data,target,model,loss_fn,optimizer,nEpoch=300,bach_size=200)
	torch.save(model.state_dict(), './saved_models/model_bb_detect')
	print('model ball-ball detect saved')
	"""
	
	





    
   
    
