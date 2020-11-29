# function base
# imports 
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.models as models
import torch.optim as optim
from scipy.spatial.distance import cdist
import numpy as np
import time
import pandas as pd
import glob
import random
import math

# gpu settings
use_cuda = torch.cuda.is_available()
print('gpu status ===',use_cuda)
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_list,test_list,val_list = [],[],[]
# load the data
max_run = 5
data_dir = './data/food_data/'
for n_run in range(max_run): #data loading loop
    train = np.load(data_dir+'train_trip_list_'+str(n_run)+'.npy')
    val = np.load(data_dir+'val_trip_list_'+str(n_run)+'.npy')
    test = np.load(data_dir+'test_trip_list_'+str(n_run)+'.npy')
    train_list.append(train)
    val_list.append(val)
    test_list.append(test)

feat = np.load(data_dir + 'feature.npy')
feat = torch.Tensor(feat).to(device)
Kb = 500

# model architecture
class PerceptNet(nn.Module): #this is for the linear one
    def __init__(self):
        super(PerceptNet, self).__init__()
        self.fcn1 = nn.Linear(6, 6)
        self.fcn2 = nn.Linear(6, 12)
        self.fcn3 = nn.Linear(12, 12, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fcn1(x))
        x = self.relu(self.fcn2(x))
        x = self.fcn3(x)
        return x
# training helper funcs
def find_dist_list(model, a_list, device):
    y0 = model(feat[a_list[:,0]].to(device))
    y1 = model(feat[a_list[:,1]].to(device))
    y2 = model(feat[a_list[:,2]].to(device))
    pdist = nn.PairwiseDistance(p=2)
    d01 = pdist(y0, y1)
    d02 = pdist(y0, y2)
    return d01,d02

def find_margin(d01,d02):
    return torch.pow(d02, 2) - torch.pow(d01, 2)

def find_margin_list(model, a_list, device):
    d01, d02 = find_dist_list(model, a_list, device)
    return find_margin(d01, d02)

def find_prob(d01,d02):
    mu = 1e-6
    num = d02 + mu
    den = d01 + d02 + 2*mu
    prob = num/den
    return prob

def find_loss(margin):
    return torch.mean(torch.exp(-1*margin))

def find_loss_list(model, a_list, device):
    margin = find_margin_list(model, a_list, device)
    return find_loss(margin)

def find_acc(model, a_list, device):
    # set them in testing mode 
    model.eval()
    with torch.no_grad():
    # pass all test features through it once
        d01,d02 = find_dist_list(model, a_list, device)
        margin = find_margin(d01,d02)
        acc = margin>=0.0
        acc = torch.mean(acc.float())
    return acc.item()
    
## AL helper functions ##

def gather_flat_grad_norm(grad):
    grad_vec = nn.utils.parameters_to_vector(grad)
    gradient_norm = torch.norm(grad_vec, p=2)   
    return gradient_norm

def find_trp_feature(model, index, a_list, device):
    y0 = model(feat[a_list[index,0]].to(device))
    y1 = model(feat[a_list[index,1]].to(device))
    y2 = model(feat[a_list[index,2]].to(device))
    pdist = nn.PairwiseDistance(p=2)
    d01 = pdist(y0, y1)
    d02 = pdist(y0, y2)
    mu = 1e-6
    num = d02 + mu
    den = d01 + d02 + 2*mu
    prob = num/den
    comb_feat1 = torch.cat([y0, y1, y2], 1)
    comb_feat2 = torch.cat([y0, y2, y1], 1)
    return comb_feat1, comb_feat2, prob

def find_trp_cent_feature(model, index, a_list, device):
    y0 = model(feat[a_list[index,0]].to(device))
    y1 = model(feat[a_list[index,1]].to(device))
    y2 = model(feat[a_list[index,2]].to(device))
    return (y0+y1+y2)/3.0

def find_center_radius_feature(model, index, a_list, device):
    euc = nn.PairwiseDistance(p=2)
    y0 = model(feat[a_list[index,0]].to(device))
    y1 = model(feat[a_list[index,1]].to(device))
    y2 = model(feat[a_list[index,2]].to(device))
    # now form a point tensor
    y = torch.stack([y0, y1, y2],dim=1)
    cent = torch.mean(y,dim=1)
    radius = torch.norm(torch.std(y,dim=1,unbiased=False), dim=-1)
    # radius = torch.norm(torch.var(y,dim=1), dim=-1)
    return cent.to(device), radius.to(device) # cent is N,d and radius is N

def find_entropy(p):
    val = -1*(p*torch.log(p) + (1-p)*torch.log(1-p))
    return val

def minmax_norm(a):
    # normalises a tensor between 0 to 1
    m = torch.min(a)
    return (a-m)/(torch.max(a) - m + 1e-8)

def maxmin(y,r,dist_mat):
    # y and r are two lists of indices we must find the min distance between them
    a = dist_mat[r,:]
    a = a[:,y]
    val = torch.min(a, 0)[0]
    index = torch.argmax(val)
    return index

def find_farthest_point(dist_mat, pool_size):
    n = len(dist_mat)
    R = list(np.arange(n))
    S = []
    max_ind = torch.argmax(dist_mat)
    a,b = max_ind//n, max_ind%n
    S.append(a.item())
    S.append(b.item())
    R.remove(a.item())
    R.remove(b.item())
    for j in range(2, pool_size):
        X = S
        Y = R
        cindex = maxmin(Y, X, dist_mat)
        index = R[cindex]
        S.append(index)
        R.remove(index)
    return np.array(S)

def paircosine(a,b):
    num = torch.matmul(a, b.t())
    anorm = torch.norm(a,dim=1).unsqueeze(1)
    bnorm = torch.norm(b,dim=1).unsqueeze(0)
    den = torch.matmul(anorm,bnorm) + 1e-8
    return torch.div(num,den)

def find_exp_grad(model, trp, device):
    model.zero_grad()
    d01, d02 = find_dist_list(model,trp,device)
    margin = find_margin(d01,d02)
    loss_abc = find_loss(margin)
    loss_acb = find_loss(-1*margin)
    prob_abc = find_prob(d01,d02)
    prob_acb = 1-prob_abc
    params = list(model.parameters())
    loss_grad_abc = torch.autograd.grad(loss_abc, params[-1], retain_graph=True)
    model.zero_grad()
    loss_grad_acb = torch.autograd.grad(loss_acb, params[-1])
    grad_abc = nn.utils.parameters_to_vector(loss_grad_abc)
    grad_acb = nn.utils.parameters_to_vector(loss_grad_acb)
    with torch.no_grad():
        entropy = find_entropy(prob_abc)
    grad = (prob_abc*grad_abc) + (prob_acb*grad_acb)
    return grad, entropy

## Active learning methods start from here  ##

def find_index_rnd(model, a_list, pool_size, device):
    t = time.time()
    if(len(a_list) <=pool_size):
        return np.arange(len(a_list))
    with torch.no_grad():
        indices = np.random.choice(len(a_list), pool_size, replace=False)
        print('sampling is rnd', (time.time() - t))
        return indices

def find_index_us(model, a_list, pool_size, device):
    t = time.time()
    if(pool_size>=len(a_list)):
        index = np.arange(len(a_list))
    else: 
        model.zero_grad()
        with torch.no_grad():
            d01, d02 = find_dist_list(model, a_list, device)
            prob_abc = find_prob(d01, d02)
            score = find_entropy(prob_abc)
        [sort_score, index] = torch.sort(score , descending=True)	
        index = index.detach().cpu().numpy()[:pool_size]
    print('sampling is us', (time.time()-t))
    return index

def find_index_centroid_fps(model, a_list, pool_size, device,fac):
    t = time.time()
    with torch.no_grad():
        euc = nn.PairwiseDistance(p=2)
        if(pool_size>=len(a_list)): # this is the base condition
            indices = np.arange(len(a_list))
            return indices
        batch_sz = fac*pool_size # take the factored valued
        us_index = find_index_us(model, a_list, batch_sz, device) 
        
        # find the centroid points and radius of the triplets
        cent_feat = find_trp_cent_feature(model, us_index, a_list, device).to(device)       
        # now make an euclidean dist matrix since cent is multidim
        fps_dist = cent_feat.unsqueeze(1) - cent_feat
        fps_dist = torch.norm(fps_dist,dim=-1) # d
        # find k farthest point from distance matrix
        k_index = find_farthest_point(fps_dist, pool_size)
        indices = us_index[k_index]
        print('sampling is centroid_fps', (time.time() - t))
    return indices
        
def find_index_us_centroid_fps(model, a_list, pool_size, device,fac):
    t = time.time()
    with torch.no_grad():
        euc = nn.PairwiseDistance(p=2)
        if(pool_size>=len(a_list)): # this is the base condition
            indices = np.arange(len(a_list))
            return indices
        batch_sz = fac*pool_size
        
        # code for finding us_score and us_index values
        d01, d02 = find_dist_list(model, a_list, device)
        prob_abc = find_prob(d01, d02)
        score = find_entropy(prob_abc)
        [us_score, us_index] = torch.sort(score, descending=True)	
        us_index = us_index.detach().cpu().numpy()[:batch_sz]
        us_score = us_score[:batch_sz]
        
        # find the centroid points and radius of the triplets
        cent_feat = find_trp_cent_feature(model, us_index, a_list, device).to(device)
        # now make an euclidean dist matrix since cent is multidim
        fps_dist = cent_feat.unsqueeze(1) - cent_feat
        fps_dist = torch.norm(fps_dist,dim=-1) # d
        fps_dist =  torch.matmul(us_score.view(-1,1),us_score.view(1,-1))*fps_dist
        # find k farthest point from distance matrix
        k_index = find_farthest_point(fps_dist, pool_size)
        indices = us_index[k_index]
        print('sampling is us_centroid_fps', (time.time() - t))
        return indices

def find_index_grad_fps(model, a_list, pool_size, device,fac):
    t = time.time()
    if(pool_size>=len(a_list)):
        indices = np.arange(len(a_list))
        return indices
    batch_sz = fac*pool_size    
    us_index = find_index_us(model, a_list, batch_sz, device) 

    params = list(model.parameters())
    count_param = params[-1].numel()

    alist_grad = torch.zeros([len(us_index), count_param]).to(device)
    for k in range(len(us_index)):
        ind = us_index[k]
        new_trp = np.array([a_list[ind]])
        alist_grad[k,:],_ = find_exp_grad(model, new_trp, device)      
    fps_dist = 1.0 - paircosine(alist_grad, alist_grad)
    # find k farthest point from distance matrix
    k_index = find_farthest_point(fps_dist, pool_size)
    ind = us_index[k_index]
    print('sampling is grad_fps', (time.time() - t))
    return ind

def find_index_us_grad_fps(model, a_list, pool_size, device,fac):
    t = time.time()
    if(pool_size>=len(a_list)):
        indices = np.arange(len(a_list))
        return indices
    batch_sz = fac*pool_size    
    params = list(model.parameters())
    count_param = params[-1].numel()
    # calculate the us score here
    with torch.no_grad():
        d01, d02 = find_dist_list(model, a_list, device)
        prob_abc = find_prob(d01, d02)
        score = find_entropy(prob_abc)
    [sort_score, index] = torch.sort(score , descending=True)   
    us_index = index.detach().cpu().numpy()[:batch_sz]
    us_score = sort_score[:batch_sz]

    alist_grad = torch.zeros([len(us_index), count_param]).to(device)
    for k in range(len(us_index)):
        ind = us_index[k]
        new_trp = np.array([a_list[ind]])
        alist_grad[k,:],_ = find_exp_grad(model, new_trp, device)      

    fps_dist = 1.0 - paircosine(alist_grad, alist_grad)
    fps_dist =  torch.matmul(us_score.view(-1,1),us_score.view(1,-1))*fps_dist
    k_index = find_farthest_point(fps_dist, pool_size)
    ind = us_index[k_index]
    print('sampling is us_grad_fps', (time.time() - t))
    return ind

def find_index_ecl_fps(model, a_list, pool_size, device,fac):
    t = time.time()
    with torch.no_grad():
        euc = nn.PairwiseDistance(p=2)
        if(pool_size>=len(a_list)): # this is the base condition
            indices = np.arange(len(a_list))
            return indices
        batch_sz = fac*pool_size
        us_index = find_index_us(model, a_list, batch_sz, device) 
        
        # find the concat feature
        cent_feat1, cent_feat2, p = find_trp_feature(model, us_index, a_list, device)
        q = 1-p
        # now add all combinations
        fps_dist11 = (p.unsqueeze(1)*p)*torch.norm((cent_feat1.unsqueeze(1) - cent_feat1), dim=-1)
        fps_dist12 = (p.unsqueeze(1)*q)*torch.norm((cent_feat1.unsqueeze(1) - cent_feat2), dim=-1)
        fps_dist21 = (q.unsqueeze(1)*p)*torch.norm((cent_feat2.unsqueeze(1) - cent_feat1), dim=-1)
        fps_dist22 = (q.unsqueeze(1)*q)*torch.norm((cent_feat2.unsqueeze(1) - cent_feat2), dim=-1)

        fps_dist = fps_dist11 + fps_dist12 + fps_dist21 + fps_dist22
        # find k farthest point from distance matrix
        k_index = find_farthest_point(fps_dist, pool_size)
        ind = us_index[k_index]
        print('sampling is ecl_fps', (time.time() - t))
        return ind

def find_index_us_ecl_fps(model, a_list, pool_size, device,fac):
    t = time.time()
    with torch.no_grad():
        euc = nn.PairwiseDistance(p=2)
        if(pool_size>=len(a_list)): # this is the base condition
            indices = np.arange(len(a_list))
            return indices
        batch_sz = fac*pool_size # take the factored valeues
        
        # code for finding us_score and us_index values
        d01, d02 = find_dist_list(model, a_list, device)
        prob_abc = find_prob(d01, d02)
        score = find_entropy(prob_abc)
        [us_score, us_index] = torch.sort(score, descending=True)	
        us_index = us_index.detach().cpu().numpy()[:batch_sz]
        us_score = us_score[:batch_sz]
        
        # find the concat feature
        cent_feat1, cent_feat2, p = find_trp_feature(model, us_index, a_list, device)
        q = 1-p
        # now add all combinations
        fps_dist11 = (p.unsqueeze(1)*p)*torch.norm((cent_feat1.unsqueeze(1) - cent_feat1), dim=-1)
        fps_dist12 = (p.unsqueeze(1)*q)*torch.norm((cent_feat1.unsqueeze(1) - cent_feat2), dim=-1)
        fps_dist21 = (q.unsqueeze(1)*p)*torch.norm((cent_feat2.unsqueeze(1) - cent_feat1), dim=-1)
        fps_dist22 = (q.unsqueeze(1)*q)*torch.norm((cent_feat2.unsqueeze(1) - cent_feat2), dim=-1)

        fps_dist = fps_dist11 + fps_dist12 + fps_dist21 + fps_dist22

        fps_dist =  torch.matmul(us_score.view(-1,1),us_score.view(1,-1))*fps_dist
        # find k farthest point from distance matrix
        k_index = find_farthest_point(fps_dist, pool_size)
        ind = us_index[k_index]
        print('sampling is us_ecl_fps', (time.time() - t))
        return ind
        
def find_index_geometry_fps(model, a_list, pool_size, device,fac):
    t = time.time()
    with torch.no_grad():
        if(pool_size>=len(a_list)): # this is the base condition
            indices = np.arange(len(a_list))
            return indices
        pfeat = model(feat) # proj feat
        batch_sz = fac*pool_size
        us_index = find_index_us(model, a_list, batch_sz, device) 
        # now we need to find the anc distance between among us_indices
        ancfeat = pfeat[a_list[us_index,0]]
        fps_dist = minmax_norm(torch.norm((ancfeat.unsqueeze(1) - ancfeat),dim=-1))
        resvec = ancfeat[a_list[us_index,1]] + ancfeat[a_list[us_index,2]] - 2*ancfeat[a_list[us_index,0]]

        fps_dist+= (1.0-paircosine(resvec,resvec))/2.0
        # find k farthest point from distance matrix
        k_index = find_farthest_point(fps_dist, pool_size)
        indices = us_index[k_index]
        print('sampling is geometry_fps', (time.time() - t))
        return indices
       
def find_index_us_geometry_fps(model, a_list, pool_size, device,fac):
    t = time.time()
    with torch.no_grad():
        if(pool_size>=len(a_list)): # this is the base condition
            indices = np.arange(len(a_list))
            return indices

        pfeat = model(feat)
        batch_sz = fac*pool_size

        # code for finding us_score and us_index values
        d01, d02 = find_dist_list(model, a_list, device)
        prob_abc = find_prob(d01, d02)
        score = find_entropy(prob_abc)
        [us_score, us_index] = torch.sort(score, descending=True)	
        us_index = us_index.detach().cpu().numpy()[:batch_sz]
        us_score = us_score[:batch_sz]

        ancfeat = pfeat[a_list[us_index,0]]
        #dist between anchors
        fps_dist = minmax_norm(torch.norm((ancfeat.unsqueeze(1) - ancfeat),dim=-1))
        resvec = ancfeat[a_list[us_index,1]] + ancfeat[a_list[us_index,2]] - 2*ancfeat[a_list[us_index,0]]
        new_cent = (ancfeat[a_list[us_index,1]] + ancfeat[a_list[us_index,2]] - 2*ancfeat[a_list[us_index,0]])/3.0 
        fps_dist+= (1.0-paircosine(resvec,resvec))/2.0
        fps_dist = torch.matmul(us_score.view(-1,1),us_score.view(1,-1))*fps_dist
        k_index = find_farthest_point(fps_dist, pool_size)
        indices = us_index[k_index]
        print('sampling is us_geometry_fps', (time.time() - t))
        return indices

def pairwise_euc(a,b, n):
    # pairwise euclidean distance find between cent_feat
    fps_dist = torch.zeros([n, n]).to(device)
    euc = nn.PairwiseDistance(p=2)
    for k in range(n):
        tiled = a[k].expand(n, -1)
        fps_dist[k, :] = euc(a, b)
    return fps_dist

def find_index_badge(model, a_list, pool_size, device,fac):
    t = time.time()
    if(pool_size>=len(a_list)):
        indices = np.arange(len(a_list))
        return indices

    params = list(model.parameters())
    count_param = params[-1].numel()

    alist_grad = torch.zeros([len(a_list), count_param]).to(device)
    for k in range(len(a_list)):
        new_trp = np.array([a_list[k]])
        alist_grad[k,:] = find_max_grad(model, new_trp, device)
    alist_grad = alist_grad.detach()
    dist = pairwise_euc(alist_grad, alist_grad, alist_grad.shape[0])
    print('computed the grad_list',time.time()-t)
    k_index = kmeans_plus_plus(dist, pool_size)
    print('sampling is grad_kmeans', (time.time() - t))
    return k_index

def kmeans_plus_plus(dist, pool_size):
    # same need to make R, S
    with torch.no_grad():
        n = dist.shape[0]
        R = list(np.arange(n))
        S = []
        a = np.random.randint(n, size=1)
        S.append(a[0])
        R.remove(a[0])
        # now the algo 
        for j in range(1, pool_size):
            cindex = sample_point(S,R,dist)
            X = S
            Y = R
            index = R[cindex]
            S.append(index)
            R.remove(index)
        ret = np.array(S)
        return ret

def find_max_grad(model, trp, device):
    model.zero_grad()
    d01, d02 = find_dist_list(model,trp,device)
    margin = find_margin(d01,d02)
    loss_abc = find_loss(margin)
    loss_acb = find_loss(-1*margin)
    prob_abc = find_prob(d01,d02)
    prob_acb = 1-prob_abc
    params = list(model.parameters())
    if(prob_abc >= 0.5):
        loss_grad = torch.autograd.grad(loss_abc, params[-1])
    else:
        loss_grad = torch.autograd.grad(loss_acb, params[-1])
    grad = nn.utils.parameters_to_vector(loss_grad)
    return grad

def cal_prob(ds):
    with torch.no_grad():
        dsq = torch.pow(ds,2)
        return dsq/(torch.sum(dsq))

def sample_point(S, R, dist):
    with torch.no_grad():
        distmat = dist[R,:]
        pdist = distmat[:,S]
        ds,_ = torch.min(pdist, dim=1)
        ds_prob = cal_prob(ds)
        ind = torch.multinomial(ds_prob, 1, replacement=False)
        return ind.item()

def find_index_us_centroid_kmean(model, a_list, pool_size, device,fac):
    t = time.time()
    with torch.no_grad():
        euc = nn.PairwiseDistance(p=2)
        if(pool_size>=len(a_list)): # this is the base condition
            indices = np.arange(len(a_list))
            return indices
        batch_sz = fac*pool_size
        
        # code for finding us_score and us_index values
        d01, d02 = find_dist_list(model, a_list, device)
        prob_abc = find_prob(d01, d02)
        score = find_entropy(prob_abc)
        [us_score, us_index] = torch.sort(score, descending=True)	
        us_index = us_index.detach().cpu().numpy()[:batch_sz]
        us_score = us_score[:batch_sz]
        
        # find the centroid points and radius of the triplets
        cent_feat = find_trp_cent_feature(model, us_index, a_list, device).to(device)
        # now make an euclidean dist matrix since cent is multidim
        fps_dist = cent_feat.unsqueeze(1) - cent_feat
        fps_dist = torch.norm(fps_dist,dim=-1) # d
        fps_dist =  torch.matmul(us_score.view(-1,1),us_score.view(1,-1))*fps_dist
        # find k farthest point using kmean clustering
        k_index = kmeans_plus_plus(fps_dist, pool_size)
        indices = us_index[k_index]
        print('sampling is us_centroid_kmean', (time.time() - t))
        return indices

def find_index_us_grad_kmean(model, a_list, pool_size, device,fac):
    t = time.time()
    if(pool_size>=len(a_list)):
        indices = np.arange(len(a_list))
        return indices
    batch_sz = fac*pool_size    
    params = list(model.parameters())
    count_param = params[-1].numel()
    # calculate the us score here
    with torch.no_grad():
        d01, d02 = find_dist_list(model, a_list, device)
        prob_abc = find_prob(d01, d02)
        score = find_entropy(prob_abc)
    [sort_score, index] = torch.sort(score , descending=True)   
    us_index = index.detach().cpu().numpy()[:batch_sz]
    us_score = sort_score[:batch_sz]

    alist_grad = torch.zeros([len(us_index), count_param]).to(device)
    for k in range(len(us_index)):
        ind = us_index[k]
        new_trp = np.array([a_list[ind]])
        alist_grad[k,:],_ = find_exp_grad(model, new_trp, device)      

    fps_dist = 1.0 - paircosine(alist_grad, alist_grad)
    fps_dist =  torch.matmul(us_score.view(-1,1),us_score.view(1,-1))*fps_dist
    k_index = kmeans_plus_plus(fps_dist, pool_size)
    ind = us_index[k_index]
    print('sampling is us_grad_mean', (time.time() - t))
    return ind
       
def find_index_us_geometry_kmean(model, a_list, pool_size, device,fac):
    t = time.time()
    with torch.no_grad():
        if(pool_size>=len(a_list)): # this is the base condition
            indices = np.arange(len(a_list))
            return indices

        pfeat = model(feat) # proj feat
        batch_sz = fac*pool_size

        # code for finding us_score and us_index values
        d01, d02 = find_dist_list(model, a_list, device)
        prob_abc = find_prob(d01, d02)
        score = find_entropy(prob_abc)
        [us_score, us_index] = torch.sort(score, descending=True)	
        us_index = us_index.detach().cpu().numpy()[:batch_sz]
        us_score = us_score[:batch_sz]

        ancfeat = pfeat[a_list[us_index,0]]
        #dist between anchors
        fps_dist = minmax_norm(torch.norm((ancfeat.unsqueeze(1) - ancfeat),dim=-1))
        resvec = ancfeat[a_list[us_index,1]] + ancfeat[a_list[us_index,2]] - 2*ancfeat[a_list[us_index,0]]
        new_cent = (ancfeat[a_list[us_index,1]] + ancfeat[a_list[us_index,2]] - 2*ancfeat[a_list[us_index,0]])/3.0 
        fps_dist+= (1.0-paircosine(resvec,resvec))/2.0
        fps_dist = torch.matmul(us_score.view(-1,1),us_score.view(1,-1))*fps_dist
        k_index = kmeans_plus_plus(fps_dist, pool_size)
        indices = us_index[k_index]
        print('sampling is us_geometry_kmean', (time.time() - t))
        return indices

def find_index_us_ecl_kmean(model, a_list, pool_size, device,fac):
    t = time.time()
    with torch.no_grad():
        euc = nn.PairwiseDistance(p=2)
        if(pool_size>=len(a_list)): # this is the base condition
            indices = np.arange(len(a_list))
            return indices
        batch_sz = fac*pool_size
        
        # code for finding us_score and us_index values
        d01, d02 = find_dist_list(model, a_list, device)
        prob_abc = find_prob(d01, d02)
        score = find_entropy(prob_abc)
        [us_score, us_index] = torch.sort(score, descending=True)	
        us_index = us_index.detach().cpu().numpy()[:batch_sz]
        us_score = us_score[:batch_sz]
        
        # find the concat feature
        cent_feat1, cent_feat2, p = find_trp_feature(model, us_index, a_list, device)
        q = 1-p
        # now add all combinations
        fps_dist11 = (p.unsqueeze(1)*p)*torch.norm((cent_feat1.unsqueeze(1) - cent_feat1), dim=-1)
        fps_dist12 = (p.unsqueeze(1)*q)*torch.norm((cent_feat1.unsqueeze(1) - cent_feat2), dim=-1)
        fps_dist21 = (q.unsqueeze(1)*p)*torch.norm((cent_feat2.unsqueeze(1) - cent_feat1), dim=-1)
        fps_dist22 = (q.unsqueeze(1)*q)*torch.norm((cent_feat2.unsqueeze(1) - cent_feat2), dim=-1)

        fps_dist = fps_dist11 + fps_dist12 + fps_dist21 + fps_dist22

        fps_dist =  torch.matmul(us_score.view(-1,1),us_score.view(1,-1))*fps_dist
        k_index = kmeans_plus_plus(fps_dist, pool_size)
        ind = us_index[k_index]
        print('sampling is us_ecl_kmean', (time.time() - t))
        return ind