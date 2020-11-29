# this is a generalised one r active learning
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import numpy as np
import time
import pandas as pd
import glob
import random
import math
import argparse
from utils_haptic import *

# just to prevent warnings
labelled_trips, unlabelled_trips, val_trips, test_trips, pool_size = None, None, None, None, None

def train(model, device, optimiser, epochs, npool, r, LR, sampling, savename, initial_pool, fac, saver):
    print('training starts ..................................')
    # saving details
    save_dir = '/home/SharedData/priyadarshini/'
    test_list,train_list,loss_list,total_acc_list,val_acc_list = [],[],[],[],[]
    best_state = model.state_dict()

    for t in range(0,npool):
        local_loss_list,local_acc_list,local_val_list = [],[],[] # local parameter lists
        tick=time.time()
        model.load_state_dict(best_state) # load the previous best model
        optimiser = optim.Adam(model.parameters(), lr=LR, eps=1e-5, amsgrad=True)
        global labelled_trips, unlabelled_trips, val_trips, test_trips, pool_size # global params which are needed        
        print('\nrun:',r,'training:',t)
        test_acc = find_acc(model, test_trips, device)
        test_list.append(test_acc)
        train_acc = find_acc(model, labelled_trips, device)
        train_list.append(train_acc)
        print('Labelled:',len(labelled_trips), 'Unlabelled:', len(unlabelled_trips))
        print('Train accuracy:',train_acc,'Test accuracy:',test_acc)

        # saving the model this should work
        if ((t%10==0 or t==(npool-1)) and saver):
            print('saving the acc file.......................................')
            acc_dict = {'test_acc': test_list, 'train_acc': train_list}
            dft = pd.DataFrame(acc_dict)
            dft.to_hdf(save_dir+str(r)+savename+str(epochs)+sampling+str(fac)+'.h5', key='data')

        # choose the sampling method
        if(len(unlabelled_trips) > 0 and pool_size>0):
            if(sampling=='rnd'):
                index = find_index_rnd(model, unlabelled_trips, pool_size, device)
            elif(sampling=='us'):
                index = find_index_us(model, unlabelled_trips, pool_size, device)
            elif(sampling=='centroid'):
                index = find_index_centroid_fps(model, unlabelled_trips, pool_size, device, fac)
            elif(sampling=='us_centroid'):
                index = find_index_us_centroid_fps(model, unlabelled_trips, pool_size, device, fac)
            elif(sampling=='grad'):
                index = find_index_grad_fps(model, unlabelled_trips, pool_size, device, fac)
            elif(sampling=='us_grad'):
                index = find_index_us_grad_fps(model, unlabelled_trips, pool_size, device, fac)
            elif(sampling=='ecl'):
                index = find_index_ecl_fps(model, unlabelled_trips, pool_size, device, fac)
            elif(sampling=='us_ecl'):
                index = find_index_us_ecl_fps(model, unlabelled_trips, pool_size, device, fac)
            elif(sampling=='geometry'):
                index = find_index_geometry_fps(model, unlabelled_trips, pool_size, device, fac)
            elif(sampling=='us_geometry'):
                index = find_index_us_geometry_fps(model, unlabelled_trips, pool_size, device, fac)
                           
            elif(sampling=='badge'):
                index = find_index_badge(model, unlabelled_trips, pool_size, device,fac)
                
            elif(sampling=='us_grad_kmean'):
                index = find_index_us_grad_kmean(model, unlabelled_trips, pool_size, device,fac)
            elif(sampling=='us_ecl_kmean'):
                index = find_index_us_ecl_kmean(model, unlabelled_trips, pool_size, device,fac)
            elif(sampling=='us_geometry_kmean'):
                index = find_index_us_geometry_kmean(model, unlabelled_trips, pool_size, device,fac)
            elif(sampling=='us_centroid_kmean'):
                index = find_index_us_centroid_kmean(model, unlabelled_trips, pool_size, device,fac)              
            else:
                print('undefined sampling index, stopping')
            
            add_elems = unlabelled_trips[index]
            # delete those indices from unlabelled data first
            unlabelled_trips = np.delete(unlabelled_trips, index, axis=0)         
            # now add these to the labelled data 
            labelled_trips = np.concatenate((labelled_trips, add_elems), axis=0)

        prev_acc = 0.0 # now train it for epochs
        for e in range(epochs):
            epoch_loss = 0.0
            model.train()
            # kbatches and nbatches defined here
            kbatch = int(math.ceil(len(labelled_trips)/float(Kb)))
            index = np.arange(len(labelled_trips))
            random.shuffle(index, random.random) # shuffling before each and every epoch
            for i in range(kbatch):
                optimiser.zero_grad()                          
                labelled_trp_baz = labelled_trips[index[Kb*i:Kb*(i+1)]]
                # do forward
                margin = find_margin_list(model, labelled_trp_baz, device)
                loss = find_loss(margin)           
                epoch_loss+= loss.item()
                loss.backward()
                optimiser.step()
                                
            epoch_loss = epoch_loss/len(labelled_trips)
            local_loss_list.append(epoch_loss)         
            with torch.no_grad(): #validate here
                acc = find_acc(model, labelled_trips, device)
                val_margin = find_margin_list(model, val_trips, device)
                val_acc = torch.mean((val_margin>=0.0).float()).item()
                val_loss = find_loss(val_margin)
                local_acc_list.append(acc)
                local_val_list.append(val_acc)
                # print('epoch: ',e, 'train_acc:', acc, 'test_acc: ', val_acc)

            if(val_acc >= prev_acc):
                best_state = model.state_dict()
                prev_acc = val_acc
        print(acc, val_acc)

def main():
    # Training settings
    # argparse thingy
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000,help='nepochs')
    parser.add_argument("--LR", type=float, default=1e-5,help='learning rate')
    parser.add_argument("--sampling", type=str, default='rnd',help='sampling method')
    parser.add_argument("--m", type=int, default=0,help='no of split to train on')
    parser.add_argument("--n", type=int, default=5,help='no of split to train on')

    parser.add_argument("--data", type=str, default='haptics',help='dataset type')
    parser.add_argument("--initial_pool", type=int, default=600, help='initial_pool')
    parser.add_argument("--budget", type=int, default=500, help='budget')
    parser.add_argument("--fac",type=int,default=2,help='factor in some samplings')

    args = parser.parse_args()
    global labelled_trips, unlabelled_trips, val_trips, test_trips, pool_size
    nepochs = args.epochs
    pool_size = args.budget
    LR = args.LR
    sampling = args.sampling
    rval = args.n
    rlow = args.m
    data = args.data
    fac = args.fac
    model = PerceptNet().to(device)
    initial_pool = args.initial_pool
    saver = 1
    savename = 'haptic_'+'init_'+str(initial_pool)+'_inr_'+str(pool_size)+'_epoch_'
    for r in range(rlow, rval):
        state_dict = torch.load('models/haptic/intialstate_'+str(initial_pool)+'_'+str(r)+'.pt')
        model.load_state_dict(state_dict)
        optimiser = optim.Adam(model.parameters(), lr=LR, eps=1e-5, amsgrad=True)
        train_trips,val_trips,test_trips = train_list[r], val_list[r], test_list[r]
        labelled_trips, unlabelled_trips = train_trips[:initial_pool], train_trips[initial_pool:]
        npool = len(unlabelled_trips)//pool_size + 1
        print('npool:', npool)
        train(model, device, optimiser, nepochs, npool, r, LR, sampling, savename, initial_pool, fac, saver)

if __name__ == '__main__':
    main()
