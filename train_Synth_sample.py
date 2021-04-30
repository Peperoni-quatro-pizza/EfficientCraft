import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import scipy.io as scio
import argparse
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import re
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp

from math import exp, gamma 
from data_loader_sample import SampleDataset , ToTensor, Resize, RandomCrop, normalize
from Gaussian import main 
from data_utils import generate_gt
from Loss_sample import OHEMloss

from collections import OrderedDict

from PIL import Image
from torchvision.transforms import transforms
from craft import CRAFT
from torch.autograd import Variable
from multiprocessing import Pool

import random

random.seed(10)

trans = transforms.Compose([RandomCrop(scale=0.25) , normalize(), Resize() ,ToTensor()])

if __name__ == '__main__': 

    sample_dataset = SampleDataset(image_folder='/root/data/SynthText' ,
                                    imnames= '/root/data/SynthText/imnames.npy',
                                    charBB = '/root/data/SynthText/charBB.npy',
                                    aff_charBB='/root/data/SynthText/aff_charBB.npy',
                                    transform=trans)  # ->  Augmentation 추가하자 

    sample_train_loader = torch.utils.data.DataLoader(
        sample_dataset,
        batch_size = 32,  
        shuffle = False, 
        num_workers = 0,
        drop_last = True,
        pin_memory = True,
        )  





    # sample_train_loader = torch.utils.data.DataLoader(
    #     sample_dataset,
    #     batch_size = 32,  
    #     shuffle = True, 
    #     num_workers = 0,
    #     drop_last = True,
    #     pin_memory = True)  




    scaler = torch.cuda.amp.GradScaler()
    net = CRAFT()
    optimizer = optim.Adam(net.parameters(), lr = 1e-4 )
    net = net.cuda()

    #DataParallel
    net = torch.nn.DataParallel(net, device_ids = [0,1]).cuda()

    #Distributed Parallel
    #net = DistributedDataParallel(net, device_ids= [0,1]).cuda()

    cudnn.benchmark = True

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)

    #criterion = OHEMloss()
    criterion = nn.MSELoss() # -> custom Loss : Online hard negative mining 구현해야한다. 

    net.train()

    loss_time = 0 
    loss_value = 0
    compare_loss = 1

    for epoch in range(10):
        loss_value = 0 

        st = time.time()
        for index, (image , region_label , affinity_label) in enumerate(sample_train_loader):
        
            image = Variable(image.type(torch.FloatTensor)).cuda()
            region_label = Variable(region_label.type(torch.FloatTensor)).cuda()
            affinity_label = Variable(affinity_label.type(torch.FloatTensor)).cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out, _ = net(image)

                out1 = out[:, :, :, 0].cuda()
                out2 = out[:, :, :, 1].cuda()

                loss_r = criterion(out1, region_label)
                loss_a = criterion(out2, affinity_label)

                loss = loss_r + loss_a 

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #loss.backward()
            #optimizer.step()
            scheduler.step()

            loss_value += loss.item()
            if index % 2 == 0 and index > 0:
                et = time.time()
                print('epoch {}:({}/{}) batch || training time for 32 batch {} || training loss {} || learning rate {}'.format(epoch, index, len(sample_train_loader), et-st, loss_value/2, optimizer.param_groups[0]['lr']))
                loss_time = 0
                loss_value = 0
                st = time.time()

            if index % 5000 ==0 and index > 0: 
                print('Save epoch : {} , index : {}'.format(epoch, index))
                torch.save({
                            'epoch' : epoch,
                            'model_state_dict' : net.state_dict(),
                            'optimizer_state_dict' : optimizer.state_dict(),
                            'loss' : loss_value
                            }, '/root/data/model_param_sample2/{}_{}.pth'.format(epoch,index))


            





