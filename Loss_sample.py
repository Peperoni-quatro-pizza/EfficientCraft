import numpy as np 
import torch 
import torch.nn as nn 

class OHEMloss(nn.Module): 
    def __init__(self, use_gpu = True, ratio = 3, standard = 0.1, topk = 50):
        super(OHEMloss, self).__init__()

        self.ratio = ratio
        self.standard = standard
        self.topk = topk

    #pixelwise_loss -> batch , pixelwise squared error 
    #score_label -> batch, region(or affinity) score 
    def batch_loss(self, pixelwise_loss, score_label): 

        pixel_num = 0

        batch_size = pixelwise_loss.size()[0]
        sum_loss = torch.mean(pixelwise_loss.view(-1))*0

        batchwise_loss = pixelwise_loss.view(batch_size, -1)
        batchwise_label = score_label.view(batch_size, -1)

        for batch in range(batch_size): 
            
            #pixel >= 0.1 -> positive pixel 
            positive_pixel = pixelwise_loss[batch][(score_label[batch] >= self.standard)]
            negative_pixel = pixelwise_loss[batch][(score_label[batch] < self.standard)]
            num_positive = len(positive_pixel)
            num_negative = len(negative_pixel)

            if num_positive == 0: 

                sum_loss += torch.mean(torch.topk(batchwise_loss[batch] , self.topk)[0])

            else: 

                sum_loss += torch.mean(positive_pixel)

                if num_negative < self.ratio*num_positive: 

                    sum_loss += torch.mean(negative_pixel)

                else: 

                    sum_loss += torch.mean(torch.topk(negative_pixel, self.ratio*num_positive)[0])

        return sum_loss 


    def forward(self, region_label, affinity_label, region_pred, affinity_pred): 

        region_label = region_label
        affinity_label = affinity_label 
        region_pred = region_pred
        affinity_pred = affinity_pred

        batch_size = region_label.shape[0]

        MSE = nn.MSELoss(reduce = False)

        assert region_label.size() == region_pred.size() and affinity_label.size() == affinity_pred.size()
        region_loss = MSE(region_pred, region_label)
        affinity_loss = MSE(affinity_pred, affinity_label)

        region_ohem = self.batch_loss(region_loss, region_label)
        affinity_ohem = self.batch_loss(affinity_loss, affinity_label)

        #나중에 학습확인용으로 3가지를 모두 반환한다. 
        return region_ohem/batch_size  + affinity_ohem/batch_size , torch.mean(region_loss.view(-1)), torch.mean(affinity_loss.view(-1))


