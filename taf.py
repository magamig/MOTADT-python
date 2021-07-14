import math

import numpy as np
import torch
from torch.optim import SGD
import cv2

from taf_rank import taf_rank_model
from taf_reg import taf_reg_model
from taf_classification import taf_clas_model, taf_clas_model_grad

torch.backends.cudnn.benchmark=True

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


def taf_model(features, filter_sizes, device):
    '''
    args:
        filter_sizes - size of exemplar feature maps [batch, channel, height, width]
        features - features of the Conv4-3 and Conv4-1 layers of VGG16 that
                   will be used to calculate target and scale sensitive features
    return: target aware features
            feature weights - the position of the weights that we select as target aware features
            balance weights - two values used to rescale the features due to scalar difference between conv4-1 and conv4-3
    '''
    feature_weights = []
    channel_num = [80,300]
    nz_num = 0
    nz_num_min = 250
    balance_weights = []
    features_rank = [features[0][0], features[1][0]]
    features = features[0]


    for i in range(len(features)):
        feature, filter_size = features[i], filter_sizes[i] #[1, 512, 45, 45], [  1 512  17  13]
        
        feature_weight, indices = taf_reg_model(feature, filter_size, device)

        # we perform scale sensitive feature selection on the conv4-1 feature, as it retains more spatial information
        #if i == 0:
        #    temp_feature_weight = taf_rank_model(features_rank, filter_size, device)
        #    feature_weight = feature_weight * temp_feature_weight

        # we perform classification feature selection on the conv4-1 feature, as it retains more spatial information
        if i == 0:
            #feature_weight = torch.ones(512)
            #indices = torch.tensor(range(512))
            temp_feature_weight = taf_clas_model_grad(features_rank, filter_size, device)
            feature_weight = feature_weight * temp_feature_weight

        feature_weight[indices[channel_num[i]:]] = 0
        nz_num = nz_num + torch.sum(feature_weight)

        # In case，there　are not enough features, we set a minimum feature number.
        # If the total number is less than the minimum number, then select more from conv4_3
        if i == 1 and nz_num < nz_num_min:
            added_indices = indices[torch.sum(feature_weight).to(torch.long): (torch.sum(feature_weight)+ nz_num_min - nz_num).to(torch.long) ]
            feature_weight[added_indices] = 1
        feature_weights.append(feature_weight)
        balance_weights.append(torch.max(torch.sum(torch.squeeze(feature)[indices[0:49],:,:],dim=0)))

    balance_weights = balance_weights[1] / torch.tensor(balance_weights, device = device)

    return feature_weights, balance_weights


def taf_model_diff(features, shift_pos, device):
    '''
    args:
        filter_sizes - size of exemplar feature maps [batch, channel, height, width]
        features - features of the Conv4-3 and Conv4-1 layers of VGG16 from both objects
    return: target aware features
            feature weights - the position of the weights that we select as target aware features
            balance weights - two values used to rescale the features due to scalar difference between conv4-1 and conv4-3
    '''

    balance_weights = []
    feature_weights = []
    channel_num = [80,300]

    obj1_features = features[0]
    obj2_features = features[1]
    obj1_shifted_features = features[2]

    for i in range(len(obj1_features)):
        
        # retrieve obj1 and obj2 conv layer
        obj1 = obj1_features[i]
        obj2 = obj2_features[i]

        # shift obj1 by shift_pos pixels width-wise
        #obj1_shifted = torch.roll(obj1, shifts=shift_pos, dims=3)
        obj1_shifted = obj1_shifted_features[i]

        # element-wise multiplication
        mult_features = obj1 * obj1_shifted
        
        # evaluate the sum over each chanel, in order to get a 1D vector, each dimension representing one chanel
        vec1 = torch.sum(mult_features, dim = (0,2,3))

        # apply the exact same process between the feature of obj1 and the ones of obj2 (without shifting)
        vec2 = torch.sum(obj1*obj2, dim = (0,2,3))

        # calculate V=vec1-vec2.
        V = vec1-vec2

        # select the top-N chanels associated with the maximum values
        sorted_cap, indices = torch.sort(V, descending = True)

        feature_weight = torch.zeros(len(indices))
        #print('indices[sorted_cap > 0] ', indices[sorted_cap > 0][0])
        #print('indices[0] ', indices[0])
        #feature_weight[indices[sorted_cap > 0]] = 1
        feature_weight[indices[sorted_cap > 0][0]] = 1

        feature_weight[indices[channel_num[i]:]] = 0

        feature_weights.append(feature_weight)
        balance_weights.append(torch.max(torch.sum(torch.squeeze(obj1)[indices[0:49],:,:],dim=0)))
    
    balance_weights = balance_weights[1] / torch.tensor(balance_weights, device = device)
    
    return feature_weights, balance_weights
    
if __name__ == '__main__':
    from feature_utils_v2 import resize_tensor

    input = torch.rand(1, 1, 3, 3)
    print(input)
    resized_input = resize_tensor(input, [7, 7])
    print(resized_input)
