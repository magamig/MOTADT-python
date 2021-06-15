import math

import numpy as np
import torch
from torch.optim import SGD
import cv2

from taf_rank import taf_rank_model
from taf_reg import taf_reg_model

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

    for i in range(len(features)):
        feature, filter_size = features[i], filter_sizes[i] #[1, 512, 45, 45], [  1 512  17  13]
        
        feature_weight, indices = taf_reg_model(feature, filter_size, device)

        # we perform scale sensitive feature selection on the conv4-1 feature, as it retains more spatial information
        if i == 0:
            temp_feature_weight = taf_rank_model(feature, filter_size, device)
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

if __name__ == '__main__':
    from feature_utils_v2 import resize_tensor

    input = torch.rand(1, 1, 3, 3)
    print(input)
    resized_input = resize_tensor(input, [7, 7])
    print(resized_input)
