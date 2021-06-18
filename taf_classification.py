import torch
import torch.nn as nn
from torch.optim import SGD

from taf_net import Classification_Net

def taf_clas_model(all_obj_features, filter_size, device):
    '''
    args:
        filter_size - size of exemplar feature maps [batch, channel, height, width]
        feature - either the features of the Conv4-3 or Conv4-1 layers of VGG16 that
                  will be used to calculate target or scale sensitive features
    return: indices - list of indices of the weights selected during regression
            reg_feature_weights - tensor containing 0's and 1's at the positions determined by indices
    '''
    feature = torch.cat((all_obj_features[0], all_obj_features[1]))
    clas_net = Classification_Net(filter_size).to(device) #used for classification features
    feature_size = torch.tensor(feature[0].shape).numpy()

    objective = nn.CrossEntropyLoss()
    optim = SGD(clas_net.parameters(),lr = 1e-9,momentum = 0.9,weight_decay = 1000)

    labels = torch.tensor([1, 0]).to(device)

    # first train the network with cross_entropy_loss
    train_clas(clas_net, optim, feature, objective, labels, device)
    clas_net_weights = clas_net.conv.weight.data
    
    # The value of the parameters equals the sum of the gradients in all BP processes.
    # And we found that using the converged parameters is more stable
    sorted_cap, indices = torch.sort(torch.sum(clas_net_weights, dim = (0,2,3)),descending = True) #GAP

    clas_net_feature_weights = torch.zeros(len(indices))
    clas_net_feature_weights[indices[sorted_cap > 0]] = 1

    return clas_net_feature_weights

def train_clas(model, optim, input, objective, targets, device, epochs = 100):
    """
    function: train the classification net and classification loss
    """
    for i in range(epochs):
        input = input.to(device)
        predict =  torch.squeeze(model(input))
        predict = predict.reshape(predict.shape[0], predict.shape[1]*predict.shape[2])

        loss = objective(predict, targets)
        if hasattr(optim,'module'):
            optim.module.zero_grad()
            loss.backward()
            optim.module.step()
        else:
            optim.zero_grad()
            loss.backward()
            optim.step()
