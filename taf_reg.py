import torch
import torch.nn as nn
from torch.optim import SGD

from taf_net import Regress_Net

def taf_reg_model(feature, filter_size, device):
    '''
    args:
        filter_size - size of exemplar feature maps [batch, channel, height, width]
        feature - either the features of the Conv4-3 or Conv4-1 layers of VGG16 that
                  will be used to calculate target or scale sensitive features
    return: indices - list of indices of the weights selected during regression
            reg_feature_weights - tensor containing 0's and 1's at the positions determined by indices
    '''
    reg = Regress_Net(filter_size).to(device) #used for target active features
    feature_size = torch.tensor(feature.shape).numpy()

    output_sigma = filter_size[-2:]* 0.1
    gauss_label = generate_gauss_label(feature_size[-2:], output_sigma).to(device)

    objective = nn.MSELoss()
    #optim = SGD(reg.parameters(),lr = 5e-7,momentum = 0.9,weight_decay = 0.0005)
    optim = SGD(reg.parameters(),lr = 1e-9,momentum = 0.9,weight_decay = 1000)

    # first train the network with mse_loss
    train_reg(reg, optim, feature, objective, gauss_label, device)
    reg_weights = reg.conv.weight.data

    # The value of the parameters equals the sum of the gradients in all BP processes.
    # And we found that using the converged parameters is more stable
    sorted_cap, indices = torch.sort(torch.sum(reg_weights, dim = (0,2,3)),descending = True) #GAP

    reg_feature_weights = torch.zeros(len(indices))
    reg_feature_weights[indices[sorted_cap > 0]] = 1

    return reg_feature_weights, indices
    
def train_reg(model, optim, input, objective, gauss_label, device, epochs = 100):
    """
    function: train the regression net and regression loss
    """
    for i in range(epochs):
        input = input.to(device)
        predict = model(input).view(1,-1)

        gauss_label = gauss_label.view(1,-1)
        loss = objective(predict, gauss_label)
        if hasattr(optim,'module'):
            optim.module.zero_grad()
            loss.backward()
            optim.module.step()
        else:
            optim.zero_grad()
            loss.backward()
            optim.step()



def generate_gauss_label(size, sigma, center = (0, 0), end_pad=(0, 0)):
    """
    function: generate gauss label for L2 loss
    """
    shift_x = torch.arange(-(size[1] - 1) / 2, (size[1] + 1) / 2 + end_pad[1])
    shift_y = torch.arange(-(size[0] - 1) / 2, (size[0] + 1) / 2 + end_pad[0])

    shift_y, shift_x = torch.meshgrid(shift_y, shift_x)

    alpha = 0.2
    gauss_label = torch.exp(-1*alpha*(
                        (shift_y-center[0])**2/(sigma[0]**2) +
                        (shift_x-center[1])**2/(sigma[1]**2)
                        ))

    return gauss_label
