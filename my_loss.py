import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-7)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def DANNORI(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)


def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
    # print(op_out.shape)
    ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCEWithLogitsLoss().cuda()(ad_out, dc_target)


def DANN(features, ad_net, entropy=None, coeff=None, cls_weight=None, len_share=0):
    ad_out = ad_net(features)
    train_bs = (ad_out.size(0) - len_share) // 2
    dc_target = torch.from_numpy(np.array([[1]] * train_bs + [[0]] * (train_bs + len_share))).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy)
    else:
        entropy = torch.ones(ad_out.size(0)).cuda()

    source_mask = torch.ones_like(entropy)
    source_mask[train_bs : 2 * train_bs] = 0
    source_weight = entropy * source_mask
    source_weight = source_weight * cls_weight

    target_mask = torch.ones_like(entropy)
    target_mask[0 : train_bs] = 0
    target_mask[2 * train_bs::] = 0
    target_weight = entropy * target_mask
    target_weight = target_weight * cls_weight

    weight = (1.0 + len_share / train_bs) * source_weight / (torch.sum(source_weight).detach().item()) + \
            target_weight / torch.sum(target_weight).detach().item()
        
    weight = weight.view(-1, 1)
    return torch.sum(weight * nn.BCELoss(reduction='none')(ad_out, dc_target)) / (1e-8 + torch.sum(weight).detach().item())


def DANNAUG(features, ad_net, len_share=0):
    ad_out = ad_net(features)
    train_bs = (ad_out.size(0) - len_share) // 2
    dc_target = torch.from_numpy(np.array([[1]] * train_bs + [[0]] * (train_bs + len_share))).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)


def marginloss(yHat, y, classes=65, alpha=1, weight=None):
    batch_size = len(y)
    classes = classes
    yHat = F.softmax(yHat, dim=1)
    Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))#.detach()
    Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
    Px = yHat / Yg_.view(len(yHat), 1)
    Px_log = torch.log(Px + 1e-10)  # avoiding numerical issues (second)
    y_zerohot = torch.ones(batch_size, classes).scatter_(
        1, y.view(batch_size, 1).data.cpu(), 0)

    output = Px * Px_log * y_zerohot.cuda()
    loss = torch.sum(output, dim=1)/ np.log(classes - 1)
    Yg_ = Yg_ ** alpha
    if weight is not None:
        weight *= (Yg_.view(len(yHat), )/ Yg_.sum())
    else:
        weight = (Yg_.view(len(yHat), )/ Yg_.sum())

    weight = weight.detach()
    loss = torch.sum(weight * loss) / torch.sum(weight)

    return loss


def SAN(input_list, ad_net_list, grl_layer_list, class_weight, use_gpu=True):
    loss = 0
    outer_product_out = torch.bmm(input_list[0].unsqueeze(2), input_list[1].unsqueeze(1))
    batch_size = input_list[0].size(0) // 2
    dc_target = Variable(torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float())
    if use_gpu:
        dc_target = dc_target.cuda()
    for i in range(len(ad_net_list)):
        ad_out = ad_net_list[i](grl_layer_list[i](outer_product_out.narrow(2, i, 1).squeeze(2)))
        loss += nn.BCELoss()(ad_out.view(-1), dc_target.view(-1))
    return loss

def EntropyLoss(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))

def PADA(features, ad_net, grl_layer, weight_ad, use_gpu=True):
    ad_out = ad_net(grl_layer(features))
    batch_size = ad_out.size(0) // 2
    dc_target = Variable(torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float())
    if use_gpu:
        dc_target = dc_target.cuda()
        weight_ad = weight_ad.cuda()
    return nn.BCELoss(weight=weight_ad.view(-1))(ad_out.view(-1), dc_target.view(-1))


def DDC(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


def CORAL(source, target):
    d = source.data.shape[1]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t().mm(xm)
    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t().mm(xmt)
    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)

    return loss

def H(x):
    return -x * torch.log2(x + 1e-10) - (1 - x) * torch.log2(1 - x + 1e-10)

def Attention_entropy(x, attention_global):
    mask = x.ge(0.000001)
    mask_out = torch.masked_select(x, mask)
    attention_global_out = torch.masked_select(attention_global, mask)
    entropy = -(torch.sum(attention_global_out.data * mask_out * torch.log(mask_out)))
    return entropy / float(x.size(0))


def TADA(features, ad_net, softmax_out):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    loss1 = nn.BCELoss().cuda()(ad_out, dc_target)
    global_out = H(ad_out)
    global_attention = global_out.data + 1
    loss2 = Attention_entropy(softmax_out, global_attention)
    # print(loss1, loss2)
    return loss1+0.1*loss2


def EAR(features, ad_net, softmax_out):
    ad_out = ad_net(features)
    global_out = H(ad_out)
    global_attention = global_out.data + 1
    return Attention_entropy(softmax_out, global_attention)
