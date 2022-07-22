import torch.nn as nn
import model.backbone as backbone
from model.modules.grl import WarmStartGradientReverseLayer
import torch.nn.functional as F
import torch
import numpy as np
import random
import math

class GradientReverseLayer(torch.autograd.Function):
    def __init__(self, iter_num=0, alpha=1.0, low_value=0.0, high_value=0.1, max_iter=1000.0):
        self.iter_num = iter_num
        self.alpha = alpha
        self.low_value = low_value
        self.high_value = high_value
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        output = input * 1.0
        return output

    def backward(self, grad_output):
        self.coeff = np.float(
            2.0 * (self.high_value - self.low_value) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (self.high_value - self.low_value) + self.low_value)
        return -self.coeff * grad_output


class TCLNet(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=31):
        super(TCLNet, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000., auto_step=True)
        self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        self.classifier_layer_2_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer_2 = nn.Sequential(*self.classifier_layer_2_list)
        self.softmax = nn.Softmax(dim=1)

        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.classifier_layer_2[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer_2[dep * 3].bias.data.fill_(0.0)
            self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[dep * 3].bias.data.fill_(0.0)


        ## collect parameters
        self.parameter_list = [{"params":self.base_network.parameters(), "lr":0.1},
                            {"params":self.bottleneck_layer.parameters(), "lr":1},
                        {"params":self.classifier_layer.parameters(), "lr":1},
                               {"params":self.classifier_layer_2.parameters(), "lr":1}]

    def forward(self, inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        features_adv = self.grl_layer(features)
        outputs_adv = self.classifier_layer_2(features_adv)
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        return features, outputs, softmax_outputs, outputs_adv

class TCL(object):
    def __init__(self, base_net='ResNet50', width=1024, class_num=31, use_bottleneck=True, use_gpu=True, srcweight=3):
        self.c_net = TCLNet(base_net, use_bottleneck, width, width, class_num)
        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight

    def get_loss(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss()
        _, outputs, softmax_outputs, outputs_adv = self.c_net(inputs)

        outputs_source = outputs.narrow(0, 0, labels_source.size(0))

        outputs_source_softmax = softmax_outputs.narrow(0, 0, labels_source.size(0))
        outputs_target_softmax = softmax_outputs.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))

        d_source = outputs_adv.narrow(0, 0, labels_source.size(0))

        if self.iter_num < 2000:
            classifier_loss = class_criterion(outputs_source, labels_source)
        else:
            classifier_loss, source_weight, _ = cal_Ly(outputs_source_softmax, labels_source)
            #target_weight = torch.ones(inputs.size(0) - source_weight.size(0)).cuda()

        en_loss = entropy(outputs_target_softmax)


        target_adv = outputs.max(1)[1]
        target_adv_src = target_adv.narrow(0, 0, labels_source.size(0))
        target_adv_tgt = target_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))

        outputs_adv_src = outputs_adv.narrow(0, 0, labels_source.size(0))
        if self.iter_num > 2000:
            index = []
            for i, j in enumerate(source_weight):
                if j != 0.0:
                    index.append(i)
            index = torch.LongTensor(np.array(index)).cuda()
            outputs_adv_src = outputs_adv_src[index]
            target_adv_src = target_adv_src[index]

        classifier_loss_adv_src = class_criterion(outputs_adv_src, target_adv_src)

        logloss_tgt = torch.log(torch.clamp(1 - F.softmax(outputs_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0)), dim = 1), min=1e-15)) #add small value to avoid the log value expansion

        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

        transfer_loss = self.srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt


        # if self.iter_num > 2000:
        #     index = []
        #     for i, j in enumerate(source_weight):
        #         if j != 0.0:
        #             index.append(i)
        #     index = torch.LongTensor(np.array(index)).cuda()
        #     outputs_adv_src = outputs_adv_src[index]
        #     target_adv_src = target_adv_src[index]

        self.iter_num += 1
        total_loss = classifier_loss + transfer_loss #+ 0.1*en_loss
        # if self.iter_num%10 == 0:
        #     print(classifier_loss, transfer_loss)#, en_loss)
        return [total_loss, classifier_loss, transfer_loss, en_loss]

    def predict(self, inputs):
        feature, _, softmax_outputs,_= self.c_net(inputs)
        return softmax_outputs, feature

    def get_parameter_list(self):
        return self.c_net.parameter_list

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode

def entropy(output_target):
    """
    entropy minimization loss on target domain data
    """
    output = output_target
    en = -torch.sum((output*torch.log(output + 1e-8)), 1)
    return torch.mean(en)

def linear_rampup(now_iter, total_iter=20000):
    if total_iter == 0:
        return 1.0
    else:
        current = np.clip(now_iter / total_iter, 0., 1.0)
        return float(current)


def cal_Ly(source_y_softmax, label):

    agey = - math.log(0.3)
    aged = - math.log(1.0 - 0.5)
    age = agey + 0.1 * aged
    y_softmax = source_y_softmax
    the_index = torch.LongTensor(np.array(range(source_y_softmax.size(0)))).cuda()
    y_label = y_softmax[the_index, label]
    y_loss = - torch.log(y_label)
    weight_loss = y_loss
    weight_var = (weight_loss < age).float().detach()
    Ly = torch.mean(y_loss * weight_var)
    source_weight = weight_var.data.clone()
    source_num = float((torch.sum(source_weight)))
    #print(source_num)
    return Ly, source_weight, source_num