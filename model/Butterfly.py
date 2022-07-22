import torch.nn as nn
import model.backbone as backbone
from model.modules.grl import WarmStartGradientReverseLayer
import torch.nn.functional as F
import torch
import numpy as np
import random



class ButterflyNet(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=31):
        super(ButterflyNet, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        self.classifier_layer_2_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer_2 = nn.Sequential(*self.classifier_layer_2_list)

        self.classifier_target_1_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_target_1 = nn.Sequential(*self.classifier_target_1_list)

        self.classifier_target_2_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_target_2 = nn.Sequential(*self.classifier_target_2_list)
        self.softmax = nn.Softmax(dim=1)

        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.classifier_layer_2[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer_2[dep * 3].bias.data.fill_(0.0)
            self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[dep * 3].bias.data.fill_(0.0)
            self.classifier_target_1[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_target_1[dep * 3].bias.data.fill_(0.0)
            self.classifier_target_2[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_target_2[dep * 3].bias.data.fill_(0.0)


        ## collect parameters
        self.parameter_list = [{"params":self.base_network.parameters(), "lr":0.1},
                               {"params":self.bottleneck_layer.parameters(), "lr":1},
                               {"params":self.classifier_layer.parameters(), "lr":1},
                               {"params":self.classifier_layer_2.parameters(), "lr":1},
                               {"params":self.classifier_target_1.parameters(), "lr":1},
                               {"params":self.classifier_target_2.parameters(), "lr":1}]

    def forward(self, inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        outputs_1 = self.classifier_layer(features)
        outputs_2 = self.classifier_layer_2(features)
        outputs_3 = self.classifier_target_1(features)
        outputs_4 = self.classifier_target_2(features)

        #softmax_outputs = self.softmax(outputs)

        return outputs_1, outputs_2, outputs_3, outputs_4, self.classifier_layer[0].weight.data, self.classifier_layer_2[0].weight.data

class Butterfly(object):
    def __init__(self, base_net='ResNet50', width=1024, class_num=31, use_bottleneck=True, use_gpu=True, srcweight=3):
        self.c_net = ButterflyNet(base_net, use_bottleneck, width, width, class_num)
        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight

    def predict(self, inputs):

        outputs1, outputs2, outputs3, outputs4, _, _ = self.c_net(inputs)
        softmax = nn.Softmax(dim=1)
    # index = torch.LongTensor(np.array(range(outputs_1.size(0)))).cuda()
        outputs1, outputs2, outputs3, outputs4 = softmax(outputs1), softmax(outputs2), softmax(outputs3), softmax(outputs4)
        return outputs1, outputs2, outputs3, outputs4

    def get_parameter_list(self):
        return self.c_net.parameter_list

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode

    def get_loss(self, inputs, labels_source, labels_target):
        total_loss = 0
        class_criterion = nn.CrossEntropyLoss()
        t = int(self.iter_num/200.0)
        if t < 5:
            lr = 0.0
        else:
            lr = linear_rampup(t)
        outputs_1, outputs_2, outputs_3, outputs_4, W_fc0, W_fc0_2 = self.c_net(inputs)

        temp_w = W_fc0
        temp_w2 = W_fc0_2
        weight_diff = torch.matmul(temp_w, temp_w2)
        weight_diff = torch.abs(weight_diff)
        weight_diff = torch.sum(weight_diff, 0)
        weight_diff = torch.mean(weight_diff)

        if t < 1:
            outputs_src_1 = outputs_1.narrow(0, 0, labels_source.size(0))
            outputs_src_2 = outputs_2.narrow(0, 0, labels_source.size(0))
            outputs_src_3 = outputs_3.narrow(0, 0, labels_source.size(0))
            outputs_src_4 = outputs_4.narrow(0, 0, labels_source.size(0))

            index_src_1 = class_rank_criterion(outputs_src_1, labels_source, lr)
            index_src_2 = class_rank_criterion(outputs_src_2, labels_source, lr)
            index_src_3 = class_rank_criterion(outputs_src_3, labels_source, lr)
            index_src_4 = class_rank_criterion(outputs_src_4, labels_source, lr)

            classifier_loss_1 = class_criterion(outputs_src_1[index_src_2], labels_source[index_src_2])
            classifier_loss_2 = class_criterion(outputs_src_2[index_src_1], labels_source[index_src_1])
            classifier_loss_3 = class_criterion(outputs_src_3[index_src_4], labels_source[index_src_4])
            classifier_loss_4 = class_criterion(outputs_src_4[index_src_3], labels_source[index_src_3])

            total_loss = classifier_loss_1 + classifier_loss_2 + classifier_loss_3 + classifier_loss_4 + weight_diff

        else:
            outputs_src_1 = outputs_1.narrow(0, 0, labels_source.size(0))
            outputs_src_2 = outputs_2.narrow(0, 0, labels_source.size(0))
            index_src_1 = class_rank_criterion(outputs_src_1, labels_source, lr)
            index_src_2 = class_rank_criterion(outputs_src_2, labels_source, lr)
            classifier_loss_1 = class_criterion(outputs_src_1[index_src_2], labels_source[index_src_2])
            classifier_loss_2 = class_criterion(outputs_src_2[index_src_1], labels_source[index_src_1])

            outputs_tgt_1 = outputs_1.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))
            outputs_tgt_2 = outputs_2.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))
            outputs_tgt_3 = outputs_3.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))
            outputs_tgt_4 = outputs_4.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))

            index_tgt_1 = class_rank_criterion(outputs_tgt_1, labels_target, lr)
            index_tgt_2 = class_rank_criterion(outputs_tgt_2, labels_target, lr)
            index_tgt_3 = class_rank_criterion(outputs_tgt_3, labels_target, lr)
            index_tgt_4 = class_rank_criterion(outputs_tgt_4, labels_target, lr)


            classifier_loss_1_1 = class_criterion(outputs_tgt_1[index_tgt_2], labels_target[index_tgt_2])
            classifier_loss_2_1 = class_criterion(outputs_tgt_2[index_tgt_1], labels_target[index_tgt_1])
            classifier_loss_3 = class_criterion(outputs_tgt_3[index_tgt_4], labels_target[index_tgt_4])
            classifier_loss_4 = class_criterion(outputs_tgt_4[index_tgt_3], labels_target[index_tgt_3])

            total_loss = classifier_loss_1 + classifier_loss_2 + classifier_loss_3 + classifier_loss_4 + classifier_loss_1_1 + classifier_loss_2_1 + weight_diff
            
        self.iter_num += 1

        return [total_loss, classifier_loss_1, classifier_loss_2, classifier_loss_3, classifier_loss_4, weight_diff]

def class_rank_criterion(outputs_source, labels_source, lr):
    remove_num = torch.ceil(torch.tensor(labels_source.size(0)*lr))
    softmax = nn.Softmax(dim=1)
    index = torch.LongTensor(np.array(range(labels_source.size(0)))).cuda()
    outputs_source = softmax(outputs_source)
    pred = outputs_source[index, labels_source]
    loss = - torch.log(pred)
    _, indices = torch.sort(loss, 0)
    topk = labels_source.size(0)-remove_num.long()
    index_src = indices[:topk]
    return index_src

# def class_rank_criterion_pl(outputs_1, outputs_2):

#     softmax = nn.Softmax(dim=1)
#     # index = torch.LongTensor(np.array(range(outputs_1.size(0)))).cuda()
#     outputs_1 = softmax(outputs_1)
#     outputs_2 = softmax(outputs_2)

#     p1_probs, p1_labels = torch.max(outputs_1, 1)
#     p2_probs, p2_labels = torch.max(outputs_2, 1)

#     index = []
#     labels = []
#     for i, l in enumerate(p1_labels):
#         if l == p2_labels[i]:
#             #if p1_probs[i] >= 0.9 or p2_probs[i] >= 0.9:
#             index.append(i)
#             labels.append(l)
#     index = torch.LongTensor(index).cuda()
#     labels = torch.LongTensor(labels).cuda()
#     train = 0 if len(index) == 0 else 1
#     #print(index, labels, train)
#     return index, labels, train

def linear_rampup(t):
    start_point = 0
    if t < start_point:
        return 0.0
    else:
        forget_rate = 0 + min(0.6 * (t - start_point) / 5, 0.6)
        return float(forget_rate)
