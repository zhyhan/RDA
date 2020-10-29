import torch.nn as nn
import model.backbone as backbone
import torch.nn.functional as F
import torch
import numpy as np

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


class RDANet(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=31):
        super(RDANet, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.grl_layer = GradientReverseLayer()
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

class PMD(object):
    def __init__(self, base_net='ResNet50', width=1024, class_num=31, use_bottleneck=True, use_gpu=True, srcweight=3):
        self.c_net = RDANet(base_net, use_bottleneck, width, width, class_num)
        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight

    def get_loss(self, inputs, labels_source, max_iter, del_rate=0.4, noisy_source_num=100):
        class_criterion = nn.CrossEntropyLoss()

        #introduce noisy source instances to improve the discrepancy
        # inputs = torch.cat((inputs_source, inputs_target, labels_source_noisy), dim=0)
        source_size, source_noisy_size, target_size = labels_source.size(0), noisy_source_num, \
            inputs.size(0) - labels_source.size(0) - noisy_source_num

        #gradual transition
        lr = linear_rampup(self.iter_num, total_iter=max_iter)

        _, outputs, _, outputs_adv = self.c_net(inputs)

        #compute cross entropy loss on source domain
        #classifier_loss = class_criterion(outputs.narrow(0, 0, labels_source.size(0)), labels_source)
        #get large loss samples index
        outputs_src = outputs.narrow(0, 0, source_size)
        classifier_loss, index_src = class_rank_criterion(outputs_src, labels_source, lr, del_rate)

        #compute discrepancy
        target_adv = outputs.max(1)[1]
        target_adv_src = target_adv.narrow(0, 0, source_size)
        target_adv_tgt = target_adv.narrow(0, source_size, target_size)
        target_adv_noisy = target_adv.narrow(0, source_size+target_size, source_noisy_size)

        outputs_adv_src = outputs_adv.narrow(0, 0, source_size)
        outputs_adv_tgt = outputs_adv.narrow(0, source_size, target_size)
        outputs_adv_noisy = outputs_adv.narrow(0, source_size+target_size, source_noisy_size)

        outputs_adv_src = outputs_adv_src[index_src]
        target_adv_src = target_adv_src[index_src]
        classifier_loss_adv_src = class_criterion(torch.cat((outputs_adv_src,outputs_adv_noisy),dim=0), \
            torch.cat((target_adv_src,target_adv_noisy),dim=0))

        logloss_tgt = torch.log(torch.clamp(1 - F.softmax(outputs_adv_tgt, dim = 1), min=1e-15))
        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

        #Todo: compute entropy loss of unlabeled examples.
        en_loss = entropy(outputs_adv_tgt) + entropy(outputs_adv_noisy)
        transfer_loss = self.srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt

        self.iter_num += 1
        total_loss = classifier_loss + transfer_loss + 0.1*en_loss
        #print(classifier_loss.data, transfer_loss.data, en_loss.data)
        return [total_loss, classifier_loss, transfer_loss, classifier_loss_adv_src, classifier_loss_adv_tgt]

    def predict(self, inputs):
        feature, _, softmax_outputs,_= self.c_net(inputs)
        return softmax_outputs, feature

    def get_parameter_list(self):
        return self.c_net.parameter_list

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode

def class_rank_criterion(outputs_source, labels_source, lr, del_rate):
    if lr > del_rate:
        lr = del_rate
    remove_num = torch.ceil(torch.tensor(labels_source.size(0)*lr))
    softmax = nn.Softmax(dim=1)
    index = torch.LongTensor(np.array(range(labels_source.size(0)))).cuda()
    outputs_source = softmax(outputs_source)
    pred = outputs_source[index, labels_source]
    loss = - torch.log(pred)
    _, indices = torch.sort(loss, 0)
    topk = labels_source.size(0)-remove_num.long()
    index_src = indices[:topk]
    loss = torch.mean(loss)
    return loss, index_src

def entropy(output_target):
    """
    entropy minimization loss on target domain data
    """
    softmax = nn.Softmax(dim=1)
    output = output_target
    output = softmax(output)
    en = -torch.sum((output*torch.log(output + 1e-8)), 1)
    return torch.mean(en)

def linear_rampup(now_iter, total_iter=20000):
    if total_iter == 0:
        return 1.0
    else:
        current = np.clip(now_iter / total_iter, 0., 1.0)
        return float(current)
