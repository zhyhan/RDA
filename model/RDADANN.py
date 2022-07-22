import torch.nn as nn
import model.backbone as backbone
from model.modules.grl import WarmStartGradientReverseLayer
import torch.nn.functional as F
import torch
import numpy as np
class RDANet(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=31):
        super(RDANet, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=1, max_iters=1000., auto_step=True)
        self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        self.classifier_layer_2_list = [nn.Linear(bottleneck_dim, width), 
                                        nn.BatchNorm1d(width), 
                                        nn.ReLU(), 
                                        nn.Linear(width, width),
                                        nn.BatchNorm1d(width),
                                        nn.ReLU(),
                                        nn.Linear(width, 1),
                                        nn.Sigmoid()]
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



# class DomainAdversarialLoss(nn.Module):

#     def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean',
#                  grl: Optional = None):
#         super(DomainAdversarialLoss, self).__init__()
#         self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) if grl is None else grl
#         self.domain_discriminator = domain_discriminator
#         self.bce = lambda input, target, weight: \
#             F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)
#         self.domain_discriminator_accuracy = None

#     def forward(self, f_s: torch.Tensor, f_t: torch.Tensor,
#                 w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
#         f = self.grl(torch.cat((f_s, f_t), dim=0))
#         d = self.domain_discriminator(f)
#         d_s, d_t = d.chunk(2, dim=0)
#         d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
#         d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
#         self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))

#         if w_s is None:
#             w_s = torch.ones_like(d_label_s)
#         if w_t is None:
#             w_t = torch.ones_like(d_label_t)
#         return 0.5 * (self.bce(d_s, d_label_s, w_s.view_as(d_s)) + self.bce(d_t, d_label_t, w_t.view_as(d_t)))


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

        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction='mean')

    def get_loss(self, inputs, labels_source, max_iter, del_rate=0.4, noisy_source_num=100):
        class_criterion = nn.CrossEntropyLoss()

        #introduce noisy source instances to improve the discrepancy
        # inputs = torch.cat((inputs_source, inputs_target, labels_source_noisy), dim=0)
        source_size, source_noisy_size, target_size = labels_source.size(0), noisy_source_num, \
            inputs.size(0) - labels_source.size(0) - noisy_source_num

        #gradual transition
        lr = linear_rampup(self.iter_num, total_iter=max_iter)

        _, outputs, _, outputs_adv = self.c_net(inputs)

        outputs_src = outputs.narrow(0, 0, source_size)
        classifier_loss, index_src = class_rank_criterion(outputs_src, labels_source, lr, del_rate)



        outputs_adv_src = outputs_adv.narrow(0, 0, source_size)
        #print(source_size, source_noisy_size, target_size, inputs.size(0))
        outputs_adv_tgt = outputs_adv.narrow(0, source_size, target_size)
        #outputs_adv_noisy = outputs_adv.narrow(0, source_size+target_size, source_noisy_size)

        #en_loss = entropy(outputs_adv_tgt) + entropy(outputs_adv_noisy) #+ entropy(outputs_adv_src)

        self.iter_num += 1

        source_domain_label = torch.FloatTensor(source_size, 1).cuda()
        target_domain_label = torch.FloatTensor(target_size, 1).cuda()
        source_domain_label.fill_(1)
        target_domain_label.fill_(0)
        #domain_label = torch.cat([source_domain_label, target_domain_label],0)

        w_s = torch.zeros_like(source_domain_label)
        w_s[index_src] = 1

        #print(w_s)
        w_t = torch.ones_like(target_domain_label)

        Ld = 0.5 * (self.bce(outputs_adv_src, source_domain_label, w_s.view_as(outputs_adv_src)) + self.bce(outputs_adv_tgt, target_domain_label, w_t.view_as(outputs_adv_tgt)))

        #Ld = domain_criterion(outputs_adv_src, source_domain_label)

        #total_loss = classifier_loss + transfer_loss + 0.1*en_loss
        total_loss = classifier_loss + Ld #+ 0.1*en_loss
        #print(classifier_loss.data, transfer_loss.data, en_loss.data)
        return [total_loss, classifier_loss, Ld]#, en_loss]

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
