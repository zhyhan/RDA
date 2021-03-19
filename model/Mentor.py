import torch.nn as nn
import model.backbone as backbone
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


class StudentNet(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=256, width=256, class_num=31):
        super(StudentNet, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.grl_layer = GradientReverseLayer()
        self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        #self.classifier_layer_2_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
        #                                nn.Linear(width, 1)]
        #self.classifier_layer_2 = nn.Sequential(*self.classifier_layer_2_list)
        self.softmax = nn.Softmax(dim=1)
        #self.sigmoid = nn.Sigmoid()
        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            #self.classifier_layer_2[dep * 3].weight.data.normal_(0, 0.01)
            #self.classifier_layer_2[dep * 3].bias.data.fill_(0.0)
            self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[dep * 3].bias.data.fill_(0.0)


        ## collect parameters, TCL does not train the base network.
        self.parameter_list = [{"params":self.bottleneck_layer.parameters(), "lr":1},
                               {"params":self.classifier_layer.parameters(), "lr":1}]

    def forward(self, inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        return features, outputs, softmax_outputs

class MentorNet(nn.Module):
    """
    Function: reprodue the MentorNet to dynamicly predict the weights V of each instance to mitigate the label-noise instance to a certain extent.
    Args：
    inpit_dim[0] is the loss of each instance at a batch
    input_dim[1] is the loss std from the average loss per batch
    input_dim[2] is the label of each instance at a batch
    input_dim[3] is the epoch/iteration number
    Return: weight of each instance at a batch.
    """
    def __init__(self, input_dim=4, output_dim=1):
        super(MentorNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigmoid = nn.Sigmoid()
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(), 
            nn.Linear(self.input_dim, self.output_dim),
            nn.Dropout(0.2)
        )
        for dep in range(2):
            self.layers[dep * 2].weight.data.normal_(0, 0.01)
            self.layers[dep * 2].bias.data.fill_(0.0)
        self.parameter_list = [{"params":self.layers.parameters(), "lr":1}]

    def forward(self, inputs):
        output = self.layers(inputs)
        w = self.sigmoid(output)
        return w


class Mentor(object):
    def __init__(self, base_net='ResNet50', width=1024, class_num=31, use_bottleneck=True, use_gpu=True, srcweight=3):
        self.studentnet = StudentNet(base_net, use_bottleneck, width, width, class_num)
        self.mentornet = MentorNet(input_dim=4, output_dim=1)
        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.studentnet = self.studentnet.cuda()
            self.mentornet = self.mentornet.cuda()
        self.srcweight = srcweight

    def get_loss(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss()
        _, outputs, softmax_outputs = self.studentnet(inputs)

        outputs_source = outputs.narrow(0, 0, labels_source.size(0))

        outputs_source_softmax = softmax_outputs.narrow(0, 0, labels_source.size(0))

        if self.iter_num < 2000:
            classifier_loss = class_criterion(outputs_source, labels_source)
        else:#Self Paced Learning
            classifier_loss = self.mentorloss(outputs_source_softmax, labels_source)

        self.iter_num += 1
        total_loss = classifier_loss
        #print(classifier_loss.data, transfer_loss.data, en_loss.data)
        return total_loss

    def predict(self, inputs):
        feature, _, softmax_outputs = self.studentnet(inputs)
        return softmax_outputs, feature

    def get_parameter_list(self):
        return self.studentnet.parameter_list + self.mentornet.parameter_list

    def set_train(self, mode):
        self.studentnet.train(mode)
        self.mentornet.train(mode)
        self.is_train = mode

    def mentorloss(self, source_y_softmax, label):
        y_softmax = source_y_softmax
        the_index = torch.LongTensor(np.array(range(label.size(0)))).cuda()
        y_label = y_softmax[the_index, label]
        y_loss = - torch.log(y_label)
        mean_loss = y_loss.mean()
        std_loss = y_loss-mean_loss
        iters = torch.tensor(self.iter_num).expand(label.size(0))

        y_loss = y_loss.reshape((label.size(0), 1))
        std_loss = std_loss.reshape((label.size(0), 1))
        label = label.reshape((label.size(0), 1)).float().cuda()
        iters = iters.reshape((label.size(0), 1)).float().cuda()
        #print(y_loss.size(), std_loss.size(), label.float().cuda().size(), iters.float().cuda().size())

        inputs = torch.cat((y_loss, std_loss, label.float().cuda(), iters), dim=-1)
        #print(inputs)
        weight_var = self.mentornet(inputs)
        #weight_var = (weight_loss < age).float().detach()
        Ly = torch.mean(y_loss * weight_var)
        #source_num = float((torch.sum(source_weight)))
        return Ly
