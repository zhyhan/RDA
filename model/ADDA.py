import torch.nn as nn
import model.backbone as backbone
import torch.nn.functional as F
import torch
import numpy as np
import random

class FeatureExtractor(nn.Module):
    def __init__(self, base_net='ResNet50'):
        super(FeatureExtractor, self).__init__()
        ## set base network
        self.backbone = backbone.network_dict[base_net]()
        self.parameter_list = {"params":self.backbone.parameters(), "lr":1}

    def forward(self, inputs):
        features = self.backbone(inputs)
        return features

class Classifier(nn.Module):
    def __init__(self, width=1024, class_num=31, input_dim=2048):
        super(Classifier, self).__init__()
        ## set base network
        self.layers = nn.Sequential(
            nn.Linear(input_dim, width), 
            nn.BatchNorm1d(width), 
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(width, width),
            nn.BatchNorm1d(width), 
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(width, class_num)
        )

        self.softmax = nn.Softmax(dim=1)
        ## initialization
        for dep in range(3):
            self.layers[dep * 4].weight.data.normal_(0, 0.01)
            self.layers[dep * 4].bias.data.fill_(0.0)

        ## collect parameters
        self.parameter_list = {"params":self.layers.parameters(), "lr":1}

    def forward(self, inputs):
        logits = self.layers(inputs)
        softmax_outputs = self.softmax(logits)
        return logits, softmax_outputs

class Discrimator(nn.Module):
    def __init__(self, input_dim=2048, width=1024, class_num=1):
        super(Discrimator, self).__init__()
        self.layers =  nn.Sequential(nn.Linear(input_dim, width), 
            nn.BatchNorm1d(width),
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(width, class_num))

        self.sigmoid = nn.Sigmoid()

        for dep in range(2):
            self.layers[dep * 4].weight.data.normal_(0, 0.01)
            self.layers[dep * 4].bias.data.fill_(0.0)

        self.parameter_list = {"params":self.layers.parameters(), "lr":1}

    def forward(self, inputs):
        logits = self.layers(inputs)
        output = self.sigmoid(logits)
        return output

class SourceModel(object):
    def __init__(self, base_net='ResNet50', width=1024, class_num=31, use_gpu=True):
        self.feature_extractor = FeatureExtractor(base_net)
        self.classifier = Classifier(width, class_num)
        self.use_gpu = use_gpu
        self.is_train = False
        self.class_num = class_num
        if self.use_gpu:
            self.feature_extractor = self.feature_extractor.cuda()
            self.classifier = self.classifier.cuda()

    def get_loss(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss()
        features = self.feature_extractor(inputs)
        logits, _ = self.classifier(features)
        classifier_loss = class_criterion(logits, labels_source)
        return classifier_loss

    def predict(self, inputs):
        features = self.feature_extractor(inputs)
        logits, softmax_outputs = self.classifier(features)
        return features, logits, softmax_outputs

    def get_parameter_list(self):
        return [self.feature_extractor.parameter_list, self.classifier.parameter_list]

    def set_train(self, mode1, mode2):
        self.feature_extractor.train(mode1)
        self.classifier.train(mode2)
        self.is_train = mode1

class DiscrimatorModel(object):
    def __init__(self, TargetFeatureExtractor, base_net='ResNet50', width=1024, class_num=31, use_gpu=True):
        self.feature_extractor = TargetFeatureExtractor
        self.discrimator = Discrimator(width=width)
        self.use_gpu = use_gpu
        self.is_train = False
        self.class_num = class_num
        if self.use_gpu:
            self.feature_extractor = self.feature_extractor.cuda()
            self.discrimator = self.discrimator.cuda()

    def get_loss(self, inputs_source, labels_source, inputs_target, source_feature_extractor):
        class_criterion = nn.BCELoss()
        source_input = inputs_source
        target_input = inputs_target
        source_fature = source_feature_extractor(source_input)
        target_feature = self.feature_extractor(target_input)
        features = torch.cat([source_fature, target_feature], 0)
        outputs = self.discrimator(features)

        source_domain_label = torch.FloatTensor(inputs_source.size(0), 1)
        target_domain_label = torch.FloatTensor(inputs_target.size(0), 1)
        source_domain_label.fill_(1)
        target_domain_label.fill_(0)
        domain_label = torch.cat([source_domain_label,target_domain_label],0)
        domain_label = torch.autograd.Variable(domain_label.cuda())

        discrimator_loss = class_criterion(outputs, domain_label)
        return discrimator_loss

    def predict(self, inputs, classifier):
        features = self.feature_extractor(inputs)
        logits, softmax_outputs = classifier(features)
        return features, logits, softmax_outputs

    def get_parameter_list(self):
        return [self.feature_extractor.parameter_list, self.discrimator.parameter_list]

    def set_train(self, mode1, mode2):
        self.feature_extractor.train(mode1)
        self.discrimator.train(mode2)
        self.is_train = mode1