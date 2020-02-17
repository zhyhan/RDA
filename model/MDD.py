import torch.nn as nn
import model.backbone as backbone
import torch.nn.functional as F
import torch
import numpy as np
import random

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


class MDDNet(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=31):
        super(MDDNet, self).__init__()
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

class MDD(object):
    def __init__(self, base_net='ResNet50', width=1024, class_num=31, use_bottleneck=True, use_gpu=True, srcweight=3):
        self.c_net = MDDNet(base_net, use_bottleneck, width, width, class_num)
        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight

    def get_loss(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss()
        _, outputs, _, outputs_adv = self.c_net(inputs)
        classifier_loss = class_criterion(outputs.narrow(0, 0, labels_source.size(0)), labels_source)

        target_adv = outputs.max(1)[1]
        target_adv_src = target_adv.narrow(0, 0, labels_source.size(0))
        target_adv_tgt = target_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))

        classifier_loss_adv_src = class_criterion(outputs_adv.narrow(0, 0, labels_source.size(0)), target_adv_src)

        logloss_tgt = torch.log(torch.clamp(1 - F.softmax(outputs_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0)), dim = 1), min=1e-15)) #add small value to avoid the log value expansion

        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

        transfer_loss = self.srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt

        outputs_target = outputs.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))
        #en_loss = entropy(outputs_target)
        self.iter_num += 1
        total_loss = classifier_loss + transfer_loss #+ 0.1*en_loss
        #print(classifier_loss.data, transfer_loss.data, en_loss.data)
        return [total_loss, classifier_loss, transfer_loss, classifier_loss_adv_src, classifier_loss_adv_tgt]

    #using proxy MDD
    def get_proxy_loss(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss()

        #mixup inputs between source and target data
        #TODO Random concat samples into new distribution.
        source_input = inputs.narrow(0, 0, labels_source.size(0))
        target_input = inputs.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))
        if source_input.size(0) >= target_input.size(0):
            source_input = source_input.narrow(0, 0, target_input.size(0))
        else:
            target_input = target_input.narrow(0, 0, source_input.size(0))

        #gradual transition
        #lr = linear_rampup(self.iter_num, total_iter=10000)
        lr = 0.5
        mix_input = lr*source_input + (1-lr)*target_input

        inputs = torch.cat((inputs, mix_input), dim=0)
        _, outputs, _, outputs_adv = self.c_net(inputs)

        #compute cross entropy loss on source domain
        classifier_loss = class_criterion(outputs.narrow(0, 0, labels_source.size(0)), labels_source)

        #compute discrepancy
        target_adv = outputs.max(1)[1]
        target_adv_src = target_adv.narrow(0, 0, labels_source.size(0))
        target_adv_tgt = target_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0) - mix_input.size(0))
        target_adv_mix = target_adv.narrow(0, labels_source.size(0) + mix_input.size(0), mix_input.size(0))

        outputs_adv_src = outputs_adv.narrow(0, 0, labels_source.size(0))
        outputs_adv_tgt = outputs_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0) - mix_input.size(0))
        outputs_adv_mix = outputs_adv.narrow(0, labels_source.size(0) + mix_input.size(0), mix_input.size(0))

        classifier_loss_adv_src = class_criterion(outputs_adv_src, target_adv_src)

        logloss_tgt = torch.log(torch.clamp(1 - F.softmax(outputs_adv_tgt, dim = 1), min=1e-15)) #add small value to avoid the log value expansion
        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

        loss_adv_mix_2 = class_criterion(outputs_adv_mix, target_adv_mix)
        logloss_mix = torch.log(torch.clamp(1 - F.softmax(outputs_adv_mix, dim = 1), min=1e-15)) #add small value to avoid the log value expansion
        loss_adv_mix_1 = F.nll_loss(logloss_mix, target_adv_mix)

        transfer_loss = self.srcweight * classifier_loss_adv_src + loss_adv_mix_1 + loss_adv_mix_2 + classifier_loss_adv_tgt #+loss_adv_mix_2

        outputs_target = outputs.narrow(0, labels_source.size(0) + mix_input.size(0), mix_input.size(0))
        en_loss = entropy(outputs_target)
        self.iter_num += 1
        total_loss = classifier_loss + transfer_loss + 0.1*en_loss
        #print(classifier_loss.data, transfer_loss.data, en_loss.data)
        return [total_loss, classifier_loss, transfer_loss, classifier_loss_adv_src, classifier_loss_adv_tgt]

    def get_gradual_loss(self, inputs, labels_source, max_iter):
        class_criterion = nn.CrossEntropyLoss()

        #mixup inputs between source and target data
        #TODO Random concat samples into new distribution.
        source_input = inputs.narrow(0, 0, labels_source.size(0))
        target_input = inputs.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))

        #gradual transition
        lr = linear_rampup(self.iter_num, total_iter=max_iter)

        _, outputs, _, outputs_adv = self.c_net(inputs)

        #compute cross entropy loss on source domain
        #classifier_loss = class_criterion(outputs.narrow(0, 0, labels_source.size(0)), labels_source)
        #get large loss samples index
        outputs_src = outputs.narrow(0, 0, labels_source.size(0))
        classifier_loss, index_src, index_tgt = class_rank_criterion(outputs_src, labels_source, target_input.size(0), lr)

        #compute discrepancy
        target_adv = outputs.max(1)[1]
        target_adv_src = target_adv.narrow(0, 0, labels_source.size(0))
        target_adv_tgt = target_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))
        target_adv_mix = torch.cat((target_adv_src[index_src], target_adv_tgt[index_tgt]), dim=0)

        outputs_adv_src = outputs_adv.narrow(0, 0, labels_source.size(0))
        outputs_adv_tgt = outputs_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))
        outputs_adv_mix = torch.cat((outputs_adv_src[index_src], outputs_adv_tgt[index_tgt]), dim=0)

        #print(target_adv_mix, outputs_adv_mix)
        classifier_loss_adv_src = class_criterion(outputs_adv_src, target_adv_src)

        logloss_tgt = torch.log(torch.clamp(1 - F.softmax(outputs_adv_tgt, dim = 1), min=1e-15)) #add small value to avoid the log value expansion
        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

        loss_adv_mix_2 = class_criterion(outputs_adv_mix, target_adv_mix)
        logloss_mix = torch.log(torch.clamp(1 - F.softmax(outputs_adv_mix, dim = 1), min=1e-15)) #add small value to avoid the log value expansion
        loss_adv_mix_1 = F.nll_loss(logloss_mix, target_adv_mix)

        transfer_loss = self.srcweight * classifier_loss_adv_src + loss_adv_mix_1 + loss_adv_mix_2 + classifier_loss_adv_tgt #+loss_adv_mix_2

        #outputs_target = outputs.narrow(0, labels_source.size(0) + mix_input.size(0), mix_input.size(0))
        #en_loss = entropy(outputs_target)
        self.iter_num += 1
        total_loss = classifier_loss + transfer_loss #+ 0.1*en_loss
        #print(classifier_loss.data, transfer_loss.data, en_loss.data)
        return [total_loss, classifier_loss, transfer_loss, classifier_loss_adv_src, classifier_loss_adv_tgt]

    def predict(self, inputs):
        features, _, softmax_outputs,_= self.c_net(inputs)
        return features, softmax_outputs

    def get_parameter_list(self):
        return self.c_net.parameter_list

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode

    def get_denoise_loss(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss()
        _, outputs, _, outputs_adv = self.c_net(inputs)
        #split source and target of classifier outputs
        outputs_source = outputs.narrow(0, 0, labels_source.size(0))
        outputs_target = outputs.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))
        #get classifier predicted labels
        target_adv = outputs.max(1)[1]
        #split source and target predicted labels
        target_adv_src = target_adv.narrow(0, 0, labels_source.size(0))
        target_adv_tgt = target_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))

        #compute source disparity discrepancy
        classifier_loss_adv_src = class_criterion(outputs_adv.narrow(0, 0, labels_source.size(0)), target_adv_src)

        #compute target disparity discrepancy
        logloss_tgt = torch.log(torch.clamp(1 - F.softmax(outputs_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0)), dim = 1), min=1e-15)) #add small value to avoid the log value expansion
        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

        #compute margin disparity discrepancy
        transfer_loss = self.srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt

        #compute classifier loss by designing a new loss function
        classifier_loss = denoise_loss(outputs_source, labels_source)

        #TODO entropy minimization on target domain
        #en_loss = entropy(outputs_target)

        #compute total loss without loss of generality
        total_loss = classifier_loss + transfer_loss  #+ 0.1*en_loss
        #print(classifier_loss.data, transfer_loss.data)
        self.iter_num += 1
        #return all loss value for final statistics
        return [total_loss, classifier_loss, transfer_loss, classifier_loss_adv_src, classifier_loss_adv_tgt]

    def get_denoise_mixmatch_loss(self, inputs, labels_source, max_iter):
        class_criterion = nn.CrossEntropyLoss()

        #mixmatch inputs between source and target data

        #source_input = inputs.narrow(0, 0, labels_source.size(0))
        #target_input = inputs.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))
        #if source_input.size(0) >= target_input.size(0):
        #    source_input = source_input.narrow(0, 0, target_input.size(0))
        #else:
        #    target_input = target_input.narrow(0, 0, source_input.size(0))
        lamda = np.random.beta(1,1)
        lamda = max(lamda, 1-lamda)#believe source domain better
        #mix_inputs = lamda*source_input + (1-lamda)*source_input
        idx = torch.randperm(inputs.size(0))
        input_a, input_b = inputs, inputs[idx]
        mix_inputs = lamda*input_a + (1-lamda)*input_b
        inputs = torch.cat((inputs, mix_inputs), dim=0)

        _, outputs, _, outputs_adv = self.c_net(inputs)#TODO use adv outputs of input_mix

        #split source and target of classifier outputs
        outputs_source = outputs.narrow(0, 0, labels_source.size(0))
        outputs_target = outputs.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0) - mix_inputs.size(0))
        output_mix = outputs.narrow(0, inputs.size(0) - mix_inputs.size(0), mix_inputs.size(0))

        #get classifier predicted labels
        target_adv = outputs.max(1)[1]
        #split source and target predicted labels
        target_adv_src = target_adv.narrow(0, 0, labels_source.size(0))
        target_adv_tgt = target_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0) - mix_inputs.size(0))

        #compute source disparity discrepancy
        classifier_loss_adv_src = class_criterion(outputs_adv.narrow(0, 0, labels_source.size(0)), target_adv_src)

        #compute target disparity discrepancy
        logloss_tgt = torch.log(torch.clamp(1 - F.softmax(outputs_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0) - mix_inputs.size(0)), dim = 1), min=1e-15)) #add small value to avoid the log value expansion
        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

        #compute margin disparity discrepancy
        transfer_loss = self.srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt

        #compute classifier loss by designing a new loss function
        classifier_loss, source_loss, mixup_loss = denoise_mixmatch_loss(outputs_source, outputs_target, labels_source, idx, disc=transfer_loss, output_mix=output_mix, now_iter=self.iter_num, max_iter=max_iter, lamda=lamda)

        #if self.iter_num % 10 == 0:
        #    print('clean loss: {}, mixup loss: {}'.format(source_loss.data.cpu(), mixup_loss.data.cpu()))
        #TODO entropy minimization on target domain
        #en_loss = entropy(outputs_target)

        #compute total loss without loss of generality
        total_loss = classifier_loss + transfer_loss  #+ 0.1*en_loss
        #print(classifier_loss.data, transfer_loss.data)
        self.iter_num += 1
        #return all loss value for final statistics
        return [total_loss, classifier_loss, transfer_loss, classifier_loss_adv_src, classifier_loss_adv_tgt]

def class_rank_criterion(outputs_source, labels_source, tgt_size, lr):
    remove_num = torch.ceil(torch.tensor(min(labels_source.size(0)*lr, tgt_size*lr)))
    softmax = nn.Softmax(dim=1)
    index = torch.LongTensor(np.array(range(labels_source.size(0)))).cuda()
    outputs_source = softmax(outputs_source)
    pred = outputs_source[index, labels_source]
    loss = - torch.log(pred)
    _, indices = torch.sort(loss, 0)
    topk = labels_source.size(0)-remove_num.long()
    index_src = indices[:topk]
    #print(remove_num, index_src)
    index_tgt = torch.LongTensor(random.sample(range(0, tgt_size), int(remove_num.data.cpu()))).cuda()
    #print(index_tgt)
    loss = torch.mean(loss)
    return loss, index_src, index_tgt


def denoise_loss(outputs_source, labels_source, gamma=0.08):
    softmax = nn.Softmax(dim=1)
    index = torch.LongTensor(np.array(range(labels_source.size(0)))).cuda()
    outputs_source = softmax(outputs_source)
    """
    clever! same cross entropy,
    find the probability of correspongding label,
    can be used for computing discrepancy between two samples
    """
    y_true_pred = outputs_source[index, labels_source]
    y_loss = - torch.log(y_true_pred)
    gamma = y_loss.median()
    clean_weight = (y_loss <= gamma).float().detach()
    #because lots of zero weight by access the median
    y_true_loss = torch.mean(y_loss * clean_weight)
    return y_true_loss

def denoise_mixmatch_loss(outputs_source, outputs_target, labels_source, idx, disc=2, output_mix=None, now_iter=0, max_iter=20000, lamda=1.0):
    softmax = nn.Softmax(dim=1)
    index = torch.LongTensor(np.array(range(labels_source.size(0)))).cuda()
    outputs_source = softmax(outputs_source)

    """
    clever! same cross entropy,
    find the probability of correspongding label,
    can be used for computing discrepancy between two samples
    """
    y_true_pred = outputs_source[index, labels_source]
    y_loss = - torch.log(torch.clamp(y_true_pred, min=1e-15))
    gamma = y_loss.median()#TODO how choose gamma?
    clean_weight = (y_loss <= gamma).float().detach()
    source_loss = torch.mean(y_loss * clean_weight)

    #mix label and unlabel source and target domain samples
    #noisy_weight = 1 - clean_weight
    #clean_index = torch.LongTensor([i for i, j in enumerate(clean_weight) if j == 1.]).cuda()
    #noisy_index = torch.LongTensor([i for i, j in enumerate(noisy_weight) if j == 1.]).cuda()
    sources_l = F.one_hot(labels_source, num_classes=31).float()
    #instead large loss samples
    #s = outputs_source
    #st = s**2
    #st = st / st.sum(dim=1, keepdim=True)
    #st = st.detach()
    #sources_l[noisy_index]=st[noisy_index]
    #print(sources_l)
    p = softmax(outputs_target)
    pt = p**2
    targets_u = pt / pt.sum(dim=1, keepdim=True)
    targets_u = targets_u.detach()
    mix_labels = torch.cat((sources_l, targets_u), dim=0)
    label_a, label_b = mix_labels, mix_labels[idx]
    mix_labels = lamda*label_a + (1-lamda)*label_b


    # mixup
    output_mix = softmax(output_mix)
    #output_mix_source = output_mix.narrow(0, 0, labels_source.size(0))
    output_mix_target = output_mix.narrow(0, labels_source.size(0), output_mix.size(0)-labels_source.size(0))
    #label_mix_source = mix_labels.narrow(0, 0, labels_source.size(0))
    label_mix_target = mix_labels.narrow(0, labels_source.size(0), output_mix.size(0)-labels_source.size(0))
#    loss_source = -torch.mean(torch.sum(torch.log(torch.clamp(output_mix_source, min=1e-15)) * label_mix_source, dim=1))
    #loss_source_noisy = torch.mean((output_mix_source[noisy_index] - label_mix_source[noisy_index])**2)
    loss_target = torch.mean((output_mix_target - label_mix_target)**2)

    lr = linear_rampup(now_iter, max_iter)
    #loss = mix_loss
    mix_loss = lr*75*(loss_target)
    #mix_loss = loss_source + lr*75*(loss_target)
    loss = source_loss + mix_loss
    return loss, source_loss, mix_loss

#    #compute mixup mse loss of untrusted sample
#    noisy_weight = clean_weight
#    #noisy_weight = 1 - clean_weight
#    noisy_weight = noisy_weight.narrow(0, 0, output_mix.size(0))
#    index = torch.LongTensor([i for i, j in enumerate(noisy_weight) if j == 1.]).cuda()
#    mix_pred = softmax(output_mix[index, :])
#    MSEloss = nn.MSELoss()
#    source_pred = F.one_hot(labels_source.narrow(0, 0, output_mix.size(0)), num_classes=31)[index,:].float()
#    #source_pred = outputs_source[index,:]
#    target_pred = outputs_target[index,:]
#    #sharpening guessed labels
#    pt = lamda*source_pred+(1-lamda)*target_pred
#    pt = pt**2
#    pt = pt / pt.sum(dim=1, keepdim=True)
#    #mix_src_loss = MSEloss(mix_pred,source_pred)
#    mixup_untrusted_loss = MSEloss(pt, mix_pred)
#    #mix_tgt_loss = MSEloss(mix_pred,target_pred)
#
#    #compute cross entropy loss of trusted sample
#    output_mix = softmax(output_mix)
#    index = torch.LongTensor(np.array(range(output_mix.size(0)))).cuda()
#    y_pred = output_mix[index, labels_source.narrow(0, 0, output_mix.size(0))]
#    if hard:
#        y_pred_loss = - torch.log(y_pred)
#    else:
#        source_soft = outputs_source[index, target_adv_src.narrow(0, 0, output_mix.size(0))]
#        y_pred_loss = - source_soft*torch.log(y_pred)
#    y_pred_loss_src = torch.mean(y_pred_loss * clean_weight.narrow(0, 0, output_mix.size(0)))
#
#    y_pred_mixup = output_mix[index, target_adv_tgt.narrow(0, 0, output_mix.size(0))]
#    if hard:
#        y_pred_loss = - torch.log(y_pred_mixup)
#    else:
#        target_soft = outputs_target[index, target_adv_tgt.narrow(0, 0, output_mix.size(0))]
#        y_pred_loss = - target_soft*torch.log(y_pred_mixup)
#    y_pred_loss_tgt = torch.mean(y_pred_loss * clean_weight.narrow(0, 0, output_mix.size(0)))
#
#    mixup_trusted_loss = lamda*y_pred_loss_src + (1-lamda)*y_pred_loss_tgt
#
#    mixup_loss = mixup_untrusted_loss
#    #mixup_loss = mixup_trusted_loss + 3.3*mixup_untrusted_loss
#    #d = max(0.05, min(0.5, 1./disc))
#    mix_loss = mixup_loss
#    #mix_loss = y_true_loss + 1*mixup_loss
#    #print(lamda, y_true_loss.data.cpu(), mixup.data.cpu(), mix_loss.data.cpu())
#
#    return mix_loss, y_true_loss, mixup_loss
#
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
