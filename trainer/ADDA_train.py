import tqdm
import argparse
from torch.autograd import Variable
import torch
import copy
import sys
sys.path.insert(0, "/home/ubuntu/nas/projects/RDA")
from utils.config import Config
class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, group_ratios, optimizer, num_iter):
        lr = self.init_lr * (1 + self.gamma * num_iter) ** (-self.decay_rate)
        i=0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * group_ratios[i]
            i+=1
        return optimizer


#==============eval
def evaluate(model_instance, input_loader):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False, False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        _, _, probabilities = model_instance.predict(inputs)

        probabilities = probabilities.data.float()
        labels = labels.data.float()
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)

    _, predict = torch.max(all_probs, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels).float() / float(all_labels.size()[0])
    model_instance.set_train(ori_train_state, ori_train_state)
    return {'accuracy':accuracy}

def target_evaluate(model_instance, input_loader, SourceClassifier):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False, False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        _, _, probabilities = model_instance.predict(inputs, SourceClassifier)

        probabilities = probabilities.data.float()
        labels = labels.data.float()
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)

    _, predict = torch.max(all_probs, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels).float() / float(all_labels.size()[0])
    model_instance.set_train(ori_train_state, ori_train_state)
    return {'accuracy':accuracy}

def pretrain(model_instance, train_source_loader, test_target_loader, group_ratios, max_iter, optimizer, lr_scheduler, eval_interval):
    model_instance.set_train(True, True)
    print("start train...")
    loss = [] #accumulate total loss for visulization.
    result = [] #accumulate eval result on target data during training.
    iter_num = 0
    epoch = 0
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=max_iter)
    while True:
        for datas_clean in tqdm.tqdm(
                train_source_loader,
                total=len(train_source_loader),
                desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):
            inputs_source, labels_source, _ = datas_clean

            optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, iter_num/5)
            optimizer.zero_grad()

            if model_instance.use_gpu:
                inputs_source, labels_source = Variable(inputs_source).cuda(), Variable(labels_source).cuda()
            else:
                inputs_source, labels_source = Variable(inputs_source), Variable(labels_source)

            total_loss = pretrain_batch(model_instance, inputs_source, labels_source, optimizer, iter_num, max_iter)

            #val
            if iter_num % eval_interval == 0 and iter_num != 0:
                eval_result = evaluate(model_instance, test_target_loader)
                print('source domain:', eval_result)
                result.append(eval_result['accuracy'].cpu().data.numpy())

            iter_num += 1
            total_progress_bar.update(1)
            loss.append(total_loss)

        epoch += 1

        if iter_num > max_iter:
            break
    print('finish train')
    #torch.save(model_instance.c_net.state_dict(), 'statistic/DANN_model.pth')
    return model_instance

def pretrain_batch(model_instance, inputs_source, labels_source, optimizer, iter_num, max_iter):
    inputs = inputs_source
    total_loss = model_instance.get_loss(inputs, labels_source)
    total_loss.backward()
    optimizer.step()
    return total_loss.cpu().data.numpy()

def discrimator_train(model_instance, SourceFeatureExtractor, TargetClassifier, train_source_loader, train_target_loader, test_target_loader, group_ratios_fe, group_ratios_disc, max_iter, optimizer_fe, optimizer_disc, lr_scheduler, eval_interval):
    model_instance.set_train(True, True)
    print("start train...")
    loss = [] #accumulate total loss for visulization.
    result = [] #accumulate eval result on target data during training.
    iter_num = 0
    epoch = 0
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=max_iter)
    while True:
        for (datas_clean, datat) in tqdm.tqdm(
                zip(train_source_loader, train_target_loader),
                total=min(len(train_source_loader), len(train_target_loader)),
                desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):
            inputs_source, labels_source, _ = datas_clean
            inputs_target, _, _ = datat

            optimizer_fe = lr_scheduler.next_optimizer(group_ratios_fe, optimizer_fe, iter_num/5)
            optimizer_fe.zero_grad()
            optimizer_disc = lr_scheduler.next_optimizer(group_ratios_disc, optimizer_disc, iter_num/5)
            optimizer_disc.zero_grad()

            if model_instance.use_gpu:
                inputs_source, inputs_target, labels_source = Variable(inputs_source).cuda(), Variable(inputs_target).cuda(), Variable(labels_source).cuda()
            else:
                inputs_source, inputs_target, labels_source = Variable(inputs_source), Variable(inputs_target), Variable(labels_source)

            total_loss = discrimator_train_batch(model_instance, SourceFeatureExtractor, inputs_source, labels_source, inputs_target, optimizer_fe, optimizer_disc)

            #val
            if iter_num % eval_interval == 0 and iter_num != 0:
                eval_result = target_evaluate(model_instance, test_target_loader, TargetClassifier)
                print('source domain:', eval_result)
                result.append(eval_result['accuracy'].cpu().data.numpy())

            iter_num += 1
            total_progress_bar.update(1)
            loss.append(total_loss)

        epoch += 1

        if iter_num > max_iter:
            break
    print('finish train')
    #torch.save(model_instance.c_net.state_dict(), 'statistic/DANN_model.pth')
    return [loss, result]


def discrimator_train_batch(model_instance, SourceFeatureExtractor, inputs_source, labels_source, inputs_target, optimizer_fe, optimizer_disc):
    total_loss = model_instance.get_loss(inputs_source, labels_source, inputs_target, SourceFeatureExtractor)
    total_loss.backward(retain_graph=True)
    optimizer_disc.step()
    feature_loss = - total_loss #Inverse loss to optimize target feature extractor.
    feature_loss.backward()
    optimizer_fe.step()
    return total_loss.cpu().data.numpy()

if __name__ == '__main__':
    from model.ADDA import SourceModel, DiscrimatorModel
    from preprocess.data_provider import load_images
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='all sets of configuration parameters',
                        default='/home/liujintao/app/transfer-lib/config/dann.yml')
    parser.add_argument('--dataset', default='Office-31', type=str,
                        help='which dataset')
    parser.add_argument('--src_address', default=None, type=str,
                        help='address of image list of source dataset')
    parser.add_argument('--tgt_address', default=None, type=str,
                        help='address of image list of target dataset')
    parser.add_argument('--stats_file', default=None, type=str,
                        help='store the training loss and and validation acc')
    parser.add_argument('--noisy_rate', default=None, type=float,
                        help='noisy rate')
    args = parser.parse_args()

    cfg = Config(args.config)
    print(args)
    source_file = args.src_address
    target_file = args.tgt_address


    if args.dataset == 'Office-31':
        class_num = 31
        width = 256
        srcweight = 4
        is_cen = False
    elif args.dataset == 'Office-home':
        class_num = 65
        width = 256
        srcweight = 2
        is_cen = False
    else:
        width = -1

    source_instance = SourceModel(base_net='ResNet50', width=width, use_gpu=True, class_num=class_num)
    train_source_loader = load_images(source_file, batch_size=16, is_cen=is_cen, split_noisy=False)
    train_target_loader = load_images(target_file, batch_size=16, is_cen=is_cen, split_noisy=False)
    test_target_loader = load_images(target_file, batch_size=16, is_train=False)

    param_groups = source_instance.get_parameter_list()
    group_ratios = [group['lr'] for group in param_groups]

    assert cfg.optim.type == 'sgd', 'Optimizer type not supported!'

    optimizer = torch.optim.SGD(param_groups, **cfg.optim.params)

    assert cfg.lr_scheduler.type == 'inv', 'Scheduler type not supported!'
    lr_scheduler = INVScheduler(gamma=cfg.lr_scheduler.gamma,
                                decay_rate=cfg.lr_scheduler.decay_rate,
                                init_lr=cfg.init_lr)
    source_instance = pretrain(source_instance, train_source_loader, test_target_loader, group_ratios, max_iter=10000, optimizer=optimizer, lr_scheduler=lr_scheduler, eval_interval=1000)

    SourceFeatureExtractor = source_instance.feature_extractor 
    SourceFeatureExtractor.train(False)
    SourceClassifier = source_instance.classifier
    SourceClassifier.train(False)

    TargetFeatureExtractor = copy.deepcopy(SourceFeatureExtractor)
    TargetClassifier = SourceClassifier
    #TODO how to use a model parameters to initlize another model
    discrimator_instance = DiscrimatorModel(TargetFeatureExtractor, base_net='ResNet50', width=width, use_gpu=True, class_num=class_num)
    
    param_groups = source_instance.get_parameter_list()
    group_ratios_fe = [param_groups[0]['lr']]
    assert cfg.optim.type == 'sgd', 'Optimizer type not supported!'
    optimizer_fe = torch.optim.SGD([param_groups[0]], **cfg.optim.params)

    group_ratios_disc = [param_groups[1]['lr']]
    assert cfg.optim.type == 'sgd', 'Optimizer type not supported!'
    optimizer_disc = torch.optim.SGD([param_groups[1]], **cfg.optim.params)

    discrimator_instance = discrimator_train(discrimator_instance, SourceFeatureExtractor, TargetClassifier,train_source_loader, train_target_loader, test_target_loader, group_ratios_fe, group_ratios_disc, max_iter=10000, optimizer_fe=optimizer_fe, optimizer_disc=optimizer_disc, lr_scheduler=lr_scheduler, eval_interval=1000)




    
