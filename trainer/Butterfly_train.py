import tqdm
import numpy as np
import argparse
from torch.autograd import Variable
import torch
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
def obtain_label(model_instance, input_loader, dir):

    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
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
        o1, o2, o3, _ = model_instance.predict(inputs)

        probabilities1 = o1.data.float()
        probabilities2 = o2.data.float()
        probabilities3 = o3.data.float()
        labels = labels.data.float()

        if first_test:
            all_probs1 = probabilities1
            all_probs2 = probabilities2
            all_probs3 = probabilities3
            all_labels = labels
            first_test = False
        else:
            all_probs1 = torch.cat((all_probs1, probabilities1), 0)
            all_probs2 = torch.cat((all_probs2, probabilities2), 0)
            all_probs3 = torch.cat((all_probs3, probabilities3), 0)
            all_labels = torch.cat((all_labels, labels), 0)

    probs1, predict1 = torch.max(all_probs1, 1)
    probs2, predict2 = torch.max(all_probs2, 1)
    _, predict3 = torch.max(all_probs3, 1)

    txt_tar = open(dir).readlines()
    new_target = []
    for i in range(len(txt_tar)):
        rec = txt_tar[i]
        reci = rec.strip().split(' ')
        if predict1[i] == predict2[i]:
            if probs1[i] > 0.95 or probs2[i] > 0.95:
                line = reci[0] + ' ' + str(int(predict1[i])) + ' ' + str(int(reci[1])) + '\n'
                new_target.append(line)

    accuracy = torch.sum(torch.squeeze(predict1).float() == all_labels).float() / float(all_labels.size()[0])
    model_instance.set_train(ori_train_state)
    return accuracy, new_target

def train_batch(model_instance, inputs_source, labels_source, inputs_target, labels_target, optimizer):
    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    total_loss = model_instance.get_loss(inputs, labels_source, labels_target)
    total_loss[0].backward()
    optimizer.step()
    return total_loss

if __name__ == '__main__':
    from model.Butterfly import Butterfly
    from preprocess.data_provider import load_images

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
    elif args.dataset == 'COVID-19':
        class_num = 3
        width = 256
        srcweight = 4
        is_cen = False
        # Another choice for Office-home:
        # width = 1024
        # srcweight = 3
        # is_cen = True
    elif args.dataset == 'Bing-Caltech':
        class_num = 257
        width = 2048
        srcweight = 2
        is_cen = False
    elif args.dataset == 'webvision':
        class_num = 1000
        width = 256
        srcweight = 4
        is_cen = False
    else:
        width = -1

    model_instance = Butterfly(base_net='ResNet50', width=width, use_gpu=True, class_num=class_num, srcweight=srcweight)

    train_source_loader = load_images(source_file, batch_size=128, is_cen=is_cen, split_noisy=False)
    train_target_loader = load_images(target_file, batch_size=128, is_cen=is_cen)
    
    val_file = '/home/ubuntu/nas/projects/RDA/data/webvision/val_filelist.txt'
    test_target_loader = load_images(val_file, batch_size=128, is_train=False)

    param_groups = model_instance.get_parameter_list()
    group_ratios = [group['lr'] for group in param_groups]

    assert cfg.optim.type == 'sgd', 'Optimizer type not supported!'

    optimizer = torch.optim.SGD(param_groups, **cfg.optim.params)

    assert cfg.lr_scheduler.type == 'inv', 'Scheduler type not supported!'
    lr_scheduler = INVScheduler(gamma=cfg.lr_scheduler.gamma,
                                decay_rate=cfg.lr_scheduler.decay_rate,
                                init_lr=cfg.init_lr)
    max_iter=100000
    # max_iter=6000
    # eval_interval=200
    eval_interval=200
    model_instance.set_train(True)
    print("start train...")
    iter_num = 0
    epoch = 0
    iter_num = 0
    while iter_num < max_iter:
        try:
            inputs_source, labels_source, _ = iter_source.next()
            inputs_target, labels_target, _ = iter_target.next()
        except:
            iter_source = iter(train_source_loader)
            iter_target = iter(train_target_loader)
            inputs_source, labels_source, _ = iter_source.next()
            inputs_target, labels_target, _ = iter_target.next()

        optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, iter_num/5)
        optimizer.zero_grad()

        if model_instance.use_gpu:
            inputs_source, inputs_target, labels_source, labels_target = Variable(inputs_source).cuda(),  Variable(inputs_target).cuda(), Variable(labels_source).cuda(), Variable(labels_target).cuda()
        else:
            inputs_source, inputs_target, labels_source, labels_target = Variable(inputs_source), Variable(inputs_target), Variable(labels_source), Variable(labels_target)

        total_loss = train_batch(model_instance, inputs_source, labels_source, inputs_target, labels_target, optimizer)

        #val
        if iter_num % eval_interval == 0 and iter_num != 0:
            #print('loss:', total_loss)
            #eval_result = evaluate(model_instance, test_target_loader)
            eval_result, new_target_file = obtain_label(model_instance, test_target_loader, target_file)
            print('iter_num={}/{}: target domain accuracy={}; reliable sample numbers={}'.format(iter_num, max_iter, eval_result, len(new_target_file)))

            if len(new_target_file) >= 64: #if the target sample is very small, it is unnesseary to train.
                train_target_loader = load_images(new_target_file, batch_size=32, is_cen=is_cen)
            else:
                train_target_loader = train_source_loader
        iter_num += 1
        if iter_num > max_iter:
            break
    print('finish train')
