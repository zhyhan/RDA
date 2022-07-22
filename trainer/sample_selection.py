"""
This code is to select out label-noisy examples according to the small loss criterion.  
"""

import tqdm
import argparse
import numpy as np
from torch.autograd import Variable
import torch
import sys
sys.path.insert(0, "/home/ubuntu/nas/projects/RDA")
from utils.config import Config
import warnings
warnings.filterwarnings("ignore")
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
def evaluate(model_instance, input_loader, loss_matrix, epoch):
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

    index = torch.LongTensor(np.array(range(all_labels.size(0)))).cuda()
    a_labels = all_labels
    pred = all_probs[index, a_labels.long()]
    loss = - torch.log(pred)
    loss = loss.data.cpu().numpy()
    loss_matrix[:,epoch]=loss
    _, predict = torch.max(all_probs, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels).float() / float(all_labels.size()[0])
    model_instance.set_train(ori_train_state)
    return accuracy, loss_matrix

def train(model_instance, train_source_loader, test_source_loader, group_ratios, max_iter, optimizer, lr_scheduler, max_epoch=30):
    model_instance.set_train(True)
    print("start train...")
    #max_epoch = 30
    sample_num = len(test_source_loader.dataset)
    loss_matrix = np.zeros((sample_num, max_epoch))
    iter_num = 0
    epoch = 0
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=max_iter)
    while True:
        for datas in tqdm.tqdm(train_source_loader, total=len(train_source_loader), desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):
            inputs_source, labels_source, _ = datas

            optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, iter_num/5)
            optimizer.zero_grad()

            if model_instance.use_gpu:
                inputs_source, labels_source = Variable(inputs_source).cuda(), Variable(labels_source).cuda()
            else:
                inputs_source, labels_source = Variable(inputs_source), Variable(labels_source)

            train_batch(model_instance, inputs_source, labels_source, optimizer)

            iter_num += 1
            total_progress_bar.update(1)

        #val
        eval_result, loss_matrix = evaluate(model_instance, test_source_loader, loss_matrix, epoch)
        loss_matrix = loss_matrix

        if epoch % 10 == 0:
            #print(loss_matrix)
            print('source domain accuracy:', eval_result)
        epoch += 1

        if epoch >= max_epoch:
            break
    print('finish train')
    return loss_matrix

def train_batch(model_instance, inputs_source, labels_source, optimizer):
    inputs = inputs_source
    total_loss = model_instance.get_loss(inputs, labels_source)
    total_loss.backward()
    optimizer.step()

if __name__ == '__main__':
    from model.Resnet import ResNetModel
    from preprocess.data_provider import load_images
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='all sets of configuration parameters',
                        default='/home/liujintao/app/transfer-lib/config/dann.yml')
    parser.add_argument('--dataset', default='Office-31', type=str,
                        help='which dataset')
    parser.add_argument('--src_address', default=None, type=str,
                        help='address of image list of source dataset')
    parser.add_argument('--stats_file', default=None, type=str,
                        help='store the training loss and and validation acc')
    parser.add_argument('--noisy_rate', default=0.4, type=float,
                        help='noisy rate')
    parser.add_argument('--noisy_type', default='feature_uniform', type=str,
                        help='noisy rate')
    args = parser.parse_args()

    cfg = Config(args.config)
    print(args)
    source_file = args.src_address
    #target_file = args.tgt_address
    epoch = 10

    if args.dataset == 'Office-31':
        class_num = 31
        width = 1024
        srcweight = 4
        is_cen = False
    elif args.dataset == 'Office-home':
        class_num = 65
        width = 2048
        srcweight = 2
        is_cen = False
    elif args.dataset == 'Bing-Caltech':
        class_num = 257
        width = 2048
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
    elif args.dataset == 'webvision':
        class_num = 1000
        width = 256
        srcweight = 4
        is_cen = False
    else:
        width = -1

    model_instance = ResNetModel(base_net='ResNet50', width=width, use_gpu=True, class_num=class_num, srcweight=srcweight)

    train_source_loader = load_images(source_file, batch_size=128, is_cen=is_cen, drop_last=True)
    test_source_loader = load_images(source_file, batch_size=128, is_train=False)

    param_groups = model_instance.get_parameter_list()
    group_ratios = [group['lr'] for group in param_groups]

    assert cfg.optim.type == 'sgd', 'Optimizer type not supported!'

    optimizer = torch.optim.SGD(param_groups, **cfg.optim.params)

    assert cfg.lr_scheduler.type == 'inv', 'Scheduler type not supported!'
    lr_scheduler = INVScheduler(gamma=cfg.lr_scheduler.gamma,
                                decay_rate=cfg.lr_scheduler.decay_rate,
                                init_lr=cfg.init_lr)
    loss_matrix = train(model_instance, train_source_loader, test_source_loader, group_ratios, max_iter=1200000, optimizer=optimizer, lr_scheduler=lr_scheduler, max_epoch=epoch)
    np.save(args.stats_file, loss_matrix)

    #detect small loss sample
    save_clean_file = source_file.split('.t')[0] + '_full_true_pred.txt'
    nr = args.noisy_rate
    save_noisy_file = source_file.split('.t')[0] + '_full_false_pred.txt'
    clean_labels, noise_labels, imgs = [], [], []
    if args.noisy_type == 'uniform':
        with open(source_file, 'r') as f:
            images = f.readlines()
            for index, i in enumerate(images):
                i =  i.split()
                img = i[0]
                imgs.append(img)
                noisy_label = i[1]
                clean_label = i[2]
                noise_labels.append(int(noisy_label))
                if noisy_label == clean_label:
                    clean_labels.append(1)
                else:
                    clean_labels.append(0)
    elif args.noisy_type == 'ood':
        with open(source_file, 'r') as f:
            images = f.readlines()
            for index, i in enumerate(images):
                i =  i.split()
                img = i[0]
                imgs.append(img)
                noisy_label = i[1]
                clean_label = i[2]
                noise_labels.append(int(noisy_label))
                if noisy_label == clean_label:
                    clean_labels.append(1)
                else:
                    clean_labels.append(0)
    elif args.noisy_type == 'ood_uniform':
        with open(source_file, 'r') as f:
            images = f.readlines()
            for index, i in enumerate(images):
                i =  i.split()
                img = i[0]
                imgs.append(img)
                noisy_label = i[1]
                clean_label = i[2]
                noise_labels.append(int(noisy_label))
                if noisy_label == clean_label:
                    clean_labels.append(1)
                else:
                    clean_labels.append(0)
    elif args.noisy_type == 'feature':
        with open(source_file, 'r') as f:
            images = f.readlines()
            for index, i in enumerate(images):
                i =  i.split()
                img = i[0]
                imgs.append(img)
                noisy_label = i[1]
                noise_labels.append(int(noisy_label))
                if img.split('_')[-1] != 'corrupted.jpg':
                    clean_labels.append(1)
                else:
                    clean_labels.append(0)

    elif args.noisy_type == 'feature_uniform':
        nr = nr/2
        with open(source_file, 'r') as f:
            images = f.readlines()
            for index, i in enumerate(images):
                i =  i.split()
                img = i[0]
                imgs.append(img)
                noisy_label = i[1]
                clean_label = i[2]
                noise_labels.append(int(noisy_label))
                if noisy_label == clean_label:
                    clean_labels.append(1)
                else:
                    clean_labels.append(0)
    elif args.noisy_type == 'ood_feature_uniform':
        nr = 0.6
        with open(source_file, 'r') as f:
            images = f.readlines()
            for index, i in enumerate(images):
                i =  i.split()
                img = i[0]
                imgs.append(img)
                noisy_label = i[1]
                clean_label = i[2]
                noise_labels.append(int(noisy_label))
                if noisy_label == clean_label:
                    clean_labels.append(1)
                else:
                    clean_labels.append(0)
    elif args.noisy_type == 'ood_feature':
        nr = nr
        with open(source_file, 'r') as f:
            images = f.readlines()
            for index, i in enumerate(images):
                i =  i.split()
                img = i[0]
                imgs.append(img)
                noisy_label = i[1]
                clean_label = i[2]
                noise_labels.append(int(noisy_label))
                if noisy_label == clean_label:
                    clean_labels.append(1)
                else:
                    clean_labels.append(0)
    else:
        raise Exception('Unsupport noise type {}'.format(args.noisy_type))
    loss_sele = loss_matrix[:, :epoch]
    loss_mean = loss_sele.mean(axis=1)
    cr = min(1-1.2*nr, 0.9*(1-nr))
    sort_index = np.argsort(loss_mean)

    #sort samples per class
    clean_index = []
    for i in range(int(class_num)):
        c = []
        for idx in sort_index:
            if noise_labels[idx] == i:
                c.append(idx)
        clean_num = int(len(c)*cr)
        clean_idx = c[:clean_num]
        clean_index.extend(clean_idx)
    #clean_num = int(np.ceil(len(sort_index)*cr))
    #clean_index = sort_index[:clean_num]
    acc_mum = 0
    for i in clean_index:
        if clean_labels[i] == 1:
            acc_mum += 1
    acc = acc_mum/len(clean_index)
    print("Epoch {} vs Acc {} vs cr {} vs nr {}".format(epoch,acc,cr,nr))

    with open(save_clean_file,'w') as f:
        with open(save_noisy_file, 'w') as ff:
            for idx, img in enumerate(imgs):
                if idx in clean_index:
                    f.write('{} {}\n'.format(img, noise_labels[idx]))
                else:
                    ff.write('{} {}\n'.format(img, noise_labels[idx]))
