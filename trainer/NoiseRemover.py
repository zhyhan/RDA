from torch import feature_alpha_dropout
import tqdm
import argparse
from torch.autograd import Variable
import torch
import numpy as np
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

T = 1
#==============eval
def evaluate(model_instance, input_loader):
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
        features, logits, softmax_outputs = model_instance.predict(inputs)
        probabilities = softmax_outputs.data.float()
        features = features.data.float()
        labels = labels.data.float()
        logits = logits.data.float()
        energy_score = - T*torch.logsumexp(logits / T, dim=1)
        if first_test:
            all_probs = probabilities
            all_energy_score = energy_score
            all_feats = features
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_feats = torch.cat((all_feats, features), 0)
            all_labels = torch.cat((all_labels, labels), 0)
            all_energy_score = torch.cat((all_energy_score, energy_score), 0)

    prob_matrix = all_probs.cpu().numpy()
    all_feats = all_feats.cpu().numpy()
    all_energy_score = all_energy_score.cpu().numpy()
    _, predict = torch.max(all_probs, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels).float() / float(all_labels.size()[0])
    model_instance.set_train(ori_train_state)
    return {'accuracy':accuracy}, prob_matrix, all_feats, all_energy_score

def train(model_instance, train_source_clean_loader, train_source_noisy_loader, group_ratios, max_iter, optimizer, lr_scheduler, eval_interval, del_rate=0.4, class_num=31):
    model_instance.set_train(True)
    print("start train...")
    
    #sample_num = len(train_source_noisy_loader.dataset)
    #prob_matrix = np.zeros((sample_num, class_num))
    iter_num = 0
    epoch = 0
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=max_iter)
    while True:
        for datas in tqdm.tqdm(train_source_clean_loader, total=len(train_source_clean_loader), desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):
            inputs_source, labels_source, _ = datas

            optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, iter_num/5)
            optimizer.zero_grad()

            if model_instance.use_gpu:
                inputs_source, labels_source = Variable(inputs_source).cuda(), Variable(labels_source).cuda()
            else:
                inputs_source, labels_source = Variable(inputs_source), Variable(labels_source)

            train_batch(model_instance, inputs_source, labels_source, optimizer)

            #val
            if iter_num % eval_interval == 0 and iter_num != 0:
                eval_result, _, _, all_energy_score = evaluate(model_instance, train_source_clean_loader)
                print(eval_result)
            iter_num += 1
            total_progress_bar.update(1)
        epoch += 1

        if iter_num > max_iter:
            break
    eval_result, prob_matrix, all_feats, all_energy_score = evaluate(model_instance, train_source_noisy_loader)
    print('accuracy:', eval_result)
    print('prob matrix:', prob_matrix)
    print('energy score:', all_energy_score)
    print('finish train')
    #torch.save(model_instance.c_net.state_dict(), 'statistic/Ours_model.pth')
    return prob_matrix, all_feats, all_energy_score

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
    parser.add_argument('--tgt_address', default=None, type=str,
                        help='address of image list of target dataset')
    parser.add_argument('--stats_file', default=None, type=str,
                        help='store the training loss and and validation acc')
    parser.add_argument('--noisy_rate', default=None, type=float,
                        help='noisy rate')
    parser.add_argument('--del_rate', default=0.4, type=float,
                        help='delete rate of sample for transfer')

    args = parser.parse_args()

    cfg = Config(args.config)
    print(args)
    source_file = args.src_address
    target_file = args.tgt_address


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
        # Another choice for Office-home:
        # width = 1024
        # srcweight = 3
        # is_cen = True
    elif args.dataset == 'COVID-19':
        class_num = 3
        width = 256
        srcweight = 4
        is_cen = False
    elif args.dataset == 'webvision':
        class_num = 1000
        width = 256
        srcweight = 4
        is_cen = False
    else:
        width = -1

    model_instance = ResNetModel(base_net='ResNet50', width=width, use_gpu=True, class_num=class_num, srcweight=srcweight)
    if args.noisy_rate == 0.:
        train_source_clean_loader = load_images(source_file, batch_size=128, is_cen=is_cen, split_noisy=False)
        train_source_noisy_loader = train_source_clean_loader
    else:
        train_source_clean_loader, train_source_noisy_loader = load_images(source_file, batch_size=32, is_cen=is_cen, split_noisy=True)

    param_groups = model_instance.get_parameter_list()
    group_ratios = [group['lr'] for group in param_groups]

    assert cfg.optim.type == 'sgd', 'Optimizer type not supported!'

    optimizer = torch.optim.SGD(param_groups, **cfg.optim.params)

    assert cfg.lr_scheduler.type == 'inv', 'Scheduler type not supported!'
    lr_scheduler = INVScheduler(gamma=cfg.lr_scheduler.gamma,
                                decay_rate=cfg.lr_scheduler.decay_rate,
                                init_lr=cfg.init_lr)
    to_dump, all_feats, all_energy_score = train(model_instance, train_source_clean_loader, train_source_noisy_loader, group_ratios, max_iter=10000, optimizer=optimizer, lr_scheduler=lr_scheduler, eval_interval=1000, del_rate=args.del_rate, class_num=class_num)
    pickle.dump(all_energy_score, open(args.stats_file, 'wb'))

    source_file_new = source_file.split('.t')[0]+'_false_pred.txt'
    save_file = source_file.split('.t')[0]+'_false_pred_refine.txt'
    with open(source_file_new, 'r') as f:
        file_dir, label = [], []
        for i in f.read().splitlines():
            file_dir.append(i.split(' ')[0])
            label.append(int(i.split(' ')[1]))

    with open(save_file,'w') as f:
        for i, d in enumerate(all_energy_score):
            if d < np.sort(all_energy_score)[int(len(all_energy_score)/6)]:#select top 1/6 data as clean data. TODO
                f.write('{} {}\n'.format(file_dir[i], label[i]))

