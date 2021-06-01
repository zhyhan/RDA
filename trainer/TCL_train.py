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
        probabilities, feature = model_instance.predict(inputs)

        probabilities = probabilities.data.float()
        labels = labels.data.float()
        feature = feature.data.float()

        if first_test:
            all_probs = probabilities
            all_labels = labels
            all_feature = feature

            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)
            all_feature = torch.cat((all_feature, feature), 0)

    _, predict = torch.max(all_probs, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels).float() / float(all_labels.size()[0])
    model_instance.set_train(ori_train_state)
    return {'accuracy':accuracy}, all_feature

def train(model_instance, train_source_clean_loader, train_source_noisy_loader, train_target_loader, test_target_loader, group_ratios, max_iter, optimizer, lr_scheduler, eval_interval):
    model_instance.set_train(True)
    print("start train...")
    loss = [] #accumulate total loss for visulization.
    result = [] #accumulate eval result on target data during training.
    iter_num = 0
    epoch = 0
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=max_iter)
    while True:
        for (datas_clean, datas_noisy, datat) in tqdm.tqdm(
                zip(train_source_clean_loader, train_source_noisy_loader, train_target_loader),
                total=min(len(train_source_clean_loader), len(train_target_loader)),
                desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):
            inputs_source, labels_source, _ = datas_clean
            inputs_source_noisy, labels_source_noisy, _ = datas_noisy
            inputs_target, labels_target, _ = datat

            optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, iter_num/5)
            optimizer.zero_grad()

            if model_instance.use_gpu:
                inputs_source, inputs_source_noisy, inputs_target, labels_source, labels_source_noisy = Variable(inputs_source).cuda(),  Variable(inputs_source_noisy).cuda(), Variable(inputs_target).cuda(), Variable(labels_source).cuda(), Variable(labels_source_noisy).cuda()
            else:
                inputs_source, inputs_source_noisy, inputs_target, labels_source, labels_source_noisy = Variable(inputs_source),  Variable(inputs_source_noisy), Variable(inputs_target), Variable(labels_source), Variable(labels_source_noisy)

            total_loss = train_batch(model_instance, inputs_source, labels_source, inputs_target, optimizer, iter_num, max_iter)

            #val
            if iter_num % eval_interval == 0 and iter_num != 0:
                eval_result, all_feature = evaluate(model_instance, test_target_loader)
                print('source domain:', eval_result)
                result.append(eval_result['accuracy'].cpu().data.numpy())

            iter_num += 1
            total_progress_bar.update(1)
            loss.append(total_loss)

        epoch += 1

        if iter_num > max_iter:
            #np.save('statistic/TCL_feature_target.npy', all_feature.cpu().numpy())
            break
    print('finish train')
    #torch.save(model_instance.c_net.state_dict(), 'statistic/TCL_model.pth')
    return [loss, result]
def train_batch(model_instance, inputs_source, labels_source, inputs_target, optimizer, iter_num, max_iter):
    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    total_loss = model_instance.get_loss(inputs, labels_source)
    total_loss[0].backward()
    optimizer.step()
    return [total_loss[0].cpu().data.numpy(), total_loss[1].cpu().data.numpy(), total_loss[2].cpu().data.numpy()]

if __name__ == '__main__':
    from model.TCL import TCL
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
    elif args.dataset == 'COVID-19':
        class_num = 3
        width = 256
        srcweight = 4
        is_cen = False
        # Another choice for Office-home:
        # width = 1024
        # srcweight = 3
        # is_cen = True
    else:
        width = -1

    model_instance = TCL(base_net='ResNet50', width=width, use_gpu=True, class_num=class_num, srcweight=srcweight)

    train_source_clean_loader = load_images(source_file, batch_size=32, is_cen=is_cen, split_noisy=False)
    train_source_noisy_loader = train_source_clean_loader
    train_target_loader = load_images(target_file, batch_size=32, is_cen=is_cen)
    test_target_loader = load_images(target_file, batch_size=32, is_train=False)

    param_groups = model_instance.get_parameter_list()
    group_ratios = [group['lr'] for group in param_groups]

    assert cfg.optim.type == 'sgd', 'Optimizer type not supported!'

    optimizer = torch.optim.SGD(param_groups, **cfg.optim.params)

    assert cfg.lr_scheduler.type == 'inv', 'Scheduler type not supported!'
    lr_scheduler = INVScheduler(gamma=cfg.lr_scheduler.gamma,
                                decay_rate=cfg.lr_scheduler.decay_rate,
                                init_lr=cfg.init_lr)
    to_dump = train(model_instance, train_source_clean_loader, train_source_noisy_loader, train_target_loader, test_target_loader, group_ratios, max_iter=20000, optimizer=optimizer, lr_scheduler=lr_scheduler, eval_interval=1000)
    pickle.dump(to_dump, open(args.stats_file, 'wb'))
