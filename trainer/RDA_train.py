import tqdm
import argparse
from torch.autograd import Variable
import torch
import warnings
import random
warnings.filterwarnings("ignore")
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
        probabilities, _ = model_instance.predict(inputs)
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
    model_instance.set_train(ori_train_state)
    return {'accuracy':accuracy}

def train(model_instance, train_source_clean_loader, train_source_noisy_loader, train_target_loader, test_target_loader, group_ratios, max_iter, optimizer, lr_scheduler, eval_interval, del_rate=0.4):
    model_instance.set_train(True)
    print("start train...")
    #model_instance.c_net.load_state_dict(torch.load('A2D_iter_10k.pth'))
    loss = [] #accumulate total loss for visulization.
    result = [] #accumulate eval result on target data during training.
    iter_num = 0
    epoch = 0
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=max_iter)
    while True:
        datas_noisy =  iter(train_source_noisy_loader)
        for (datas_clean, datat) in tqdm.tqdm(
                zip(train_source_clean_loader, train_target_loader),
                total=min(len(train_source_clean_loader), len(train_target_loader)),
                desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):
            try:
                inputs_source_noisy, labels_source_noisy, _ = datas_noisy.next()
            except:
                #datas_noisy =  iter(train_source_noisy_loader)
                inputs_source_noisy, labels_source_noisy, _ = datas_clean
            inputs_source, labels_source, _ = datas_clean
            
            inputs_target, _, _ = datat

            optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, iter_num/5)
            optimizer.zero_grad()

            if model_instance.use_gpu:
                inputs_source, inputs_source_noisy, inputs_target, labels_source, labels_source_noisy = Variable(inputs_source).cuda(),  Variable(inputs_source_noisy).cuda(), Variable(inputs_target).cuda(), Variable(labels_source).cuda(), Variable(labels_source_noisy).cuda()
            else:
                inputs_source, inputs_source_noisy, inputs_target, labels_source, labels_source_noisy = Variable(inputs_source),  Variable(inputs_source_noisy), Variable(inputs_target), Variable(labels_source), Variable(labels_source_noisy)

            total_loss = train_batch(model_instance, inputs_source, labels_source, inputs_target, optimizer, max_iter, del_rate, inputs_source_noisy)

            #val
            if iter_num % eval_interval == 0 and iter_num != 0:
                eval_result = evaluate(model_instance, test_target_loader)
                print(eval_result)
                result.append(eval_result['accuracy'].cpu().data.numpy())

            iter_num += 1
            total_progress_bar.update(1)
            loss.append(total_loss)

        epoch += 1

        if iter_num > max_iter:
            break
    print('finish train')
    #torch.save(model_instance.c_net.state_dict(), 'statistic/Ours_model.pth')
    return [loss, result]

def train_batch(model_instance, inputs_source, labels_source, inputs_target, optimizer, max_iter, del_rate, inputs_source_noisy):
    inputs = torch.cat((inputs_source, inputs_target, inputs_source_noisy), dim=0)
    total_loss = model_instance.get_loss(inputs, labels_source, max_iter, del_rate=del_rate, noisy_source_num=len(inputs_source_noisy))
    total_loss[0].backward()
    optimizer.step()
    return [total_loss[0].cpu().data.numpy(), total_loss[1].cpu().data.numpy(), total_loss[2].cpu().data.numpy(), total_loss[3].cpu().data.numpy(), total_loss[4].cpu().data.numpy()]

if __name__ == '__main__':
    from model.RDA import PMD
    from preprocess.data_provider_new import load_images
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
    else:
        width = -1

    model_instance = PMD(base_net='ResNet50', width=width, use_gpu=True, class_num=class_num, srcweight=srcweight)
    if args.noisy_rate == 0.:
        train_source_clean_loader = load_images(source_file, batch_size=32, is_cen=is_cen, split_noisy=False)
        train_source_noisy_loader = train_source_clean_loader
    else:
        train_source_clean_loader, train_source_noisy_loader = load_images(source_file, batch_size=32, is_cen=is_cen, split_noisy=True)
    train_target_loader = load_images(target_file, batch_size=32, is_cen=is_cen) #is_cen means preprocess of center crop.
    test_target_loader = load_images(target_file, batch_size=32, is_train=False)

    param_groups = model_instance.get_parameter_list()
    group_ratios = [group['lr'] for group in param_groups]

    assert cfg.optim.type == 'sgd', 'Optimizer type not supported!'

    optimizer = torch.optim.SGD(param_groups, **cfg.optim.params)

    assert cfg.lr_scheduler.type == 'inv', 'Scheduler type not supported!'
    lr_scheduler = INVScheduler(gamma=cfg.lr_scheduler.gamma,
                                decay_rate=cfg.lr_scheduler.decay_rate,
                                init_lr=cfg.init_lr)
    to_dump = train(model_instance, train_source_clean_loader, train_source_noisy_loader, train_target_loader, test_target_loader, group_ratios, max_iter=20000, optimizer=optimizer, lr_scheduler=lr_scheduler, eval_interval=1000, del_rate=args.del_rate)
    pickle.dump(to_dump, open(args.stats_file, 'wb'))
