import os
import numpy as np
from numpy.testing import assert_array_almost_equal
from imgaug import augmenters as iaa
from PIL import Image
import random
import tqdm
import glob

def listing_file(domains, data_dir, save_dir):
    for d in domains:
        str_labels, num_labels = [], []
        file_dir = data_dir + d + '/*/*'
        save_file = save_dir + d + '.txt'
        img_files = glob.glob(file_dir, recursive=True)
        label, first = 0, True
        for img in img_files:
            str_label = img.split('/')[7]
            if first:
                num_labels.append(label)
                str_labels.append(str_label)
                first = False
            else:
                if str_label not in str_labels:
                    label += 1
                    str_labels.append(str_label)
                    num_labels.append(label)
                else:
                    num_labels.append(label)
        with open(save_file,'w') as f:
            for i, img in enumerate(img_files):
                f.write('{} {}\n'.format(img, num_labels[i]))


def multiclass_noisify(y, T, random_state=0):
    """
    Flip classes according to transition probability matrix T.
    """
    y = np.asarray(y)
    #print(np.max(y), T.shape[0])
    assert T.shape[0] == T.shape[1]
    assert np.max(y) < T.shape[0]
    assert_array_almost_equal(T.sum(axis=1), np.ones(T.shape[1]))
    assert (T >= 0.0).all()
    m = y.shape[0]
    #print(m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)
    for idx in np.arange(m):
        i = y[idx]
        #print(i, T[i,:])
        flipped = flipper.multinomial(1, T[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]
    return new_y

def symmetric_noisy(y, noisy_rate, random_state=None, class_number=31):
    """
    flip y in the symmetric way
    """
    r = noisy_rate
    T = np.ones((class_number, class_number))
    T = (r / (class_number - 1)) * T
    if r > 0.0:
        T[0, 0] = 1. - r
        for i in range(1, class_number-1):
            T[i, i] = 1. - r
        T[class_number-1, class_number-1] = 1. - r
        y_noisy = multiclass_noisify(y, T=T, random_state=random_state)
        actural_noise_rate = (y != y_noisy).mean()
        assert actural_noise_rate > 0.0
        print('Actural noisy rate is {}'.format(actural_noise_rate))
    #print(T)
    return y_noisy, actural_noise_rate

def asymmetric_noisy(y, noisy_rate, random_state=None, class_number=31):

    """
    flip in the pair
    """
    T = np.eye(class_number)
    r = noisy_rate
    if r > 0.0:
        T[0,0], T[0,1] = 1. - r, r
        for i in range(1, class_number-1):
            T[i,i], T[i,i+1] = 1.- r, r
        T[class_number-1, class_number-1], T[class_number-1,0] = 1. - r, r

        y_noisy = multiclass_noisify(y, T=T, random_state=random_state)
        actural_noise_rate = (y != y_noisy).mean()
        assert actural_noise_rate > 0.0
        print('Actural noisy rate is {}'.format(actural_noise_rate))
    #print(T)
    return y_noisy, actural_noise_rate

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def sp_blur_noise(image):
    '''
    Add salt and pepper noise to image and gaussian bluring image.
    '''
    image = np.asarray(image)
    sp_blur = iaa.Sequential([iaa.GaussianBlur(sigma=8.00),
        iaa.CoarseSaltAndPepper(p=0.5, size_percent=0.04)])
    output = sp_blur.augment_image(image)
    output = Image.fromarray(output)
    return output

def corrupt_image(file_dir):
    for path in tqdm.tqdm(file_dir):
        save_path = path.split('.')[0] + '_corrupted.jpg'
        if not os.path.isfile(save_path):
            image = pil_loader(path)
            image = sp_blur_noise(image)
            image.save(save_path)
    print('complete corrupting images!')

#def ood_noise():


if __name__ == '__main__':

    dataset = 'office-31' # or office-home
    if dataset is 'office-home':
        class_number = 65
        data_files = ['Office-home/Art.txt', 'Office-home/Clipart.txt', 'Office-home/Product.txt', 'Office-home/Real_world.txt']
        domains = ['Art', 'Clipart', 'Product', 'Real_world']
        data_dir = '/data1/hanzhongyi/datasets/office/office-home/'
        save_dir = 'Office-home/'
    elif dataset is 'office-31':
        class_number = 31
        data_files = ['Office-31/webcam.txt', 'Office-31/dslr.txt', 'Office-31/amazon.txt']
        domains = ['webcam', 'dslr', 'amazon']
        data_dir = '/data1/hanzhongyi/datasets/office/office-31/'
        save_dir = 'Office-31/'
    else:
        raise Exception("Sorry, unsupported dataset")

    listing_file = False
    corrupt_image = False


    #listing source files of images into text if not exists.
    if listing_file:
        listing_file(domains, data_dir, save_dir)
    
    for data_file in data_files: #Read text files
        with open(data_file, 'r') as f:
            file_dir, label = [], []
            for i in f.read().splitlines():
                file_dir.append(i.split(' ')[0])
                label.append(int(i.split(' ')[1]))
        if corrupt_image:
            corrupt_image(file_dir)

        #Todo: noisy label
        """
        noisy_rate = [0.2,0.4,0.6,0.8]
        noisy_type = ['uniform', 'pair']
        for tp in noisy_type:
            for rate in noisy_rate:
                if tp is 'pair':
                    label_noisy, acutal_noisy_rate = asymmetric_noisy(label, rate, class_number=class_number)
                elif tp is 'uniform':
                    label_noisy, acutal_noisy_rate = symmetric_noisy(label, rate, class_number=class_number)
                print('generate noisy rate:', acutal_noisy_rate)
                #save all the images in a txt
                save_file = data_file.split('.')[0] + '_{}_noisy_{}.txt'.format(tp, rate)
                with open(save_file,'w') as f:
                    for i, d in enumerate(file_dir):
                        f.write('{} {} {}\n'.format(d, label_noisy[i], label[i]))

                '''split data into noisy and clean data for validating the idea'''
                #save_clean_file = data_file.split('.')[0] + '_{}_noisy_{}_true.txt'.format(tp, rate)
                #save_noisy_file = data_file.split('.')[0] + '_{}_noisy_{}_false.txt'.format(tp, rate)
                #with open(save_clean_file,'w') as f:
                #    with open(save_noisy_file, 'w') as ff:
                #        for i, d in enumerate(file_dir):
                #            if label[i] == label_noisy[i]:
                #                f.write('{} {}\n'.format(d, label_noisy[i]))
                #            else:
                #                ff.write('{} {}\n'.format(d, label_noisy[i]))

        print('complete corrupting labels!')
        """
        """
        #Todo: noisy feature
        noisy_feature_rate = [0.1,0.2,0.3,0.4,0.6,0.8]
        for rate in noisy_feature_rate:
            save_file = data_file.split('.')[0] + '_feature_noisy_{}.txt'.format(rate)
            num = 0
            with open(save_file, 'w') as f:
                for i, d in enumerate(file_dir):
                    rdn = random.random()
                    if rdn < rate:
                        num+=1
                        d = d.split('.')[0] + '_corrupted.jpg'
                    f.write('{} {}\n'.format(d, label[i]))
            print('feature noise true/given {}/{}'.format(float(num)/len(file_dir), rate))
        print('complete corrupting features!')
        """
        #Todo: introduce out-of-distribution (ood) noise
        ood_dataset = "Bing-Caltech/Caltech.txt" #Introduce other dataset for providing 
        with open(ood_dataset, 'r') as f:
            ood_file_dir, ood_label = [], []
            for i in f.read().splitlines():
                ood_file_dir.append(i.split(' ')[0])
                ood_label.append(int(i.split(' ')[1]))
        ood_noisy_rate = [0.1,0.2,0.3,0.4,0.6,0.8]
        for rate in ood_noisy_rate:
            save_file = data_file.split('.')[0] + '_ood_noisy_{}.txt'.format(rate)
            #know every class number and add %rate
            num_per_class = np.zeros(class_number)
            for i in label:
                num_per_class[i] += 1
            #Todo: compute num_per_class
            num_per_class = np.ceil(num_per_class*(rate/(1-rate)))       
            random.shuffle(ood_file_dir)
            new_file_dir = ood_file_dir[:int(num_per_class.sum())]
            new_label = []
            for i, j in enumerate(num_per_class):
                new_label.extend([i]*int(j))
            
            with open(save_file,'w') as f:
                    for i, d in enumerate(file_dir):
                        f.write('{} {}\n'.format(d, label[i]))
                    for i, d in enumerate(new_file_dir):
                        f.write('{} {}\n'.format(d, new_label[i]))

        """
        #Todo mix: feature noise + label noise
        for rate in noisy_rate:
            rate = rate #for fair comparison with TCL
            feature_noisy_file = data_file.split('.')[0] + '_feature_noisy_{}.txt'.format(rate/2)
            with open(feature_noisy_file, 'r') as f:
                file_dir, label = [], []
                for i in f.read().splitlines():
                    file_dir.append(i.split(' ')[0])
                    label.append(int(i.split(' ')[1]))
            #noisy label
            for tp in noisy_type:
                if tp is 'pair':
                    label_noisy, acutal_noisy_rate = asymmetric_noisy(label, rate/2, class_number=class_number)
                elif tp is 'uniform':
                    label_noisy, acutal_noisy_rate = symmetric_noisy(label, rate/2, class_number=class_number)
                print('generate noisy rate:', acutal_noisy_rate)
                #save all the images in a txt
                save_file = data_file.split('.')[0] + '_feature_{}_noisy_{}.txt'.format(tp, rate)
                with open(save_file,'w') as f:
                    for i, d in enumerate(file_dir):
                        f.write('{} {} {}\n'.format(d, label_noisy[i], label[i]))

                #save split noisy and clean data for validate the idea
                save_clean_file = data_file.split('.')[0] + '_feature_{}_noisy_{}_true.txt'.format(tp, rate)
                save_noisy_file = data_file.split('.')[0] + '_feature_{}_noisy_{}_false.txt'.format(tp, rate)
                with open(save_clean_file,'w') as f:
                    with open(save_noisy_file, 'w') as ff:
                        for i, d in enumerate(file_dir):
                            if label[i] == label_noisy[i]:
                                f.write('{} {}\n'.format(d, label_noisy[i]))
                            else:
                                ff.write('{} {}\n'.format(d, label_noisy[i]))
        """ 
        #Todo mix: feature noise + label noise + ood noise