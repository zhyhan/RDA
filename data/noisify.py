




import os
import numpy as np
from numpy.testing import assert_array_almost_equal
from imgaug import augmenters as iaa
from PIL import Image
import random
import cv2
import tqdm

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


if __name__ == '__main__':
    # listing source files into txt if not exists.
    listing_file = False
    if listing_file:
        import glob
        domains = ['Art', 'Clipart', 'Product', 'Real_world']
        data_dir = '/data1/hanzhongyi/datasets/office/office-home/'
        save_dir = 'Office-home/'
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

    data_files = ['Office-home/Art.txt', 'Office-home/Clipart.txt', 'Office-home/Product.txt', 'Office-home/Real_world.txt']
    #data_files = ['Office-31/webcam.txt', 'Office-31/dslr.txt', 'Office-31/amazon.txt']
    #class_number = 31
    class_number = 65
    noisy_rate = [0.2,0.4,0.6,0.8]
    noisy_type = ['uniform', 'pair']
    corrupt_image = True
    for data_file in data_files:
        with open(data_file, 'r') as f:
            file_dir, label = [], []
            for i in f.read().splitlines():
                file_dir.append(i.split(' ')[0])
                label.append(int(i.split(' ')[1]))

        if corrupt_image:
            for path in tqdm.tqdm(file_dir):
                image = pil_loader(path)
                image = sp_blur_noise(image)
                save_path = path.split('.')[0] + '_corrupted.jpg'
                image.save(save_path)
            print('complete corrupting images!')

        #noisy label
        """
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

                #save split noisy and clean data for validate the idea
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
        #noisy feature
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

        #mix noise: feature noise + label noise
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
