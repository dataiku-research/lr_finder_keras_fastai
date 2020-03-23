import math
import random
import numpy as np
from scipy.misc import imresize

from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator


# RandomResizedCrop() from fastai
class RandomResizedCrop(object):
    """ RandomResizedCrop augmentation class"""
    
    def __init__(self, size, min_scale = 0.08):
        self.min_scale = min_scale
        self.size = size
    
    def __call__(self, img):
        
        aspect_ratio=(3/4, 4/3)
        
        w,h,c = img.shape
        rand_scale = np.random.uniform(self.min_scale, 1.)
        area = rand_scale * w * h
        
        ratio = math.exp(random.uniform(math.log(aspect_ratio[0]), math.log(aspect_ratio[1])))
        
        nw = int(round(math.sqrt(area * ratio)))
        nh = int(round(math.sqrt(area / ratio)))
        if nw <= w and nh <= h:
            cp_size = (nw,nh)
            tl = random.randint(0,w-nw), random.randint(0,h - nh)
        else:
            if   w/h < aspect_ratio[0]: cp_size = (w, int(w/aspect_ratio[0]))
            elif w/h > aspect_ratio[1]: cp_size = (int(h*aspect_ratio[1]), h)
            else: cp_size = (w, h)
            tl = ((w-cp_size[0])//2, (h-cp_size[1])//2)
        
        img = img[tl[0]:tl[0]+cp_size[0], tl[1]:tl[1]+cp_size[1], :]
        img = imresize(img, self.size, interp='bilinear', mode='RGB').astype('float64')
                
        return img


def get_cifar_data(num_classes=10):
    assert num_classes in [10, 100]
    if num_classes==10:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # fastai uses -mean/std
    #means = np.mean(x_train, axis=(0, 1, 2))
    #stds = np.std(x_train, axis=(0, 1, 2))
    #x_train = (x_train - means) / stds
    #x_test = (x_test - means) / stds

    # fastai random horizontal flip and reflection padding of 4 pixels.
    aug = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True,
                             fill_mode="nearest")

    return x_train, y_train, x_test, y_test, aug

