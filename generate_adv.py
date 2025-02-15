#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tensorflow.keras
from tensorflow.keras import Model,Input
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation,Flatten
import math
import numpy as np
import pandas as pd
import foolbox
from tqdm import tqdm
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import scipy
import sys,os
import SVNH_DatasetUtil
import itertools
sys.path.append('./fashion-mnist/utils')
import warnings
warnings.filterwarnings("ignore")
import multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def adv_func(x,y,model_path='./model/model_mnist.hdf5',dataset='mnist',attack='fgsm'):
    tensorflow.keras.backend.set_learning_phase(0)
    model=load_model(model_path)
    foolmodel=foolbox.models.KerasModel(model,bounds=(0,1),preprocessing=(0,1))
    if attack=='cw':
        #attack=foolbox.attacks.IterativeGradientAttack(foolmodel)
        attack_func=foolbox.attacks.L2BasicIterativeAttack(foolmodel)
    elif attack=='fgsm':
        # FGSM
        attack_func=foolbox.attacks.GradientSignAttack(foolmodel)
    elif attack=='bim':
        # BIM
        attack_func=foolbox.attacks.L1BasicIterativeAttack(foolmodel)
    elif attack=='jsma':
        # JSMA
        attack_func=foolbox.attacks.SaliencyMapAttack(foolmodel)
        # CW
        #attack=foolbox.attacks.DeepFoolL2Attack(foolmodel)
    result=[]
    if dataset=='mnist':
        w,h=28,28
    elif dataset=='cifar10':
        w,h=32,32
    else:
        return False
    for image in tqdm(x):
        try:
            #adv=attack(image.reshape(28,28,-1),label=y,steps=1000,subsample=10)
            #adv=attack(image.reshape(w,h,-1),y,epsilons=[0.01,0.1],steps=10)
            if attack!='fgsm':
                img = image.reshape(w, h, -1)
                # from PIL import Image
                # im = Image.fromarray(img * 255)
                # im.save("img.jpeg")
                import imageio
                # imageio.imwrite('outfile.jpg', img)



                # img = image.reshape(-1, w, h)
                img = np.expand_dims(img, axis=0)
                adv=attack_func(img, np.array([y]))

                adv = adv[0]
                # imageio.imwrite('adv.jpg', adv)

                # adv=attack_func(image.reshape(w,h,-1),y)
                # adv=attack_func(image.reshape(w,h,-1),y)
            else:
                adv=attack_func(image.reshape(w,h,-1),y,[0.01,0.1])

            if isinstance(adv,np.ndarray):
                result.append(adv)
            else:
                print('adv fail')
        except Exception as e:
            print(e)
    return np.array(result)


def generate_mnist_sample(label,attack):
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()  # 28*28
    X_train = X_train.astype('float32').reshape(-1,28,28,1)
    X_test = X_test.astype('float32').reshape(-1,28,28,1)
    X_train /= 255
    X_test /= 255
    image_org=X_test[Y_test==label]
    adv=adv_func(image_org,label,model_path='./model/model_mnist.hdf5',dataset='mnist',attack=attack)
    return adv

def generate_cifar_sample(label,attack):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()  # 32*32
    X_train = X_train.astype('float32').reshape(-1,32,32,3)
    X_test = X_test.astype('float32').reshape(-1,32,32,3)
    X_train /= 255
    X_test /= 255

    Y_train=Y_train.reshape(-1)
    Y_test=Y_test.reshape(-1)

    image_org=X_test[Y_test==label]

    adv=adv_func(image_org,label,model_path='./model/cifar10_ResNet20v1_model.h5',dataset='cifar10',attack=attack)
    return adv

def generate_cifar100_sample(label,attack):
    (X_train, Y_train), (X_test, Y_test) = cifar100.load_data(label_mode='coarse')  # 32*32
    X_train = X_train.astype('float32').reshape(-1,32,32,3)
    X_test = X_test.astype('float32').reshape(-1,32,32,3)
    X_train /= 255
    X_test /= 255

    Y_train=Y_train.reshape(-1)
    Y_test=Y_test.reshape(-1)
    image_org=X_test[Y_test==label]

    adv=adv_func(image_org,label,model_path='./model/model_cifar20.h5',dataset='cifar10',attack=attack)
    return adv

def generate_fashion_sample(label,attack):
    import mnist_reader
    path='./fashion-mnist/data/fashion'
    X_train, Y_train = mnist_reader.load_mnist(path, kind='train')
    X_test, Y_test = mnist_reader.load_mnist(path, kind='t10k')
    X_train = X_train.astype('float32').reshape(-1,28,28,1)
    X_test = X_test.astype('float32').reshape(-1,28,28,1)
    X_train /= 255
    X_test /= 255

    image_org=X_test[Y_test==label]
    adv=adv_func(image_org,label,model_path='./model/model_fashion.hdf5',dataset='mnist',attack=attack)
    return adv

def generate_svhn_sample(label,attack):

    (X_train, Y_train), (X_test, Y_test) = SVNH_DatasetUtil.load_data()  # 32*32

    image_org=X_test[np.argmax(Y_test,axis=1)==label]

    adv=adv_func(image_org,label,model_path='./model/model_svhn.hdf5',dataset='cifar10',attack=attack)
    return adv




def generate_adv_sample(dataset,attack):
    if dataset=='mnist':
        sample_func=generate_mnist_sample
    elif dataset=='svhn':
        sample_func=generate_svhn_sample
    # elif dataset=='fashion':
    #     sample_func=generate_fashion_sample
    elif dataset=='cifar10':
        sample_func=generate_cifar_sample
    elif dataset=='cifar20':
        sample_func=generate_cifar100_sample
    else:
        print('erro')
        return
    image=[]
    label=[]
    for i in range(10):
        adv=sample_func(label=i,attack=attack)
        temp_image=adv
        temp_label=i*np.ones(len(adv))
        image.append(temp_image.copy())
        label.append(temp_label.copy())
    image=np.concatenate(image,axis=0)
    label=np.concatenate(label,axis=0)
    np.save('./adv_image/{}_{}_image'.format(attack,dataset),image)
    np.save('./adv_image/{}_{}_label'.format(attack,dataset),label)
if __name__=='__main__':
    '''
    mnist svhn fashion cifar10 cifar20
    cw fgsm bim jsma
    '''
    # data_lst=['svhn','fashion','cifar10','mnist']
    data_lst=['mnist']
    # attack_lst=['cw','fgsm','bim','jsma']

    # generate_adv_sample('cifar10', 'cw')
    # generate_adv_sample('cifar10', 'fgsm')
    # generate_adv_sample('cifar10', 'bim')
    # generate_adv_sample('cifar10', 'jsma')


    # generate_adv_sample('mnist', 'cw')
    generate_adv_sample('mnist', 'fgsm')
    # generate_adv_sample('mnist', 'bim')
    # generate_adv_sample('mnist', 'jsma')
    # pool = multiprocessing.Pool(processes=4)
    # for dataset,attack in (itertools.product(data_lst,attack_lst)):
    #     pool.apply_async(generate_adv_sample, (dataset,attack))
    # pool.close()
    # pool.join()
