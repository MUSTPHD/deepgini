#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist,cifar10,cifar100
from tensorflow.keras.models import load_model
import metrics
import time
import os
import tensorflow as tf
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
physical_devices = tf.config.list_physical_devices('GPU') 
print('-----', len(physical_devices))
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

def gen_data(use_adv=True,deepxplore=False):
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()  # 28*28
    X_train = X_train.astype('float32').reshape(-1,28,28,1)
    X_test = X_test.astype('float32').reshape(-1,28,28,1)
    X_train /= 255
    X_test /= 255
    model_path='./model/model_mnist.hdf5'
    if use_adv:
        attack_lst=['fgsm','jsma','bim','cw']
        attack_lst=['jsma','bim','cw']
        adv_image_all=[]
        adv_label_all=[]
        for attack in attack_lst:
            adv_image_all.append(np.load('./adv_image/{}_mnist_image.npy'.format(attack)))
            adv_label_all.append(np.load('./adv_image/{}_mnist_label.npy'.format(attack)))
        adv_image_all=np.concatenate(adv_image_all,axis=0)
        adv_label_all=np.concatenate(adv_label_all,axis=0)
        test=np.concatenate([X_test,adv_image_all],axis=0)
        true_test=np.concatenate([Y_test,adv_label_all],axis=0)
    else:
        test=X_test
        true_test=Y_test
    train=X_train
    model=load_model(model_path)
    pred_test_prob=model.predict(test)
    pred_test=np.argmax(pred_test_prob,axis=1)
    input=model.layers[0].output
    if not deepxplore:
        layers=[model.layers[2].output,model.layers[3].output,model.layers[5].output,model.layers[6].output,model.layers[8].output,model.layers[9].output,model.layers[10].output]
    else:
        layers=[model.layers[1].output,model.layers[3].output,model.layers[4].output,model.layers[8].output,model.layers[8].output,model.layers[9].output,model.layers[10].output]


    layers=list(zip(4*['conv']+3*['dense'],layers))

    return input,layers,test,train,pred_test,true_test,pred_test_prob

def exp(coverage,use_adv,std=0,k=1,k_bins=1000):
    input,layers,test,train,pred_test,true_test,pred_test_prob=gen_data(use_adv)
    rank_lst2=None
    if coverage=='kmnc':
        # 只能贪心排
        km=metrics.kmnc(train,input,layers,k_bins=k_bins)
        rank_lst=km.rank_fast(test)
        rate=km.fit(test)
    elif coverage=='nbc':
        #可以贪心排
        #可以单个样本比较排
        #0 0.5 1
        bc=metrics.nbc(train,input,layers,std=std)
        rank_lst=bc.rank_fast(test,use_lower=True)
        rank_lst2=bc.rank_2(test,use_lower=True)
        rate=bc.fit(test,use_lower=True)
    elif coverage=='snac':
        #可以贪心排
        #可以单个样本比较排
        #0 0.5 1
        bc=metrics.nbc(train,input,layers,std=std)
        rank_lst=bc.rank_fast(test,use_lower=False)
        rank_lst2=bc.rank_2(test,use_lower=False)
        rate=bc.fit(test,use_lower=False)
    elif coverage=='tknc':
        # 只能贪心排
        #1 2 3
        tk=metrics.tknc(test,input,layers,k=k)
        rank_lst=tk.rank(test)
        rate=tk.fit(list(range(len(test))))

    df=pd.DataFrame([])
    df['right']=(pred_test==true_test).astype('int')
    df['cam']=0
    df['cam'].loc[rank_lst]=list(range(1,len(rank_lst)+1))
    if rank_lst2 is not None:
        df['ctm']=0
        df['ctm'].loc[rank_lst2]=list(range(1,len(rank_lst2)+1))
    if use_adv:
        dataset='mnist_adv'
    else:
        dataset='mnist'
    df['rate']=rate
    if coverage=='kmnc':
        pass
        #df.to_csv('./output_mnist/{}_{}_k_bins_{}.csv'.format(dataset,coverage,k_bins))
    elif coverage=='nbc':
        pass
        #df.to_csv('./output_mnist/{}_{}_std_{}.csv'.format(dataset,coverage,std))
    elif coverage=='snac':
        pass
        #df.to_csv('./output_mnist/{}_{}_std_{}.csv'.format(dataset,coverage,std))
    elif coverage=='tknc':
        pass
        #df.to_csv('./output_mnist/{}_{}_k_{}.csv'.format(dataset,coverage,k))

def exp_nac(use_adv,t):
    input,layers,test,train,pred_test,true_test,pred_test_prob=gen_data(use_adv,deepxplore=True)
    rank_lst2=None
    #可以贪心排
    #可以单个样本比较排
    #0 0.5 1
    ac=metrics.nac(test,input,layers,t=t)
    rate=ac.fit()
    rank_lst=ac.rank_fast(test)
    rank_lst2=ac.rank_2(test)

    df=pd.DataFrame([])
    df['right']=(pred_test==true_test).astype('int')
    df['cam']=0
    df['cam'].loc[rank_lst]=list(range(1,len(rank_lst)+1))
    if rank_lst2 is not None:
        df['ctm']=0
        df['ctm'].loc[rank_lst2]=list(range(1,len(rank_lst2)+1))
    if use_adv:
        dataset='mnist_adv'
    else:
        dataset='mnist'
    df['rate']=rate

    df.to_csv('./output_mnist/{}_nac_t_{}.csv'.format(dataset,t))

def exp_deep_metric(use_adv):
    input,layers,test,train,pred_test,true_test,pred_test_prob=gen_data(use_adv,deepxplore=False)
    rank_lst=metrics.deep_metric(pred_test_prob)
    df=pd.DataFrame([])
    df['right']=(pred_test==true_test).astype('int')
    df['cam']=0
    df['cam'].loc[rank_lst]=list(range(1,len(rank_lst)+1))
    df['rate']=0
    if use_adv:
        dataset='mnist_adv'
    else:
        dataset='mnist'
    df.to_csv('./output_mnist/{}_deep_metric.csv'.format(dataset))


def exp_deep_L1(use_adv):
    input,layers,test,train,pred_test,true_test,pred_test_prob=gen_data(use_adv,deepxplore=False)
    rank_lst=metrics.deep_L1(pred_test_prob)
    df=pd.DataFrame([])
    df['right']=(pred_test==true_test).astype('int')
    df['cam']=0
    df['cam'].loc[rank_lst]=list(range(1,len(rank_lst)+1))
    df['rate']=0
    if use_adv:
        dataset='mnist_adv'
    else:
        dataset='mnist'
    df.to_csv('./output_mnist/{}_deep_L1.csv'.format(dataset))


if __name__=='__main__':
    pass
    dic={}

    # start=time.time()
    # exp_nac(use_adv=False,t=0)
    # end=time.time()
    # dic['mnist_nac_t_0']=(start-end)


    # start=time.time()
    # exp_nac(use_adv=True,t=0)
    # end=time.time()
    # dic['mnist_adv_nac_t_0']=(start-end)

    # start=time.time()
    # exp_nac(use_adv=False,t=0.75)
    # end=time.time()
    # dic['mnist_nac_t_0.75']=(start-end)

    # start=time.time()
    # exp_nac(use_adv=True,t=0.75)
    # end=time.time()
    # dic['mnist_adv_nac_t_0.75']=(start-end)

    #exp(coverage='kmnc',use_adv=False,k_bins=1000)
    #exp(coverage='kmnc',use_adv=False,k_bins=10000)
    #exp(coverage='kmnc',use_adv=True,k_bins=1000)
    #exp(coverage='kmnc',use_adv=True,k_bins=10000)

    start=time.time()
    exp_deep_metric(use_adv=False)
    end=time.time()
    dic['mnist_ours']=(start-end)

    start=time.time()
    exp_deep_metric(use_adv=True)
    end=time.time()
    dic['mnist_adv_ours']=(start-end)

    start=time.time()
    exp_deep_L1(use_adv=False)
    end=time.time()
    dic['mnist_L1']=(start-end)
    
    start=time.time()
    exp_deep_L1(use_adv=True)
    end=time.time()
    dic['mnist_adv_L1']=(start-end)

    # start=time.time()
    # exp(coverage='tknc',use_adv=False,k=1)
    # end=time.time()
    # dic['mnist_tknc_k_1']=(start-end)

    # start=time.time()
    # exp(coverage='tknc',use_adv=False,k=2)
    # end=time.time()
    # dic['mnist_tknc_k_2']=(start-end)

    # start=time.time()
    # exp(coverage='tknc',use_adv=False,k=3)
    # end=time.time()
    # dic['mnist_tknc_k_3']=(start-end)

    # start=time.time()
    # exp(coverage='tknc',use_adv=True,k=1)
    # end=time.time()
    # dic['mnist_adv_tknc_k_1']=(start-end)

    # start=time.time()
    # exp(coverage='tknc',use_adv=True,k=2)
    # end=time.time()
    # dic['mnist_adv_tknc_k_2']=(start-end)

    # start=time.time()
    # exp(coverage='tknc',use_adv=True,k=3)
    # end=time.time()
    # dic['mnist_adv_tknc_k_3']=(start-end)

    # start=time.time()
    # exp(coverage='nbc',use_adv=False,std=0.5)
    # end=time.time()
    # dic['mnist_nbc_std_0.5']=(start-end)


    # start=time.time()
    # exp(coverage='nbc',use_adv=False,std=1)
    # end=time.time()
    # dic['mnist_nbc_std_1']=(start-end)

    # start=time.time()
    # exp(coverage='nbc',use_adv=False,std=0)
    # end=time.time()
    # dic['mnist_nbc_std_0']=(start-end)



    # start=time.time()
    # exp(coverage='nbc',use_adv=True,std=0.5)
    # end=time.time()
    # dic['mnist_adv_nbc_std_0.5']=(start-end)

    # start=time.time()
    # exp(coverage='nbc',use_adv=True,std=1)
    # end=time.time()
    # dic['mnist_adv_nbc_std_1']=(start-end)

    # start=time.time()
    # exp(coverage='nbc',use_adv=True,std=0)
    # end=time.time()
    # dic['mnist_adv_nbc_std_0']=(start-end)


    # start=time.time()
    # exp(coverage='snac',use_adv=False,std=0.5)
    # end=time.time()
    # dic['mnist_snac_std_0.5']=(start-end)

    # start=time.time()
    # exp(coverage='snac',use_adv=False,std=1)
    # end=time.time()
    # dic['mnist_snac_std_1']=(start-end)

    # start=time.time()
    # exp(coverage='snac',use_adv=False,std=0)
    # end=time.time()
    # dic['mnist_snac_std_0']=(start-end)


    # start=time.time()
    # exp(coverage='snac',use_adv=True,std=0.5)
    # end=time.time()
    # dic['mnist_adv_snac_std_0.5']=(start-end)

    # start=time.time()
    # exp(coverage='snac',use_adv=True,std=1)
    # end=time.time()
    # dic['mnist_adv_snac_std_1']=(start-end)

    # start=time.time()
    # exp(coverage='snac',use_adv=True,std=0)
    # end=time.time()
    # dic['mnist_adv_snac_std_0']=(start-end)

    print(dic)


# {'mnist_nac_t_0': -16.287529468536377, 'mnist_adv_nac_t_0': -40.104525327682495, 'mnist_nac_t_0.75': -12.855152368545532, 'mnist_adv_nac_t_0.75': -44.695390939712524, 'mnist_ours': -1.0195002555847168, 'mnist_adv_ours': -2.595906972885132, 'mnist_tknc_k_1': -18.440149545669556, 'mnist_tknc_k_2': -15.049455165863037, 'mnist_tknc_k_3': -14.735110521316528, 'mnist_adv_tknc_k_1': -70.78050541877747, 'mnist_adv_tknc_k_2': -59.126999378204346, 'mnist_adv_tknc_k_3': -54.15341329574585, 'mnist_nbc_std_0.5': -29.539813995361328, 'mnist_nbc_std_1': -29.086140394210815, 'mnist_nbc_std_0': -33.23535990715027, 'mnist_adv_nbc_std_0.5': -64.28236150741577, 'mnist_adv_nbc_std_1': -60.84338617324829, 'mnist_adv_nbc_std_0': -82.47223234176636, 'mnist_snac_std_0.5': -28.5828800201416, 'mnist_snac_std_1': -27.237147092819214, 'mnist_snac_std_0': -28.553428411483765, 'mnist_adv_snac_std_0.5': -58.533217430114746, 'mnist_adv_snac_std_1': -56.78029537200928, 'mnist_adv_snac_std_0': -66.64217805862427}