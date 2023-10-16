import pickle
import numpy as np
from TwoLayerNet import *
import matplotlib.pyplot as plt

def SGD(net,train_set,train_labels,lr = 0.01,epoch = 100,batch = 100,alpha = False):
    loss_list = []
    acc_list = []
    num = train_set.shape[0]
    if num % batch != 0:
        print('batch error!')
        exit(0)
    shuffle_train_set = np.zeros_like(train_set)
    shuffle_train_labels = np.zeros_like(train_labels)
    for i in range(epoch):
        print(i+1,'epoch(s) is learning!')
        shuffle = np.arange(train_set.shape[0])
        np.random.shuffle(shuffle)
        for k in range(train_set.shape[0]):
            shuffle_train_set[k] = train_set[shuffle[k]]
            shuffle_train_labels[k] = train_labels[shuffle[k]]
        for j in range(int(num/batch)):
            print('123')
            x = train_set[batch*j:batch*j+batch,:]
            labels = train_labels[batch*j:batch*j+batch]
            grads = net.gradient(x,labels)
            if alpha:
                for key in grads.keys():
                    grads[key] = grads[key] + alpha * net.params[key]

            for param in grads.keys():
                print(grads[param])

            for param in grads.keys():
                grads[param] = -lr*grads[param]
            net.update_params(grads)

            # if j==50 or j==100 or j==150 or j==200 or j==250 or j==300 or j==350 or j==400 or j==450 or j==500 or j==550:
            print(net.loss(x, labels))
            file = open('./Params/paper_params', 'wb')
            pickle.dump(net.params, file)
            file.close()


        # loss = net.loss(x, labels)
        # print('---loss:', loss)
        # loss_list.append(loss)
        # acc = net.accuracy(x, labels)
        # print('---acc:', acc)
        # acc_list.append(acc)
    file = open('./Params/final_params','wb')
    pickle.dump(net.params,file)
    file.close()

    # visualization
    # plt.figure(1)
    # plt.plot(loss_list)
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.figure(2)
    # plt.plot(acc_list)
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    #
    # plt.show()

def Momentum(net,train_set,train_labels,lr = 0.01,beta = 0.9,epoch = 100,batch = 100,alpha = False):
    num = train_set.shape[0]
    if num % batch != 0:
        print('batch error!')
        exit(0)
    shuffle_train_set = np.zeros_like(train_set)
    shuffle_train_labels = np.zeros_like(train_labels)
    grads = {}
    for key,value in net.params.items():
        grads[key] = np.zeros_like(value)
    for i in range(epoch):
        print(i+1,'epoch(s) is learning!')
        shuffle = np.arange(train_set.shape[0])
        np.random.shuffle(shuffle)
        for k in range(train_set.shape[0]):
            shuffle_train_set[k] = train_set[shuffle[k]]
            shuffle_train_labels[k] = train_labels[shuffle[k]]
        for j in range(int(num/batch)):
            x = train_set[batch*j:batch*j+batch,:]
            labels = train_labels[batch*j:batch*j+batch,:]
            new_grads = net.gradient(x,labels)
            if alpha:
                for key in grads.keys():
                    new_grads[key] = new_grads[key] + alpha * net.params[key]
            for param in net.params.keys():
                grads[param] = beta*grads[param]-lr*new_grads[param]
            net.update_params(grads)
    file = open('./Params/Net_params','wb')
    pickle.dump(net.params,file)
    file.close()

def AdaGrad(net,train_set,train_labels,lr = 0.01,epoch = 100,batch = 100,alpha = False):
    loss_list = []
    acc_list = []
    num = train_set.shape[0]
    if num % batch != 0:
        print('batch error!')
        exit(0)
    sum_grads = {}
    for key, value in net.params.items():
        sum_grads[key] = np.zeros_like(value)
    shuffle_train_set = np.zeros_like(train_set)
    shuffle_train_labels = np.zeros_like(train_labels)
    for i in range(epoch):
        print(i + 1, 'epoch(s) is learning!')
        shuffle = np.arange(train_set.shape[0])
        np.random.shuffle(shuffle)
        for k in range(train_set.shape[0]):
            shuffle_train_set[k] = train_set[shuffle[k]]
            shuffle_train_labels[k] = train_labels[shuffle[k]]
        for j in range(int(num / batch)):
            x = shuffle_train_set[batch * j:batch * j + batch]
            labels = train_labels[batch * j:batch * j + batch]
            grads = net.gradient(x, labels)
            if alpha:
                for key in grads.keys():
                    grads[key] = grads[key] + alpha * net.params[key]
            for key in grads.keys():
                # multiply exp(-1) to forget previous gradient
                sum_grads[key] = np.exp(-1) * sum_grads[key]
                sum_grads[key] = sum_grads[key] + grads[key]*grads[key]
            for param in grads.keys():
                grads[param] = -lr * 1/(np.sqrt(sum_grads[param])+np.exp(-7)) * grads[param]
            net.update_params(grads)

            print(net.loss(x, labels))
            file = open('./Params/paper_params', 'wb')
            pickle.dump(net.params, file)
            file.close()



        loss = net.loss(x,labels)
        print('---loss:',loss)
        loss_list.append(loss)
        acc = net.accuracy(x,labels)
        print('---acc:',acc)
        acc_list.append(acc)
    file = open('./Params/Net_params', 'wb')
    pickle.dump(net.params, file)
    file.close()

    plt.figure(1)
    plt.plot(loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.figure(2)
    plt.plot(acc_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.show()