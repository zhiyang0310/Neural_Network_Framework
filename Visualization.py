import matplotlib.pyplot as plt
import numpy as np
import pickle

def visualization(x,y):
    i = 0
    lisence = 'y'
    while(i < x.shape[0] and lisence == 'y'):
        file = open('./Params/mean_image','rb')
        mean_image = pickle.load(file)
        image = x[i] + mean_image.reshape(1,28,28)
        image = image*255
        plt.imshow(image[0])
        plt.show()
        predict = y[i,:]
        predict = np.argmax(predict)
        print(i,predict)
        i += 1