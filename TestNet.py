from Loader import *
from TwoLayerNet import *
import pickle
import Visualization
import numpy as np
import time
from PIL import Image

test_images = Test_images_load('./DataSet/t10k-images.idx3-ubyte')
test_images = test_images.reshape((-1,1,28,28))
test_labels = Test_labels_load('./DataSet/t10k-labels.idx1-ubyte')
print('Data Loaded!')
image = Image.open('./mario_gray.jpg')
zzy = np.zeros((1,1,28,28))
for i in range(28):
    for j in range(28):
        zzy[0,0,i,j] = image.getpixel((i,j))/255

net = TwoLayerNet(4320,100,10)

test = np.zeros((100,1,28,28))
test[:,:,:,:] = test_images[0:100]
print(net.loss(zzy,test_labels))

file = open('./Params/paper_params','rb')
net_params = pickle.load(file)
net.set_params(net_params)
file.close()
# print(net_params['W2'])
# print('----------------')

print(net.loss(zzy,test_labels))

# compute accuracy
# start = time.clock()
# print('accuracy:',net.accuracy(test_images,test_labels))
# end = time.clock()
# print(end - start)

# print('loss:',net.loss(test_images,test_labels))
# print('loss:',net.loss(test_images,test_labels))

#predict
# predict = net.predict(test_images)

#visualization
# Visualization.visualization(test_images,predict)

