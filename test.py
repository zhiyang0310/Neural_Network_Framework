import numpy as np
import Functions
from TwoLayerNet import *
import pickle
import matplotlib.pyplot as plt
import ComputeLayers
from PIL import Image
import csv



# a = np.array([1,2,3])
# b = np.array([[4,5,6]])
# a += b
# print(a)

# dict = {}
# dict['1'] = 1
# dict['2'] = 2
# dict['3'] = 3
#
# print(dict)
# for i in dict.values():
#     print(i)

# data = np.random.randn(2,3,3,3)
#
# image1 = np.arange(9).reshape(3,3)
# shuff = np.arange(5,14).reshape(3,3)
# data1 = np.random.randn(3,3,3)
# for i in range(3):
#     data1[i] = image1
# data1[2] = shuff
# image2 = np.arange(9,18).reshape(3,3)
# data2 = np.random.randn(3,3,3)
# for i in range(3):
#     data2[i] = image2
#
# data[0] = data1
# data[1] = data2
# print(data)
#
#
# col = Functions.im2col(data,3,3,3,0)
# print(col)
#
# aa = Functions.col2im(col,2,3,3,3,3,3,0)
# print(aa)

# data11 = np.array([[0,1],[2,3]])
# data12 = np.array([[4,5],[6,7]])
# data = np.zeros((1,2,2,2))
# data[0,0] = data11
# data[0,1] = data12
# Net = TwoLayerNet()
# y = Net.predict(data)
# data11 = np.array([[0,1,2],[3,4,5],[6,7,8]])
# data = np.zeros((1,1,3,3))
# data[0,0] = data11
# y = Net.layers['cov'].backward(data)
#
# print(Net.layers['cov'].db)

# a = np.arange(10)
# print(a)
# np.random.shuffle(a)
# print(a)

# plt.figure(1)
# plt.plot([5,200,89,8])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.figure(2)
# plt.plot([2,4,56,7])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()

# a = np.arange(0,28*28*100).reshape((10,10,28,28))
# print(a)
# capsule = Capsule(5)
# print(capsule.cap_dim)
# print('////////////////')
# print(capsule.forward(a))
# print('////////////////')
# b = capsule.backward(capsule.forward(a))
# print(a-b)


# b = np.zeros((2,2))
# layer = DynamicRouting(b)
# data = np.zeros((2,2,2,2))
# data[0,0] = np.array([[1,2],[3,4]])
# data[0,1] = np.array([[5,6],[7,8]])
# data[1,0] = np.array([[0,1],[2,3]])
# data[1,1] = np.array([[4,5],[6,7]])
#
# dout = (26**2)*np.eye(2,2)
#
# a = layer.forward(data)
# b = layer.backward(dout)
# print(b)

# b = np.zeros((2,2))
# data = np.zeros((2,2,2,2))
# data[0,0] = np.array([[1,2],[3,4]])
# data[0,1] = np.array([[5,6],[7,8]])
# data[1,0] = np.array([[0,1],[2,3]])
# data[1,1] = np.array([[4,5],[6,7]])
#
# cap = np.zeros((2,2,2))
# cap[0] = np.array([[0,1],[2,3]])
# cap[1] = np.array([[1,2],[3,4]])
#
# layer = Liner()
# a = layer.forward(cap)
# b = layer.backward(a)
#
# print(b)




# w = np.array([[1,2],[3,4]])
# layer = Weighting(w)
# a = layer.forward(data)
#
# print(a)

# layer = LosswithCorrelation()
# dout = np.zeros_like(data)
# dout[0,0] = data[1,0]
# dout[0,1] = np.array([[4,5],[2,3]])
# dout[1,0] = data[0,0]
# dout[1,1] = data[0,1]
# cap = np.zeros((2,2,2))
# cap[0] = np.array([[1,1],[2,2]])
# cap[1] = np.array([[0,1],[3,4]])
# dout = np.zeros_like(cap)
# dout[:] = cap[:]
# dout[1,1,0] = 0
# dout[1,1,1] = 4
# a = layer.forward(cap)
# b = layer.backward(dout)
# print(a)

# a = layer.forward(cap)
# b = layer.backward()
# print(a)

# image = Image.open('./IMG_1432.jpg')
# image.show()

# tt = open('/Volumes/Data/ZhangZhiyang/Desktop/data_sets/sign-language-mnist/sign_mnist_test.csv')
# csv = csv.reader(tt,dialect = 'excel')
# sign_mnist_test_data = np.zeros((7172,1,28,28))
# sign_mnist_test_label = np.zeros(7172)
# next(csv)
# i = 0
# for row in csv:
#     sign_mnist_test_label[i] = row[0]
#     image = row[1:785]
#     for pixel in image:
#         pixel = int(pixel)
#     sign_mnist_test_data[i,0] = np.array(image).reshape(28,28)
#     i += 1
# plt.imshow(sign_mnist_test_data[5,0])
# plt.show()


# a = np.array([0,1,2])
# # b = np.linalg.norm(a)
# # print(b)
# print(np.exp(a))

# file = open('/Volumes/Data/ZhangZhiyang/Desktop/params.pkl','rb')
# net_params = pickle.load(file)
# file.close()
# count = 0
# net_params = {}
# net_params['W1'] = 0.01 * np.random.randn(30, 1, 5, 5)
# for i in range(30):
#     print(i,'-----------------------')
#     for j in range(i+1,30):
#         cov1 = net_params['W1'][i,0].reshape((1,-1))
#         cov2 = net_params['W1'][j,0].reshape((-1,1))
#
#         print(np.linalg.norm(cov1 - cov2, ord=2))

        # if abs(np.dot(cov1,cov2)/(l1*l2)) < 0.5:
        #     count += 1
        # count += abs(np.dot(cov1,cov2)/(l1*l2))
        # print(np.dot(cov1,cov2)/(l1*l2))
# print(count/(16*15/2))
# print(count)

a = np.arange(0,100).reshape((10,10))
print(a[0:4:2])