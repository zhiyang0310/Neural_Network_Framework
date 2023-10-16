import numpy as np
import Functions

class Sigmoid:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self,x):
        self.x = x
        self.y = Functions.Sigmoid(x)
        return self.y
    def backward(self,dout):
        dx = dout*self.y*(1-self.y)
        return dx

class ReLu:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self,x):
        self.x = x
        self.y = Functions.ReLu(x)
        return self.y
    def backward(self,dout):
        mask = self.x > 0
        dx = np.zeros(self.x.shape)[mask] = 1
        dx = dout*dx
        return dx

class Affine:
    def __init__(self,w,b):
        self.w = w
        self.b = b
    def forward(self,x):
        self.x = x
        y = np.dot(self.x,self.w)+self.b
        return y
    def backward(self,dout):
        self.dw = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0)
        dx = np.dot(dout,self.w.T)
        return dx

class Convolution:
    def __init__(self,w,b,stride = 1,pad = 0):
        self.w = w
        self.b = b
        self.stride = stride
        self.pad = pad
        self.filter_num,self.filter_channel,self.filter_height,self.filter_width = w.shape
        self.col_data = None
    def forward(self,data):
        self.data_num, data_channel, self.data_height, self.data_width = data.shape
        data = np.pad(data, [(0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)], 'constant')
        data_height = self.data_height + 2*self.pad
        data_width = self.data_width + 2*self.pad
        self.out_height = int((data_height - self.filter_height) / self.stride) + 1
        self.out_width = int((data_width - self.filter_width) / self.stride) + 1
        self.col_data = Functions.im2col(data,self.filter_height,self.filter_width,self.stride,self.pad)
        col_w = self.w.reshape(self.filter_num,-1).T
        out = (np.dot(self.col_data,col_w)+self.b).reshape((-1,self.out_height,self.out_width,self.filter_num)).transpose((0,3,1,2))
        return out
    def backward(self,dout):
        dout = dout.transpose((1,0,2,3))
        col_dout = dout.reshape((self.filter_num,self.out_height*self.out_width*self.data_num)).T
        dw = np.dot(self.col_data.T,col_dout)
        self.dw = dw.reshape((self.filter_channel,self.filter_height,self.filter_width,self.filter_num)).transpose((3,0,1,2))
        self.db = np.sum(col_dout,axis=0)
        dx = np.dot(col_dout,self.w.reshape(self.filter_num,-1))
        dx = Functions.col2im(dx,self.data_num,self.data_height,self.data_width,self.filter_height,self.filter_width,self.stride,self.pad)
        return dx

class Pooling():
    def __init__(self,filter_height,filter_width,stride = 1,pad = 0):
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.stride = stride
        self.pad = pad
        self.arg = None
    def forward(self,data):
        self.data_num,self.data_channel,self.data_height,self.data_width = data.shape
        self.out_height = int((self.data_height+ 2*self.pad- self.filter_height) / self.stride) + 1
        self.out_width = int((self.data_width+2*self.pad - self.filter_width) / self.stride) + 1

        new_data = np.zeros((self.data_num,self.data_channel,self.out_height,self.out_width,self.filter_height,self.filter_width))

        for i in range(self.out_height):
            for j in range(self.out_width):
                new_data[:,:,i,j,:,:] = data[:,:,i*self.stride:i*self.stride+self.filter_height,j*self.stride:j*self.stride+self.filter_width]

        new_data = new_data.reshape((-1,self.filter_height*self.filter_width))
        self.arg = np.argmax(new_data,axis=1)
        new_data = np.max(new_data,axis=1)

        out = new_data.reshape((self.data_num,self.data_channel,self.out_height,self.out_width))
        return out
    def backward(self,dout):
        dout = dout.reshape((-1,1))
        row = dout.shape[0]
        new_data = np.zeros((row,self.filter_height*self.filter_width))
        row = np.arange(row)
        new_data[row,self.arg] = dout.reshape((1,-1))
        fold_filter_data = new_data.reshape((self.data_num,self.data_channel,self.out_height,self.out_width,self.filter_height,self.filter_width))
        folder_data = np.zeros((self.data_num,self.data_channel,self.data_height+2*self.pad,self.data_width+2*self.pad))
        for i in range(self.out_height):
            for j in range(self.out_width):
                folder_data[:,:,i*self.stride:i*self.stride+self.filter_height,j*self.stride:j*self.stride+self.filter_width] = \
                fold_filter_data[:,:,i,j,:,:]

        out = folder_data[:,:,self.pad:self.pad+self.data_height,self.pad:self.pad+self.data_width]
        return out

class Fullconnection:
    def forward(self,data):
        self.data_num, self.data_channel, self.data_height, self.data_width = data.shape
        y = data.reshape((self.data_num,-1))
        return y
    def backward(self,dout):
        dx = dout.reshape((self.data_num, self.data_channel, self.data_height, self.data_width))
        return dx

class Dropout:
    def __init__(self,dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self,data,train_flag = True):
        if train_flag:
            self.mask = np.random.rand(data.shape) > self.dropout_rate

            return data*self.mask

        else:
            return data*(1 - self.dropout_rate)

    def backward(self,dout):
        dx = dout*self.mask

        return dx

class Softmaxwithloss:
    def forward(self,x,label):
        self.batch = x.shape[0]
        self.label = label
        self.y = Functions.Softmax(x)
        return Functions.CrossEntropy(self.y,self.label)
    def backward(self):
        dx = (1/self.batch)*(self.y-self.label)
        return dx

class CapsuleForCorrelation:
    def __init__(self,cap_dim):
        self.cap_dim = cap_dim

    def forward(self,x):
        x = x.transpose((0,2,3,1))
        y = x.reshape((x.shape[0],-1,self.cap_dim))
        self.data_num,self.data_height,self.data_width,self.data_channel = x.shape

        # print(y[0,0])
        # print(y[0, 1])
        # print(y[0, 2])
        # print(y[0, 8])
        # print(y[0, 9])
        # print(y[0, 10])

        # for num in range(self.data_num):
        #     for row in range(y.shape[1]):
        #         count = 0
        #         for col in range(y.shape[2]):
        #             if y[num,row,col] != 0:
        #                 count += 1
        #         if count > 1:
        #             print(y[num,row])
        return y

    def backward(self,dout):
        dx = dout.reshape((self.data_num,self.data_height,self.data_width,self.data_channel)).transpose((0,3,1,2))
        return dx

class CapsuleForCombine:
    def __init__(self,feature_num,feature_capsules,cap_dim):
        self.feature_num = feature_num
        self.feature_capsules = feature_capsules
        self.cap_dim = cap_dim

    def forward(self,x):
        data_num ,channel,height,width= x.shape
        x = x.transpose((0, 2, 3, 1))
        y = x.reshape((data_num,self.feature_capsules,self.feature_num,self.cap_dim)).transpose((0,2,1,3))

        self.data_num = data_num
        self.channel = channel
        self.height = height
        self.width = width
        return y

    def backward(self,dout):
        dout = dout.transpose((0,2,1,3))
        dx = dout.reshape((self.data_num,self.height,self.width,self.channel)).transpose((0,3,1,2))

        return dx

class UnitForCorrelation:
    def forward(self,x):
        self.data = np.zeros_like(x)
        self.data[:] = x[:]


        length = np.linalg.norm(x,ord=2,axis=2)
        for i in range(x.shape[0]):
            x[i] = (x[i].T/(length[i]+np.e**-50)).T

        self.length = length
        self.cap_dim = x.shape[2]

        return x

    def backward(self,dout):
        dx = np.zeros_like(self.data)
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                y = -np.dot(self.data[i,j].reshape((-1,1)),self.data[i,j].reshape((1,-1)))/(self.length[i,j]**3+np.e**-50)+\
                    (1/(self.length[i,j]+np.e**-50))*np.eye(self.cap_dim)
                dx[i,j] = np.dot(dout[i,j].reshape((1,-1)),y)
        return dx

class UnitForCombine:
    def forward(self,x):
        y = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                length = np.linalg.norm(x[i,j],ord=2,axis=1)
                y[i,j] = ((x[i,j].T)/(length+np.e**-50)).T

        return y


class Correlation:
    def __init__(self,group_size):
        self.group_size = group_size
    def forward(self,x):

        data_num,cap_num,cap_dim = x.shape
        y = np.zeros((data_num,int(cap_num/self.group_size),self.group_size,self.group_size))
        for num in range(data_num):
            for i in range(int(cap_num/self.group_size)):
                y[num,i] = np.dot(x[num,self.group_size*i:self.group_size*(i+1)],x[num,self.group_size*i:self.group_size*(i+1)].T)

        self.data = x
        self.data_num = x.shape[0]

        return y


        # y = np.zeros((x.shape[0],x.shape[1],x.shape[1]))
        # for i in range(x.shape[0]):
        #     y[i] = np.dot(x[i],x[i].T)
        #
        # self.data = x
        # self.data_num = x.shape[0]
        #
        # return y

    def backward(self,dout):
        dx = np.zeros_like(self.data)
        for num in range(self.data_num):
            for i in range(self.group_size):
                dx[num,self.group_size*i:self.group_size*(i+1)] = 2*np.dot(dout[num,i],self.data[num,self.group_size*i:self.group_size*(i+1)])

        return dx

        # dx = np.zeros_like(self.data)
        # for i in range(self.data_num):
        #     dx[i] = 2*np.dot(dout[i],self.data[i])
        #
        # return dx

class LosswithNorm2:
    def forward(self,x):
        # print(x)
        # print(x.shape)

        self.data = x
        self.data_num = x.shape[0]
        self.group_size = x.shape[2]

        return Functions.LossNorm2(x)

    def backward(self):
        dx = 2*self.data
        for num in range(self.data_num):
            for i in range(self.data.shape[1]):
                dx[num,i] -= np.eye(self.group_size)
        # diag = np.eye(self.data.shape[1])
        # for i in range(self.data_num):
        #     self.data[i] -= diag

        return dx/self.data_num

class LosswithCorrelation:
    def forward(self,x):
        print(x)

        self.data = x
        self.data_num = x.shape[0]

        return Functions.LossCorrelation(x)

    def backward(self):
        dx = np.zeros_like(self.data)
        dx[:] = 2*self.data[:]
        for num in range(self.data_num):
            sum = (np.linalg.norm(self.data[num])) ** 2
            diag_sum = (np.linalg.norm(np.diag(self.data[num]), ord=2)) ** 2
            dx[num] = dx[num]/(diag_sum+np.e**-50)
            alpha = (diag_sum-sum)/(diag_sum+np.e**-50)
            for i in range(dx.shape[1]):
                dx[num,i,i] = alpha*dx[num,i,i]

        return dx/self.data_num

class Combine:
    def __init__(self,b):
        self.b = b

    def forward(self,x):
        softmax = Functions.Softmax(self.b)
        data_num,feature_num,feature_capsule,capsule_dim = x.shape
        y = np.zeros((data_num,feature_num,capsule_dim))

        for num in range(data_num):
            for i in range(feature_num):
                y[num,i] = np.sum((softmax[i]*x[num,i].T).T,axis=0)

        self.softmax = softmax
        self.data = x
        self.data_num = data_num
        self.feature_num = feature_num

        return  y

    def backward(self,dout):
        dx = np.zeros_like(self.data)
        db = np.zeros_like(self.b)

        for num in range(self.data_num):
            for i in range(self.feature_num):
                dx[num,i] = np.dot(self.softmax[i].reshape(-1,1),dout[num,i].reshape(1,-1))

        dsoftmax = np.zeros_like(db)
        for i in range(dsoftmax.shape[0]):
            for num in range(self.data_num):
                dsoftmax[i] += np.dot(dout[num,i],self.data[num,i].T)

        for i in range(db.shape[0]):
            e = np.exp(self.b[i])
            sum = np.sum(e)
            M = (-np.dot(e.reshape(1,-1).T,e.reshape(1,-1)))/(sum**2)+np.diag(e)/sum
            db[i] = np.dot(dsoftmax[i].reshape(1,-1),M)

        self.db = db

        return dx

class Liner:
    def forward(self,x):
        self.data_num, self.feature_num, self.capsule_dim = x.shape
        y = np.zeros((self.data_num,self.feature_num*self.capsule_dim))
        y = x.reshape((self.data_num,self.feature_num*self.capsule_dim))


        return y

    def backward(self,dout):

        return dout.reshape((self.data_num, self.feature_num, self.capsule_dim))



class AffineGroup:
    def __init__(self,W):
        self.W = W      #Wij
    def forward(self,x):
        class_num = self.W.shape[1]
        cap_dim_out = self.W.shape[3]
        data_num,cap_num,cap_dim = x.shape

        y = np.zeros((data_num,cap_num,class_num,cap_dim_out))
        for i in range(data_num):
            for j in range(cap_num):
                for k in range(class_num):
                    y[i,j,k] = np.dot(x[i,j],self.W[j,k])

        self.data_num, self.cap_num, self.cap_dim = data_num,cap_num,cap_dim
        self.class_num = class_num
        self.data = x

        return y
    def backward(self,dout):
        dx = np.zeros((self.data_num, self.cap_num, self.cap_dim))
        self.dW = np.zeros_like(self.W)

        for i in range(self.data_num):
            for j in range(self.cap_num):
                for k in range(self.class_num):
                    dx[i,j] += np.sum(dout[i,j,k]*self.W[j,k], axis=1).T

        for i in range(self.cap_num):
            for j in range(self.class_num):
                for num in range(self.data_num):
                    self.dW[i,j] += np.dot(dout[num,i,j].reshape((-1,1)),self.data[num,i].reshape((1,-1))).T

        return dx

class Weighting:
    def __init__(self,w):
        self.w = w
    def forward(self,x):
        data_num, cap_num, class_num, cap_dim_out = x.shape
        y = np.zeros((data_num,class_num,cap_dim_out))
        for i in range(data_num):
            for j in range(class_num):
                y[i, j] = np.sum((self.w[:, j].T * x[i, :, j, :].T).T, axis=0)

        self.length = np.linalg.norm(y, ord=2, axis=2)
        for i in range(data_num):
            y[i] = (y[i].T * (self.length[i] / (1 + self.length[i] ** 2))).T

        probability = np.linalg.norm(y, ord=2, axis=2)

        self.data_num = data_num
        self.cap_num = cap_num
        self.class_num = class_num
        self.cap_dim_out = cap_dim_out
        self.y = y
        self.data = x
        # self.probability = probability

        return  probability

    def backward(self,dout):
        dy = np.zeros((self.data_num, self.class_num, self.cap_dim_out))
        for i in range(self.data_num):
            dy[i] = (((2 * self.y[i]).T / (((1 + self.length[i] ** 2)) ** 2)) * dout[i]).T
        # for i in range(self.data_num):
        #     dy[i] = dy[i] = ((( self.y[i]).T / (self.probability[i] )) * dout[i]).T
        dx = np.zeros((self.data_num, self.cap_num, self.class_num, self.cap_dim_out))
        for i in range(self.data_num):
            for j in range(self.cap_num):
                dx[i, j] = ((self.w[j]) * (dy[i].T)).T

        dw = np.zeros_like(self.w)
        for i in range(self.cap_num):
            for j in range(self.class_num):
                for num in range(self.data_num):
                    dw[i,j] += np.dot(self.data[num,i,j],dy[num,j])

        self.dw = dw


        return dx

class DynamicRouting:
    def __init__(self,b):
        self.b = b
    def forward(self,x):
        data_num,cap_num,class_num,cap_dim_out = x.shape
        y = np.zeros((data_num,class_num,cap_dim_out))

        for r in range(3):
            softmax = Functions.Softmax(self.b)
            for i in range(data_num):
                for j in range(class_num):
                    y[i,j] = np.sum((softmax[:,j].T * x[i,:,j,:].T).T,axis = 0)
            self.y = np.zeros_like(y)
            self.y[:] = y[:]
            self.softmax = np.zeros_like(softmax)
            self.softmax[:] = softmax[:]

            self.length = np.linalg.norm(y,ord = 2,axis = 2)
            for i in range(data_num):
                y[i] = (y[i].T*(self.length[i]/(1+self.length[i]**2))).T

            for i in range(cap_num):
                for j in range(class_num):
                    for k in range(data_num):
                        self.b[i,j] += np.dot(y[k,j],x[k,i,j].T)

        probability = np.linalg.norm(y, ord=2, axis=2)

        self.data_num = data_num
        self.cap_num = cap_num
        self.class_num = class_num
        self.cap_dim_out = cap_dim_out

        return probability

    def backward(self,dout):
        dy = np.zeros((self.data_num,self.class_num,self.cap_dim_out))
        for i in range(self.data_num):
            dy[i] = (((2*self.y[i]).T/(((1+self.length[i]**2))**2)) * dout[i]).T
        dx = np.zeros((self.data_num,self.cap_num,self.class_num,self.cap_dim_out))
        for i in range(self.data_num):
            for j in range(self.cap_num):
                dx[i,j] = ((self.softmax[j])*(dy[i].T)).T

        return dx



class LossCapsuleMargin:
    def forward(self,x,label):# label is not one-hot type
        Loss = 0
        data_num = x.shape[0]
        class_num = x.shape[1]
        flag = np.zeros_like(x)
        for i in range(data_num):
            for j in range(class_num):
                if label[i] == j:
                    Loss += (max(0, 0.9 - x[i, j])) ** 2
                    flag[i,j] = np.argmax([0, 0.9 - x[i, j]])
                else:
                    Loss += 0.5 * (max(0, x[i, j] - 0.1)) ** 2
                    flag[i, j] = np.argmax([0, x[i, j] - 0.1])

        self.x = x
        self.flag = flag
        self.label = label

        return Loss/data_num

    def backward(self):
        dx = np.zeros_like(self.x)
        data_num = self.x.shape[0]
        class_num = self.x.shape[1]
        for i in range(data_num):
            for j in range(class_num):
                if self.label[i] == j:
                    if self.flag[i,j] == 0:
                        dx[i,j] = 0
                    else:
                        dx[i,j] = -2*(0.9-self.x[i,j])
                else:
                    if self.flag[i,j] == 0:
                        dx[i,j] = 0
                    else:
                        dx[i,j] = self.x[i,j]-0.1

        return dx/data_num

class LossEntropy:
    def forward(self,x,label):
        Loss = 0
        for data_num in range(x.shape[0]):
            Loss -= Functions.Log(x[data_num,int(label[data_num])])

        self.data = x
        self.label = label

        return Loss/x.shape[0]

    def backward(self):
        dx = np.zeros_like(self.data)
        for data_num in range(dx.shape[0]):
            dx[data_num,int(self.label[data_num])] = -1/(self.data[data_num,int(self.label[data_num])]+np.e**-50)

        return dx/dx.shape[0]