import numpy as np

def ReLu(x):
    return np.maximum(0,x)

def Sigmoid(x):
    return 1/(1+np.exp(-x))

def Softmax(x):
    batch = x.shape[0]
    z = np.zeros(x.shape)
    for i in range(batch):
        y = x[i,:]
        max = np.max(y)
        y = y - max
        y = np.e**y
        s = np.sum(y)
        y = y/s
        z[i,:] = y
    return z

def Log(x):
    delta = np.e**-50
    y = np.log(x + delta)
    return y

def CrossEntropy(y,label):
    if(y.shape[0] != label.shape[0]):
        print('error: cross entropy error!')
        exit(0)
    x = -np.sum(Log(y)*label)/y.shape[0]
    return x

def im2col(data,filter_height,filter_width,stride = 1,pad = 0):
    data = np.pad(data,[(0,0),(0,0),(pad,pad),(pad,pad)],'constant')
    data_num, data_channel, data_height, data_width = data.shape

    # if (data_width-filter_width) % stride != 0 or (data_height-filter_height)%stride != 0:
    #     print('filter_height or filter_width error in im2col!')

    out_height = int((data_height + 2*pad - filter_height)/stride) + 1
    out_width = int((data_width + 2*pad - filter_width)/stride) + 1

    new_data = np.zeros((data_num,data_channel,filter_height,filter_width,out_height,out_width))

    for i in range(out_height):
        for j in range(out_width):
            new_data[:,:,:,:,i,j] = data[:,:,i*stride:i*stride+filter_height,j*stride:j*stride+filter_width]
    new_data = new_data.transpose((0,4,5,1,2,3))
    # col = np.zeros((int(data_num * out_height * out_width),int(data_channel*filter_height*filter_width)))
    # for i in range(int(data_num * out_height * out_width)):
    #     n = int(i/(out_height * out_width))# data_num
    #     m = i%(out_height * out_width)# move_num in one data
    #     k = int(m/out_width)# move in height
    #     j = m % out_width# move in width
    #     col[i,:] = data[n,:,int(stride*k):int(stride*k+filter_height),int(stride*j):int(stride*j+filter_width)].reshape((1,-1))
    col = new_data.reshape((data_num*out_height*out_width,data_channel*filter_height*filter_width))
    return col

def col2im(col_data,data_num,data_height,data_width,filter_height,filter_width,stride = 1,pad = 0):
    out_height = int((data_height + 2*pad - filter_height) / stride) + 1
    out_width = int((data_width + 2*pad - filter_width) / stride) + 1

    filter_channel = int(col_data.shape[1] / (filter_height * filter_width))

    col_data = col_data.reshape((data_num,out_height,out_width,filter_channel,filter_height,filter_width)).transpose(0,3,1,4,2,5)
    col_data = col_data.reshape((data_num,filter_channel,out_height*filter_height,out_width*filter_width))

    data = np.zeros((data_num, filter_channel, data_height + 2 * pad, data_width + 2 * pad))

    for i in range(out_height):
        for j in range(out_width):
            data[:,:,i*stride:i*stride+filter_height,j*stride:j*stride+filter_width] += \
                col_data[:,:,i*filter_height:(i+1)*filter_height,j*filter_width:(j+1)*filter_width]
    # row = 0
    # while(row < col_data.shape[0]):
    #     for i in range(data_num):
    #         for j in range(out_height):
    #             for k in range(out_width):
    #                 component = col_data[row,:].reshape((filter_dim,filter_height,filter_width))
    #                 data[i,:,stride*j:stride*j+filter_height,stride*k:stride*k+filter_width] += component
    #                 row += 1
    data = data[:,:,pad:data_height+pad,pad:data_width+pad]
    return data

# def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
#     """
#
#     Parameters
#     ----------
#     input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
#     filter_h : 滤波器的高
#     filter_w : 滤波器的长
#     stride : 步幅
#     pad : 填充
#
#     Returns
#     -------
#     col : 2维数组
#     """
#     N, C, H, W = input_data.shape
#     out_h = (H + 2*pad - filter_h)//stride + 1
#     out_w = (W + 2*pad - filter_w)//stride + 1
#
#     img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
#     col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
#
#     for y in range(filter_h):
#         y_max = y + stride*out_h
#         for x in range(filter_w):
#             x_max = x + stride*out_w
#             col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
#
#     col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
#     return col


# def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
#     """
#
#     Parameters
#     ----------
#     col :
#     input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
#     filter_h :
#     filter_w
#     stride
#     pad
#
#     Returns
#     -------
#
#     """
#     N, C, H, W = input_shape
#     out_h = (H + 2*pad - filter_h)//stride + 1
#     out_w = (W + 2*pad - filter_w)//stride + 1
#     col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
#
#     img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
#     for y in range(filter_h):
#         y_max = y + stride*out_h
#         for x in range(filter_w):
#             x_max = x + stride*out_w
#             img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
#
#     return img[:, :, pad:H + pad, pad:W + pad]

def CapsuleMargin(probability,label):
    Loss = 0
    data_num = probability.shape[0]
    class_num = probability.shape[1]
    for i in range(data_num):
        for j in range(class_num):
            if label[i] == j:
                Loss += (max(0,0.9-probability[i,j]))**2
            else:
                Loss += 0.5*(max(0,probability[i,j]-0.1))**2

    return Loss/data_num

def LossNorm2(x):
    data_num = x.shape[0]
    Loss = 0
    for i in range(data_num):
        Loss += np.linalg.norm(x[i])

    return Loss/data_num

def LossCorrelation(x):
    data_num = x.shape[0]
    Loss = 0
    for i in range(data_num):
        sum = (np.linalg.norm(x[i]))**2
        diag_sum = (np.linalg.norm(np.diag(x[i]),ord=2))**2
        Loss += (sum-diag_sum)/(diag_sum+np.e**-50)

    return Loss/data_num