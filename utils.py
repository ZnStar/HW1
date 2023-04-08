import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(train = True,valid_size = 0.1):
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    if train == True:
        X = np.load('X_train.npy')
        y = np.load('y_train.npy')
        X, X_valid, y, y_valid = train_test_split(X, y, test_size=valid_size, random_state=0)
        return X, X_test, X_valid, y,y_test, y_valid
    if train == False:
        return X_test, y_test

def load_model(random_initial=True, nHidden=None, path=None,input_dim = 784,output_dim = 10):
    """
    随机化初始模型或者读入模型
    :param nHidden: 训练模型的隐藏层大小
    :param random_intial: 随机化参数（用于训练）
    :param path: 模型参数位置
    :return:
    """
    if random_initial == True:
        if nHidden == None:
            print("没有输入模型结构参数！")
        # Count number of parameters and initialize weights 'w'
        nParams = input_dim * nHidden[0]
        bParams = nHidden[0]
        for h in range(1, nHidden.shape[0]):
            nParams += nHidden[h - 1] * nHidden[h]
            bParams += nHidden[h]

        # last layer
        nParams += nHidden[-1] * output_dim
        bParams += output_dim

        # initialize the weight
        # data = scio.loadmat('w.mat')
        # w = data['w']
        w = np.random.randn(nParams, 1)
        b = np.random.randn(bParams, 1)

        return w, b
    if random_initial == False:
        if path == None:
            print("没有输入模型路径，请重试！")
        para = scio.loadmat(path)
        w = para['w']
        b = para['b']
        mu = para['mu']
        sigma = para['sigma']
        nHidden = para['nHidden'].reshape([1])
        nLabels = int(para['nLabels'])
        return w, b, mu, sigma, nHidden, nLabels

def standardizeCols(M, mu=None, sigma2=None):
    """
    对每一列进行归一化处理
    :param M: 输入数据矩阵
    :param mu: mean， 如果为None，根据矩阵计算
    :param sigma2: std， 如果为None，根据矩阵计算
    :return: [S,mu,sigma2]
    """
    (nrows, ncols) = M.shape
    M.astype('float')
    if mu is None:
        mu = np.average(M, axis=0)
    if sigma2 is None:
        sigma2 = np.std(M, axis=0, ddof=1)
        sigma2 = (sigma2 < 2e-16) * 1 + (sigma2 >= 2e-16) * sigma2

    S = M - np.tile(mu, (nrows, 1))
    if ncols > 0:
        S = S / np.tile(sigma2, (nrows, 1))

    return S,mu,sigma2

def MLPL2Loss(w, lamb):
    ret = lamb * w
    return ret

if __name__ == "__main__":
    # 可视化参数
    w, b, mu, sigma, nHidden, nLabels = load_model(path='best_model.mat', random_initial=False)
    nInstances, nVars = 54000, 784

    inputWeights = w[0:(nVars * nHidden[0])].reshape(nVars, nHidden[0], order='F')
    offset = nVars * nHidden[0]
    inputBias = b[0:nHidden[0]].reshape(1, nHidden[0], order='F')

    b_offset = nHidden[0]
    outputWeights = w[offset:offset + nHidden[-1] * nLabels]
    outputWeights = outputWeights.reshape(nHidden[-1], nLabels, order='F')
    outputBias = b[b_offset:b_offset + nLabels].reshape(1, nLabels, order='F')

    # 热力图
    plt.figure(figsize=(10,10))
    plt.imshow(inputWeights, cmap='Reds', interpolation='nearest')
    plt.title("first layer")
    plt.axis([0,300,0,784])
    # plt.axis('off')
    plt.savefig("reds_first.png")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(outputWeights, cmap='Reds', interpolation='nearest')
    plt.title("second layer")
    plt.axis([0,10,0,300])
    plt.savefig("reds_second.png")
    plt.show()

    # 直方图
    layer1_weights = inputWeights.flatten().tolist()
    plt.hist(layer1_weights, bins=100)
    plt.title("layer1 weights")
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.savefig("hist_layer1_weights.png")
    plt.close()

    layer1_bias = inputBias.flatten().tolist()
    plt.hist(layer1_bias, bins=100)
    plt.title("layer1 bias")
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.savefig("hist_layer1_bias.png")
    plt.close()

    layer2_weights = outputWeights.flatten().tolist()
    plt.hist(layer2_weights, bins=100)
    plt.title("layer2 weights")
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.savefig("hist_layer2_weights.png")
    plt.close()

    layer2_bias = outputBias.flatten().tolist()
    plt.hist(layer2_bias, bins=100)
    plt.title("layer2 bias")
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.savefig("hist_layer2_bias.png")
    plt.close()
