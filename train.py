import argparse
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from model import *


def train(learning_rate=0.001, layer_size=300, regular=0.01, maxIter=700000, curve=True, lr_decay = 0):
    """
    训练
    :param learning_rate:
    :param layer_size:
    :param regular:
    :param maxIter:
    :return:
    """
    X, X_test, X_valid, y, y_test, y_valid = load_data(train=True)
    (n, d) = X.shape  # 54000*784
    nLabels = 10  # 标签共10类
    t = X_valid.shape[0]  # 6000
    t2 = X_test.shape[0]  # 10000
    # Standardize columns and add bias
    X, mu, sigma = standardizeCols(X)
    # Make sure to apply the same transformation to the validation/test data
    X_valid, _, _ = standardizeCols(X_valid, mu, sigma)
    X_test, _, _ = standardizeCols(X_test, mu, sigma)
    # network structure
    nHidden = np.array([layer_size])

    # 参数初始化
    w, b = load_model(random_initial=True, nHidden=nHidden, path=None, input_dim=d, output_dim=nLabels)

    Mw = np.zeros(w.shape)
    Gw = np.zeros(w.shape)
    Mb = np.zeros(b.shape)
    Gb = np.zeros(b.shape)
    # Train with stochastic gradient
    initial_lr = learning_rate
    beta1 = 0.9
    beta2 = 0.99
    lamb = regular
    minibatch = 8
    eps = 1e-9

    # early stopping
    max_not_decay = 20
    not_decay = 0
    min_error = 1
    if curve:
        train_error = []
        train_loss = []
        valid_error = []
        valid_loss = []
        times = []

    print('Training with learning_rate = {a}, layer_size = {b}, regular = {c}'.
          format(a=learning_rate, b=layer_size, c=regular))

    # 迭代训练
    for iter in range(maxIter):
        if iter % 5000 == 0:
            # validation error
            prob = MLPclassificationPredict(w, b, X_valid, nHidden, nLabels)
            yhat = prob.argmax(axis=1).reshape(prob.shape[0], 1)
            error = float(sum(yhat != y_valid) / t)
            print('Training iteration = {iter}, validation error = {error}'.format(
                iter=iter, error=error))
            # 保存error，loss
            if curve:
                times.append(iter)
                # valid：error,loss保存
                valid_error.append(error)
                loss = 0
                for i in range(t):
                    loss += - np.log(prob[i][y_valid[i]])
                valid_loss.append(loss / t)

                # train：error,loss保存
                prob = MLPclassificationPredict(w, b, X, nHidden, nLabels)
                yhat = prob.argmax(axis=1).reshape(prob.shape[0], 1)
                train_error.append(float(sum(yhat != y) / n))
                loss = 0
                for i in range(n):
                    loss += - np.log(prob[i][y[i]])
                train_loss.append(loss / n)

            # 记录最优模型
            if error < min_error:
                min_error = error
                best_b = b.copy()
                best_w = w.copy()
                not_decay = 0
            else:
                not_decay += 1
                if not_decay >= max_not_decay:  # early stopping
                    break
        # 小批量梯度下降，L2正则化
        i = np.floor(np.random.rand(minibatch) * n).astype(int)
        gw, gb = MLPclassificationLoss(w, b, X[i, :], y[i], nHidden, nLabels)
        gw += MLPL2Loss(w, lamb)

        iter += 1
        stepSize = initial_lr / (1 + lr_decay * iter)

        Mw = beta1 * Mw + (1 - beta1) * gw
        Gw = beta2 * Gw + (1 - beta2) * gw * gw
        w = w - stepSize / np.sqrt(Gw / (1 - beta2 ** iter) + eps) * Mw / (1 - beta1 ** iter)

        Mb = beta1 * Mb + (1 - beta1) * gb
        Gb = beta2 * Gb + (1 - beta2) * gb * gb
        b = b - stepSize / np.sqrt(Gb / (1 - beta2 ** iter) + eps) * Mb / (1 - beta1 ** iter)
    # 输出最后的结果并保存最佳模型
    scio.savemat('para.mat', mdict={'w': best_w, 'b': best_b, 'nHidden': nHidden,
                                    'nLabels': nLabels, 'mu': mu, 'sigma': sigma})

    prob = MLPclassificationPredict(best_w, best_b, X, nHidden, nLabels)
    yhat = prob.argmax(axis=1).reshape(prob.shape[0], 1)
    print('Train error with final model = {error}'.format(
        error=float(sum(yhat != y) / n)))

    prob = MLPclassificationPredict(best_w, best_b, X_valid, nHidden, nLabels)
    yhat = prob.argmax(axis=1).reshape(prob.shape[0], 1)
    va_error = float(sum(yhat != y_valid) / t)
    print('Validation error with final model = {error}'.format(error=error))

    # Evaluate test error
    prob = MLPclassificationPredict(best_w, best_b, X_test, nHidden, nLabels)
    yhat = prob.argmax(axis=1).reshape(prob.shape[0], 1)
    print('Test error with final model = {error}\n'.format(
        error=float(sum(yhat != y_test) / t2)))

    if curve:
        return train_error, train_loss, valid_error, valid_loss, times

    return best_w, best_b, va_error, nHidden, nLabels, mu, sigma


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--iter', type=int, default=300000,
                        help='iteration times')
    parser.add_argument('--layer', type=int, default=300,
                        help='size for the hidden layer')
    parser.add_argument('--regular', type=float, default=0.01,
                        help='coefficient of L2 regularization')
    parser.add_argument('--lr_decay', type=float, default=0,
                        help='coefficient of learning rate decay')
    args = parser.parse_args()

    train_error, train_loss, valid_error, valid_loss, times = train(args.lr, args.layer, args.regular, args.iter,lr_decay=args.lr_decay)

    plt.plot(times, train_error, label='Error on the training set')
    plt.plot(times, valid_error, label='Error on the validation set')
    plt.xlabel('Iteration')
    plt.ylabel('log Error')
    plt.legend()
    plt.savefig('Error.jpg')

    plt.figure()
    plt.plot(times, train_loss, label='Loss on the training set')
    plt.plot(times, valid_loss, label='Loss on the validation set')
    plt.xlabel('Iteration')
    plt.ylabel('log Loss')
    plt.legend()
    plt.savefig('Loss.jpg')
