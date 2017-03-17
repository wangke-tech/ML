#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

def computeCost(X, y, theta=[[0], [0]]):
    m = y.size
    h =X.dot(theta)
    J = 1.0 / (2 * m) * (np.sum(np.square(h - y)))

    return J


def gradientDecent(X, y, theta=[[0], [0]], alpha=0.01, num_iters=1500):
    m = y.size
    J_History = np.zeros(num_iters)

    for iter in xrange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha * 1.0 / m * X.T.dot(h -y)
        J_History[iter] = computeCost(X, y, theta)

    return theta, J_History

def main():

    # 动态开关
    DYNC_plt_points, DYNC_plt_GD =  0 , 0

    data = np.loadtxt('linear_regression_data1.txt', delimiter=',')
    X = np.c_[np.ones(data.shape[0]), data[:, 0]]
    y = np.c_[data[:, 1]]

    if DYNC_plt_points:
        # 画出点
        print computeCost(X, y)
        plt.scatter(X[:, 1], y, s=30, c='r', marker='x', linewidth=1)
        plt.xlim(4, 24)
        plt.xlabel('Population of City in 10,000 s')
        plt.ylabel('Profit in $10,000 s')

        plt.show()

    theta, Cost_J = gradientDecent(X, y)
    if DYNC_plt_GD:
        # 画出每一次迭代和损失函数的变化
        print 'theta: ', theta.ravel()
        plt.plot(Cost_J)
        plt.ylabel('Cost J')
        plt.xlabel('Iterations')
        plt.show()

    xx = np.arange(5, 22)
    yy = theta[0] + theta[1] * xx

    # 画出我们自己写的线性回归梯度下降的收敛情况
    plt.scatter(X[:, 1], y, s=30, c='r', marker='x', linewidth=1)
    plt.plot(xx, yy, label='Linear Regression(Gradient Decent)')

    # 和sci-learn中的线性回归对比一下
    regr = LinearRegression()
    regr.fit(X[:, 1].reshape(-1, 1), y.ravel())
    plt.plot(xx, regr.intercept_ + regr.coef_ * xx, label="Linear Regression(Sci-Learn GLM)" )
    plt.xlim(4, 24)
    plt.xlabel('Population of City in 10,000 s')
    plt.ylabel('Profit in $10,000 s')
    plt.legend(loc=4)
    plt.show()

    print theta.T
    print theta.T.dot([1, 3.5]) * 100000

if '__main__' == __name__:
    main()
