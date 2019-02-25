#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==============================================================================
# brief        誤差楕円
#
# author       Takuya Niibori
# attention    none
#==============================================================================

import numpy as np
from scipy import interpolate as sp
from matplotlib import patches
import matplotlib.pyplot as plt

class ErrorEllipse(object):
    '''誤差楕円'''

    def __init__(self, p):
        '''コンストラクタ
        引数：
            p：上側累積パーセント点(2自由)
        '''
        # 上側累積パーセント点マップ[%]
        self.p = np.array([99.9, 99.5, 99, 98.5, 98, 97.5, 97, 96, 95, 94, 93, 92, 91, 90, 85, 80, 75, 70, 65, 60, 55,
                           50, 45, 40, 35, 30, 25, 20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2.5, 2, 1.5, 1, 0.5, 0])
        # カイの二乗値マップ
        self.square_x = np.array([13.81551056, 10.59663473, 9.210340372, 8.399410156, 7.824046011, 7.377758908,
                                  7.013115795, 6.43775165, 5.991464547, 5.626821434, 5.318520074, 5.051457289,
                                  4.815891217, 4.605170186, 3.79423997, 3.218875825, 2.772588722, 2.407945609,
                                  2.099644249, 1.832581464, 1.597015392, 1.386294361, 1.195674002, 1.021651248,
                                  0.861565832, 0.713349888, 0.575364145, 0.446287103, 0.325037859, 0.210721031,
                                  0.188621359, 0.166763218, 0.145141386, 0.123750807, 0.102586589, 0.081643989,
                                  0.060918415, 0.050635616, 0.040405415, 0.030227276, 0.020100672, 0.010025084, 0])

        # 線形補間により、カイの二乗値を算出
        self.chi_squared_distribution = sp.interpolate.interp1d(self.p, self.square_x)
        self.__chi = self.chi_squared_distribution(p)

    def calc_error_ellipse(self, sigma):
        '''誤差楕円パラメータ算出
        引数：
            sigma：共分散行列(2x2)
        返り値：
            l：長軸の長さ
            s：短軸の長さ
            ang_rad：楕円の角度[rad]
        '''
        val, vec = np.linalg.eigh(sigma)
        idxmax = np.argmax(val)
        idxmin = np.argmin(val)
        vecmax = vec[idxmax]
        ang_rad = np.arctan2(vecmax[1], vecmax[0])
        l = np.sqrt(val[idxmax] * self.__chi) * 2
        y = np.sqrt(val[idxmin] * self.__chi) * 2
        return l, y, ang_rad

    def calc_chi(self, p, sigma):
        '''カイの二乗値算出
        引数：
            p：上側累積パーセント点(2自由)
        返り値：
            l：長軸の長さ
        '''
        chi = self.chi_squared_distribution(p)
        val, vec = np.linalg.eigh(sigma)
        idxmax = np.argmax(val)
        l = np.sqrt(val[idxmax] * chi) * 2
        return l

if __name__ == '__main__':
    # データ数
    data_num = 1000

    # 平均
    mu = np.array([[24.0],
                   [12.0]])
    # 共分散
    cov_xx = 16.00
    cov_yy = 9.00
    cov_xy = 5.48
    cov = np.array([[ cov_xx, cov_xy ],
                   [ cov_xy, cov_yy ]])

    P = np.random.multivariate_normal([mu[0, 0], mu[1, 0]], cov, data_num).T

    # 誤差楕円の信頼区間[%]
    confidence_interval = 99.0

    # グラフ描画
    # 背景を白にする
    fig = plt.figure(facecolor="w")
    ax = fig.add_subplot(111, aspect='equal')

    # 散布図をプロットする
    plt.scatter(P[0], P[1], color='r', marker='x', label="$K_1$")

    # ラベル
    plt.xlabel('$x$', size = 20)
    plt.ylabel('$y$', size = 20)

    # 軸
    plt.title('Error Ellipse')

    ee = ErrorEllipse(confidence_interval)
    x, y, ang_rad = ee.calc_error_ellipse(cov)

    e = patches.Ellipse((mu[0, 0], mu[1, 0]), x, y, angle = np.rad2deg(ang_rad), linewidth = 2, alpha = 0.2,
                         facecolor = 'yellow', edgecolor = 'black', label = 'Confidence Interval: %.2f[%%]' % confidence_interval)

    ax.add_patch(e)

    plt.axis('equal')
    plt.grid(True)
    plt.legend()

    plt.show()

