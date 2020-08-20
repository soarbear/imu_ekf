#!/usr/bin/env python
# -*- coding: utf-8 -*-
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# brief        IMU angular estimation with extended Kalman Filter
# author       Tateo YANAGI
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
from matplotlib import animation, patches
import matplotlib.pyplot as plot
from lib import error_ellipse

P1 = []
P2 = []
P3 = []
time_s = 0

class extKalmanFilter(object):
    def __init__(self, period_ms):
        self.dts = period_ms / 1000

        # initialize angular velocity
        angular_vel_x = np.deg2rad(10.0)
        angular_vel_y = np.deg2rad(10.0)
        angular_vel_z = np.deg2rad(0.0)
        self.gyro_angular_vel = np.array([[angular_vel_x], [angular_vel_y], [angular_vel_z]])

        # observation matrix H
        self.H = np.diag([1.0, 1.0])

        # system noise variance
        self.Q = np.diag([1.74E-2*self.dts*self.dts, 1.74E-2*self.dts*self.dts])

        # observation noise variance
        self.R = np.diag([1.0*self.dts*self.dts, 1.0*self.dts*self.dts])

        # initialize true status
        self.x_true = np.array([[0.0], [0.0]])

        # initialize prediction
        self.x_bar = self.x_true

        # initialize eastimation
        self.x_hat = self.x_true

        # initialize covariance
        self.P = self.Q

        # initialize jacbian matrix of H
        self.jacobianH = np.diag([1.0, 1.0])

    def get_ekf(self):
        # Ground Truth
        self.x_true = self.get_status(self.x_true)

        # [step1] prediction
        self.x_bar = self.get_status(self.x_hat)

        # jacobian matrix
        jacobianF = self.get_jacobianF(self.x_bar)

        # pre_covariance
        P_bar = (jacobianF @ self.P @ jacobianF.T) + self.Q

        # observation
        w = np.random.multivariate_normal([0.0, 0.0], self.R, 1).T
        y = (self.H @ self.x_true) + w

        # [step2] update the filter
        s = (self.H @ P_bar @ self.H.T) + self.R
        K = (P_bar @ self.H.T) @ np.linalg.inv(s)

        # eastimation
        e = y - (self.jacobianH @ self.x_bar)
        self.x_hat = self.x_bar + (K @ e)

        # post_covariance
        I = np.identity(self.x_hat.shape[0])
        self.P = (I - K @ self.H) @ P_bar

        return self.x_true, y, self.x_hat, self.P

    # get status
    def get_status(self, x):
        tri = self.get_trigonometrxic(x)
        Q = np.array([[1, tri[0,1]*tri[1,2], tri[0,0]*tri[1,2]], [0, tri[0,0], -tri[0,1]]])
        x =  x + (Q @ self.gyro_angular_vel) * self.dts
        return x

    # get jacobian matrix of F
    def get_jacobianF(self, x):
        g = self.gyro_angular_vel
        tri = self.get_trigonometrxic(x)
        jacobianF = np.array([[1.0+(tri[0,0]*tri[1,2]*g[1][0]-tri[0,1]*tri[1,2]*g[2][0])*self.dts, (tri[0,1]/tri[1,0]/tri[1,0]*g[1][0]+tri[0,0]/tri[1,0]/tri[1,0]*g[2][0])*self.dts], 
                              [-(tri[0,1]*g[1][0]+tri[0,0]*g[2][0])*self.dts, 1.0]])
        return jacobianF

    # get trigonometrxic of roll&pitch
    def get_trigonometrxic(self, x):
        return np.array([[np.cos(x[0][0]), np.sin(x[0][0]), np.tan(x[0][0])], [np.cos(x[1][0]), np.sin(x[1][0]), np.tan(x[1][0])]])

#==============================================================================
# brief        animate
# author       Takuya Niibori
#==============================================================================
def animate(i, ekf, period_ms):
    global P1, P2, P3
    global time_s, ee
    col_x_true = 'red'
    col_y = 'purple'
    col_x_hat = 'blue'

    time_s += period_ms / 1000
    x_true, obs, x_hat, P = ekf.get_ekf()
    plot.cla()
    ax1 = plot.subplot2grid((1, 1), (0, 0))

    # scatter true x
    P1.append(x_true[0:2, :])
    a, b = np.array(np.concatenate(P1, axis=1))
    ax1.plot(a, b, c=col_x_true, linewidth=1.0, linestyle='-', label='Ground Truth')
    ax1.scatter(x_true[0], x_true[1], c=col_x_true, marker='o', alpha=0.5)

    # scatter observation y
    P2.append(obs)
    a, b = np.array(np.concatenate(P2, axis=1))
    ax1.scatter(a, b, c=col_y, marker='o', alpha=0.5, label='Observation')

    # scatter estimation x
    P3.append(x_hat[0:2, :])
    a, b = np.array(np.concatenate(P3, axis=1))
    ax1.plot(a, b, c=col_x_hat, linewidth=1.0, linestyle='-', label='Estimation')
    ax1.scatter(x_hat[0], x_hat[1], c=col_x_hat, marker='o', alpha=0.5)

    # create error ellipse
    Pxy = P[0:2, 0:2]
    x, y, ang_rad = ee.calc_error_ellipse(Pxy)
    e = patches.Ellipse((x_hat[0, 0], x_hat[1, 0]), x, y, angle = np.rad2deg(ang_rad), linewidth=2, alpha=0.2, facecolor='yellow', edgecolor='black', label='Error Ellipse: %.2f[%%]' % confidence_interval)
    print('time:{0:.3f}[s], x-cov:{1:.3e}[rad2], y-cov:{2:.3e}[rad2], xy-cov:{3:.3e}[rad2]'.format(time_s, P[0, 0], P[1, 1], P[1, 0]))

    ax1.add_patch(e)
    ax1.set_xlabel('Roll [rad]')
    ax1.set_ylabel('Pitch [rad]')
    ax1.set_title('Localization by EKF')
    ax1.axis('equal', adjustable='box')
    ax1.grid()
    ax1.legend(fontsize=10)

if __name__ == '__main__':
    global ee
    # reliable interval of error ellipse[%]
    confidence_interval = 99.0
    ee = error_ellipse.ErrorEllipse(confidence_interval)

    period_ms = 100
    frame_cnt = int(12000 / period_ms)
    fig = plot.figure(figsize=(12, 9))
    ekf = extKalmanFilter(period_ms)
    ani = animation.FuncAnimation(fig, animate, frames=frame_cnt, fargs=(ekf, period_ms), blit=False, interval=period_ms, repeat=False)
    ani.save('imu_by_ekf.mp4', bitrate=5000)
    #ani.save('imu_by_ekf.gif', writer='imagemagick')
    plot.show()

