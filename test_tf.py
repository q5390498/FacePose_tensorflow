#coding=utf8
import tensorflow as tf
import numpy as np
import math


def PoseEstimation(pt2d, pt3d):
    # 参照论文Optimum Fiducials Under Weak Perspective Projection，使用弱透视投影
    # 减均值，排除t，便于求出R

    pt2dm = np.zeros(pt2d.shape)
    pt3dm = np.zeros(pt3d.shape)

    pt2dm[0, :] = pt2d[0, :] - np.mean(pt2d[0, :])
    pt2dm[1, :] = pt2d[1, :] - np.mean(pt2d[1, :])
    pt3dm[0, :] = pt3d[0, :] - np.mean(pt3d[0, :])
    pt3dm[1, :] = pt3d[1, :] - np.mean(pt3d[1, :])
    pt3dm[2, :] = pt3d[2, :] - np.mean(pt3d[2, :])
    # 最小二乘方法计算R
    R1_ = np.dot(np.dot(np.mat(np.dot(pt3dm, pt3dm.T)).I, pt3dm), pt2dm[0, :].T)
    print ("numpy R1_:", R1_)
    R2_ = np.dot(np.dot(np.mat(np.dot(pt3dm, pt3dm.T)).I, pt3dm), pt2dm[1, :].T)
    print ("numpy R2_:", R2_)
    # 计算出f
    f = (math.sqrt(R1_[0, 0] ** 2 + R1_[0, 1] ** 2 + R1_[0, 2] ** 2) + math.sqrt(
        R2_[0, 0] ** 2 + R2_[0, 1] ** 2 + R2_[0, 2] ** 2)) / 2
    R1 = R1_ / f
    R2 = R2_ / f
    R3 = np.cross(R1, R2)

    print ("numpy R1:",(R1))
    print ("numpy R2:", (R2))
    print ("numpy R3:", (R3))

    # SVD 分解，重构 use cpu
    U, s, V = np.linalg.svd(np.concatenate((R1, R2, R3), axis=0), full_matrices=True)
    # print (type(U),type(s),type(V))
    # print ("np shape : ",[U.shape,s.shape,V.shape])
    print ("numpy U:", U)
    print ("numpy s:", s)
    print ("numpy V:", V)

    U = np.matrix(U)
    V = np.matrix(V)

    R = np.dot(U, V)
    R1 = R[0, :]
    R2 = R[1, :]
    R3 = R[2, :]
    # 使用旋转矩阵R恢复出旋转角度
    phi = math.atan(R2[0, 2] / R3[0, 2])
    gamma = math.atan(-R1[0, 2] / (math.sqrt(R1[0, 0] ** 2 + R1[0, 1] ** 2)))
    theta = math.atan(R1[0, 1] / R1[0, 0])
    # 使用R重新计算旋转平移矩阵，求出t
    pt3d = np.row_stack((pt3d, np.ones((1, pt3d.shape[1]))))
    R1_orig = np.dot(np.dot(np.mat(np.dot(pt3d, pt3d.T)).I, pt3d), pt2d[0, :].T)
    R2_orig = np.dot(np.dot(np.mat(np.dot(pt3d, pt3d.T)).I, pt3d), pt2d[1, :].T)

    t3d = np.array([R1_orig[0, 3], R2_orig[0, 3], 0]).reshape((3, 1))
    print ("numpy ret : ", phi, gamma, theta)
    return (phi, gamma, theta, t3d, f, R1)


def init_graph(pt2d, pt3d, pt2dm=None, pt3dm=None):
    sess = tf.Session()
    pt2d = tf.constant(pt2d, tf.float64)
    pt3d = tf.constant(pt3d, tf.float64)
    pt2dm_0 = tf.expand_dims(tf.subtract(pt2d[0], tf.reduce_mean(pt2d[0])), axis=0)
    pt2dm_1 = tf.expand_dims(tf.subtract(pt2d[1], tf.reduce_mean(pt2d[1])), axis=0)
    pt3dm_0 = tf.expand_dims(tf.subtract(pt3d[0], tf.reduce_mean(pt3d[0])), axis=0)
    pt3dm_1 = tf.expand_dims(tf.subtract(pt3d[1], tf.reduce_mean(pt3d[1])), axis=0)
    pt3dm_2 = tf.expand_dims(tf.subtract(pt3d[2], tf.reduce_mean(pt3d[2])), axis=0)
    pt2dm = tf.concat([pt2dm_0, pt2dm_1], axis=0)
    # print ('pt2dm :', pt2dm.get_shape)
    pt3dm = tf.concat([pt3dm_0, pt3dm_1, pt3dm_2], axis=0)
    # print ('pt3dm :', pt3dm.get_shape)
    # 最小二乘方法计算R

    R1_ = tf.matmul(
        tf.matmul(tf.matrix_inverse(tf.matmul(pt3dm, tf.transpose(pt3dm), name="matmul_c")), pt3dm, name="matmul_b"),
        tf.reshape(pt2dm[0], (-1, 1)), name="matmul_a")
    print ("tf R1_:", sess.run(R1_))
    R2_ = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(pt3dm, tf.transpose(pt3dm))), pt3dm),
                    tf.reshape(pt2dm[1], (-1, 1)))
    print ("tf R2_:", sess.run(R2_))
    # 计算出f
    R1 = tf.transpose(R1_)
    print ("tf R1:", sess.run(R1))
    R2 = tf.transpose(R2_)
    f = (tf.sqrt(R1[0][0] ** 2 + R1[0][1] ** 2 + R1[0][2] ** 2) + tf.sqrt(
        R2[0][0] ** 2 + R2[0][1] ** 2 + R2[0][2] ** 2)) / 2
    R1 = tf.divide(R1, f)
    R2 = tf.divide(R2, f)
    R3 = tf.cross(R1, R2)
    print ("tf R1:", sess.run(R1))
    print ("tf R2:", sess.run(R2))
    print ("tf R3:", sess.run(R3))
    specVar = tf.concat((R1, R2, R3), axis=0)
    # 这里算出的u,s,v与numpy里不一样，tf的作者说是numpy为了计算快一点,少做了一些操作
    s, U, V = tf.svd(specVar);
    V = tf.transpose(V) * tf.constant([[1., 1., 1.], [-1., -1., -1.], [1., 1., 1.]], tf.float64)
    U = U * tf.constant([[1., 1., 1.], [1., 1., 1.], [-1., -1., -1.]], tf.float64)
    print ("tf U:", sess.run(U))
    print ("tf s:", sess.run(s))
    print ("tf V:", sess.run(V))


    R = tf.matmul(U, V)
    R1 = R[0]
    R2 = R[1]
    R3 = R[2]

    R1 = tf.transpose(R1)
    R2 = tf.transpose(R2)
    R3 = tf.transpose(R3)

    # 使用旋转矩阵R恢复出旋转角度
    phi = tf.atan(R2[2] / R3[2])
    gamma = tf.atan(-R1[2] / (tf.sqrt(R1[0] ** 2 + R1[1] ** 2)))
    theta = tf.atan(R1[1] / R1[0])
    # 使用R重新计算旋转平移矩阵，求出t
    pt3d = tf.concat([pt3d, tf.ones((1, pt3d.shape[1]), tf.float64)], axis=0)
    R1_orig = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(pt3d, tf.transpose(pt3d))), pt3d),
                        tf.transpose(tf.reshape(pt2d[0], (1, -1))))
    R2_orig = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(pt3d, tf.transpose(pt3d))), pt3d),
                        tf.transpose(tf.reshape(pt2d[1], (1, -1))))

    # t3d = tf.array([R1_orig[0, 3], R2_orig[0, 3], 0]).reshape((3, 1))
    t3d = tf.reshape(tf.concat([R1_orig[3], R2_orig[3], tf.constant([0.0], tf.float64)], axis=0), (3, 1))
    print ("tf U:", sess.run(phi), sess.run(gamma), sess.run(theta))
    return (phi, gamma, theta, t3d, f, R1_)


np.random.seed(2019)
pt2d_ = np.random.random((2, 44))
pt3d_ = np.random.random((3, 44))

PoseEstimation(pt2d_, pt3d_)
init_graph(pt2d_, pt3d_)

