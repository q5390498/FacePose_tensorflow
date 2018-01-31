# -*- coding:utf-8 -*-
__author__ = 'Merle'
import math

import numpy as np
import tensorflow as tf
import time

def poseEstimation(img, pt2d, pt3d, sess):

    height, width, nChannels = img.shape

    LeftVis = np.concatenate((np.arange(9), np.arange(31)+17, np.array([48, 60, 64, 54])))
    RightVis = np.concatenate((np.arange(9)+8, np.arange(31)+17, np.array([48, 60, 64, 54])))

    # Coarse Pose Estimate
    
    phil, gammal, thetal, t3d, f, r1 = PoseEstimation(pt2d[:, LeftVis], pt3d[:, LeftVis])
    phir, gammar, thetar, t3d, f, r1_ = PoseEstimation(pt2d[:, RightVis], pt3d[:, RightVis])

    if abs(gammal) > abs(gammar):
        phi = phil
        gamma = gammal
        theta = thetal
    else:
        phi = phir
        gamma = gammar
        theta = thetar

    return([phi, gamma, theta])

def tf_dot(a,b):
    b = tf.transpose(b)
    return tf.reduce_sum(tf.multiply(a,b))

def init_graph(pt2d, pt3d, pt2dm, pt3dm):
    pt2dm_0 = tf.expand_dims(tf.subtract(pt2d[0],tf.reduce_mean(pt2d[0])), axis=0)
    pt2dm_1 = tf.expand_dims(tf.subtract(pt2d[1],tf.reduce_mean(pt2d[1])), axis=0)
    pt3dm_0 = tf.expand_dims(tf.subtract(pt3d[0],tf.reduce_mean(pt3d[0])), axis=0)
    pt3dm_1 = tf.expand_dims(tf.subtract(pt3d[1],tf.reduce_mean(pt3d[1])), axis=0)
    pt3dm_2 = tf.expand_dims(tf.subtract(pt3d[2],tf.reduce_mean(pt3d[2])), axis=0)
    pt2dm = tf.concat([pt2dm_0, pt2dm_1], axis=0)
    # print ('pt2dm :', pt2dm.get_shape)
    pt3dm = tf.concat([pt3dm_0, pt3dm_1, pt3dm_2], axis=0)
    # print ('pt3dm :', pt3dm.get_shape)
    # 最小二乘方法计算R
    
    R1 = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(pt3dm, tf.transpose(pt3dm), name="matmul_c")), pt3dm, name="matmul_b"), tf.reshape(pt2dm[0],(-1,1)), name="matmul_a")
    R2 = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(pt3dm, tf.transpose(pt3dm))), pt3dm), tf.reshape(pt2dm[1],(-1,1)))
    # 计算出f
    R1 = tf.transpose(R1)
    R2 = tf.transpose(R2)
    f = (tf.sqrt(R1[0][0]**2+R1[0][1]**2+R1[0][2]**2)+tf.sqrt(R2[0][0]**2+R2[0][1]**2+R2[0][2]**2))/2
    R1 = tf.divide(R1, f)
    R2 = tf.divide(R2, f)
    R3 = tf.cross(R1, R2)
    print ("R1R2R3:", R1.shape,R2.shape,R3.shape,tf.concat((R1, R2, R3), axis=0))

    specVar = tf.concat((R1, R2, R3), axis=0)
    #这里算出的u,s,v与numpy里不一样，tf的作者说是numpy为了计算快一点,少做了一些操作
    s, U, V = tf.svd(specVar);
    V = tf.transpose(V) * tf.constant([[-1., -1., -1.], [1., 1., 1.], [-1., -1., -1.]], tf.float64)
    U = U * tf.constant([[-1., 1., -1.], [-1., 1., -1.], [-1., 1., -1.]], tf.float64)


    R = tf.matmul(U, V)
    R1 = R[0]
    R2 = R[1]
    R3 = R[2]
    
    R1 = tf.transpose(R1)
    R2 = tf.transpose(R2)
    R3 = tf.transpose(R3)
    
    # 使用旋转矩阵R恢复出旋转角度
    phi = tf.atan(R2[2]/R3[2])
    gamma = tf.atan(-R1[2]/(tf.sqrt(R1[0]**2+R1[1]**2)))
    theta = tf.atan(R1[1]/R1[0])
    # 使用R重新计算旋转平移矩阵，求出t
    pt3d = tf.concat([pt3d, tf.ones((1, pt3d.shape[1]), tf.float64)], axis=0)
    R1_orig = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(pt3d, tf.transpose(pt3d))), pt3d), tf.transpose(tf.reshape(pt2d[0],(1,-1))))
    R2_orig = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(pt3d, tf.transpose(pt3d))), pt3d), tf.transpose(tf.reshape(pt2d[1],(1,-1))))

    # t3d = tf.array([R1_orig[0, 3], R2_orig[0, 3], 0]).reshape((3, 1))
    print (R1_orig[3].shape)
    t3d = tf.reshape(tf.concat([R1_orig[3], R2_orig[3], tf.constant([0.0], tf.float64)], axis=0),(3,1))
    
    return (phi, gamma, theta, t3d, f)

def PoseEstimation_tf(pt2d, pt3d):
# 参照论文Optimum Fiducials Under Weak Perspective Projection，使用弱透视投影
    # 减均值，排除t，便于求出R
    pt2d_placeholder = tf.placeholder(tf.float64, pt2d.shape)
    pt3d_placeholder = tf.placeholder(tf.float64, pt3d.shape)
    # SVD use gpu
    pt2dm = tf.Variable(pt2d, "pt2dm", dtype=tf.float64)
    pt3dm = tf.Variable(pt3d, "pt3dm", dtype=tf.float64)
    
    init_OP = tf.global_variables_initializer();
    init_local_OP =tf.local_variables_initializer(); 
    phi, gamma, theta, t3d, f = init_graph(pt2d_placeholder, pt3d_placeholder, pt2dm, pt3dm)
    with tf.Session() as sess:
    # Initialize all tensorflow variables
        start = time.time()
        sess.run(init_OP)
        #sess.run(init_local_OP)
        #print 'initializing variables: {} s'.format(time.time()-start);

        start_time = time.time()

        feed_dict = {pt2d_placeholder:pt2d*1.0, pt3d_placeholder:pt3d*1.0}
        phi, gamma, theta, t3d, f = sess.run([phi, gamma, theta, t3d, f],feed_dict=feed_dict);
        #print("Tensorflow SVD ---: {} s" . format(time.time() - start_time));

    
    return (phi, gamma, theta, t3d, f)

def PoseEstimation(pt2d, pt3d):
# 参照论文Optimum Fiducials Under Weak Perspective Projection，使用弱透视投影
# 减均值，排除t，便于求出R
    
    pt2dm = np.zeros(pt2d.shape)
    pt3dm = np.zeros(pt3d.shape)
    
    pt2dm[0, :] = pt2d[0, :]-np.mean(pt2d[0, :])
    pt2dm[1, :] = pt2d[1, :]-np.mean(pt2d[1, :])
    pt3dm[0, :] = pt3d[0, :]-np.mean(pt3d[0, :])
    pt3dm[1, :] = pt3d[1, :]-np.mean(pt3d[1, :])
    pt3dm[2, :] = pt3d[2, :]-np.mean(pt3d[2, :])
    # 最小二乘方法计算R
    R1_ = np.dot(np.dot(np.mat(np.dot(pt3dm, pt3dm.T)).I, pt3dm), pt2dm[0, :].T)
    R2_ = np.dot(np.dot(np.mat(np.dot(pt3dm, pt3dm.T)).I, pt3dm), pt2dm[1, :].T)
    # 计算出f
    f = (math.sqrt(R1_[0, 0]**2+R1_[0, 1]**2+R1_[0, 2]**2)+math.sqrt(R2_[0, 0]**2+R2_[0, 1]**2+R2_[0, 2]**2))/2
    R1 = R1_/f
    R2 = R2_/f
    R3 = np.cross(R1, R2)
    print ("R1R2R3:", R1.shape,R2.shape,R3.shape,np.concatenate((R1, R2, R3), axis=0))

    # SVD 分解，重构 use cpu
    U, s, V = np.linalg.svd(np.concatenate((R1, R2, R3), axis=0), full_matrices=True)
    # print (type(U),type(s),type(V))
    # print ("np shape : ",[U.shape,s.shape,V.shape])


    U = np.matrix(U)
    V = np.matrix(V)

    R = np.dot(U, V)
    R1 = R[0, :]
    R2 = R[1, :]
    R3 = R[2, :]
    # 使用旋转矩阵R恢复出旋转角度
    phi = math.atan(R2[0, 2]/R3[0, 2])
    gamma = math.atan(-R1[0, 2]/(math.sqrt(R1[0, 0]**2+R1[0, 1]**2)))
    theta = math.atan(R1[0, 1]/R1[0, 0])
    # 使用R重新计算旋转平移矩阵，求出t
    pt3d = np.row_stack((pt3d, np.ones((1, pt3d.shape[1]))))
    R1_orig = np.dot(np.dot(np.mat(np.dot(pt3d, pt3d.T)).I, pt3d), pt2d[0, :].T)
    R2_orig = np.dot(np.dot(np.mat(np.dot(pt3d, pt3d.T)).I, pt3d), pt2d[1, :].T)

    t3d = np.array([R1_orig[0, 3], R2_orig[0, 3], 0]).reshape((3, 1))
    
    return(phi, gamma, theta, t3d, f, R1)
