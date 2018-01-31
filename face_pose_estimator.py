#coding=utf8
import dlib
import facial_feature_detector as feature_detection
import time
import numpy as np
import tensorflow as tf
import math

predictor_path = "./dlib_models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


class face_pose_estimator(object):
    """
    这个类依赖于facial_feature_detector.py

    具体使用:
    pose_estimator = face_pose_estimator('./pt3d.npy')
    angles, lmarks = pose_estimator.get_angles(resized_face)s

    """

    def __init__(self, pt3d_np):
        self.pt3d = np.load(pt3d_np)
        self.pt2d_shape = (2, 44) #(2, 44)是指左边或者右边关键点的数量，这个对于dlib来说，是固定值
        self.pt3d_shape = (3, 44) #同上
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.sess = tf.Session(config=config)

        self.pt2d_placeholder = tf.placeholder(tf.float64, self.pt2d_shape)
        self.pt3d_placeholder = tf.placeholder(tf.float64, self.pt3d_shape)

        self.phi_tensor, self.gamma_tensor, self.theta_tensor, self.t3d_tensor, self.f_tensor = self.init_graph(
                                                                                    self.pt2d_placeholder, self.pt3d_placeholder)
        with self.sess.as_default():
            init_OP = tf.global_variables_initializer()
            self.sess.run(init_OP)

    def get_landmark_2d(self, im):
        return feature_detection.get_landmarks(im, detector, predictor)

    def get_value_by_tensor(self, pt2d, pt3d):
        feed_dict = {self.pt2d_placeholder: pt2d * 1.0, self.pt3d_placeholder: pt3d * 1.0}
        return self.sess.run([self.phi_tensor, self.gamma_tensor, self.theta_tensor, self.t3d_tensor, self.f_tensor], \
                             feed_dict=feed_dict)

    def _poseEstimation(self, pt2d, pt3d):
        LeftVis = np.concatenate((np.arange(9), np.arange(31) + 17, np.array([48, 60, 64, 54])))
        RightVis = np.concatenate((np.arange(9) + 8, np.arange(31) + 17, np.array([48, 60, 64, 54])))

        # Coarse Pose Estimate
        phil, gammal, thetal, t3d, f = self.get_value_by_tensor(pt2d[:, LeftVis], pt3d[:, LeftVis])
        phir, gammar, thetar, t3d, f = self.get_value_by_tensor(pt2d[:, RightVis], pt3d[:, RightVis])

        # 根据gammal和gammar的大小，返回更大的那个
        if abs(gammal) > abs(gammar):
            phi = phil
            gamma = gammal
            theta = thetal
        else:
            phi = phir
            gamma = gammar
            theta = thetar

        return ([phi, gamma, theta])

    def init_graph(self, pt2d, pt3d):
        """
        参照论文Optimum Fiducials Under Weak Perspective Projection，使用弱透视投影

        :param pt2d: 2d关键点
        :param pt3d: 3d关键点
        :return:　phi, gamma, theta分别为三个方向的角度
        """
        # 减均值，排除t，便于求出R
        pt2dm_0 = tf.expand_dims(tf.subtract(pt2d[0], tf.reduce_mean(pt2d[0])), axis=0)
        pt2dm_1 = tf.expand_dims(tf.subtract(pt2d[1], tf.reduce_mean(pt2d[1])), axis=0)
        pt3dm_0 = tf.expand_dims(tf.subtract(pt3d[0], tf.reduce_mean(pt3d[0])), axis=0)
        pt3dm_1 = tf.expand_dims(tf.subtract(pt3d[1], tf.reduce_mean(pt3d[1])), axis=0)
        pt3dm_2 = tf.expand_dims(tf.subtract(pt3d[2], tf.reduce_mean(pt3d[2])), axis=0)
        pt2dm = tf.concat([pt2dm_0, pt2dm_1], axis=0)
        pt3dm = tf.concat([pt3dm_0, pt3dm_1, pt3dm_2], axis=0)

        # 最小二乘方法计算R
        R1 = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(pt3dm, tf.transpose(pt3dm), name="matmul_c")), pt3dm,name="matmul_b"), \
                        tf.reshape(pt2dm[0], (-1, 1)), name="matmul_a")
        R2 = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(pt3dm, tf.transpose(pt3dm))), pt3dm),tf.reshape(pt2dm[1], (-1, 1)))
        # 计算出f
        R1 = tf.transpose(R1)
        R2 = tf.transpose(R2)
        f = (tf.sqrt(R1[0][0] ** 2 + R1[0][1] ** 2 + R1[0][2] ** 2) + tf.sqrt(
            R2[0][0] ** 2 + R2[0][1] ** 2 + R2[0][2] ** 2)) / 2
        R1 = tf.divide(R1, f)
        R2 = tf.divide(R2, f)
        R3 = tf.cross(R1, R2)

        specVar = tf.concat((R1, R2, R3), axis=0)
        # 这里算出的u,v与numpy里不一样，tf的作者说是numpy为了计算快一点,少做了一些操作
        s, U, V = tf.svd(specVar)
        # 将U. V转换成与numpy计算结果一样
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
        phi = tf.atan(R2[2] / R3[2])
        gamma = tf.atan(-R1[2] / (tf.sqrt(R1[0] ** 2 + R1[1] ** 2)))
        theta = tf.atan(R1[1] / R1[0])
        # 使用R重新计算旋转平移矩阵，求出t
        pt3d = tf.concat([pt3d, tf.ones((1, pt3d.shape[1]), tf.float64)], axis=0)
        R1_orig = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(pt3d, tf.transpose(pt3d))), pt3d),
                            tf.transpose(tf.reshape(pt2d[0], (1, -1))))
        R2_orig = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(pt3d, tf.transpose(pt3d))), pt3d),
                            tf.transpose(tf.reshape(pt2d[1], (1, -1))))

        t3d = tf.reshape(tf.concat([R1_orig[3], R2_orig[3], tf.constant([0.0], tf.float64)], axis=0), (3, 1))

        return (phi, gamma, theta, t3d, f)

    def get_angles(self, im):
        """

        :param im:  使用crop出来的脸预测，但是不能使用全是脸的，最好在crop脸的时候往四个方向都扩充一些像素，比如20
        :return: 三个角度，以及关键点
        """
        # 取关键点
        pt2d = self.get_landmark_2d(im)
        lmarks = pt2d

        if len(pt2d):
            start = time.time()
            # 根据关键点计算角度
            Pose_Para = self._poseEstimation(pt2d, self.pt3d)
            print ("calauate pose cost : ", time.time() - start)
            return np.array(Pose_Para) * 180 / math.pi, lmarks
        else:
            print('can not find landmark!')
            return None, None
