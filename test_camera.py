# coding=utf-8
# Powered by SoaringNova Technology Company
import cv2
import numpy as np
import math
import sys
import time
import tensorflow as tf
sys.path.insert(0, '../facenet/src/')
import detect_face

import dlib
from face_pose_estimator import face_pose_estimator

predictor_path = "./dlib_models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

cam_matrix = lambda img_shape: np.array([
    [img_shape[0], 0, img_shape[0] / 2],  # image.width/2
    [0, img_shape[0], img_shape[1] / 2],  # image.height/2
    [0, 0, 1]], dtype=np.double)


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


class human_face_detector():
    """Usage
    hfd = human_face_detector()
    source_path = '/root/dl-data/datasets/ageandgender/192.168.199.31/datashare/luoyulong/dataset/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/faces/100003415@N08/coarse_tilt_aligned_face.2174.9523333835_c7887c3fde_o.jpg'
    face_image = misc.imread(source_path)  # must be misc.imread , cv2.imread did not work , don't know why
    ret = hfd.detect_human(face_image)
    """

    def __init__(self):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        self.sess = tf.Session(config=config)
        with self.sess.as_default():
            with tf.variable_scope('pnet'):
                data = tf.placeholder(tf.float32, (None, None, None, 3), 'input')
                pnet = detect_face.PNet({'data': data})
                pnet.load('../facenet/src/align/det1.npy', self.sess)
            with tf.variable_scope('rnet'):
                data = tf.placeholder(tf.float32, (None, 24, 24, 3), 'input')
                rnet = detect_face.RNet({'data': data})
                rnet.load('../facenet/src/align/det2.npy', self.sess)
            with tf.variable_scope('onet'):
                data = tf.placeholder(tf.float32, (None, 48, 48, 3), 'input')
                onet = detect_face.ONet({'data': data})
                onet.load('../facenet/src/align/det3.npy', self.sess)

            self.pnet_fun = lambda img: self.sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'),
                                                      feed_dict={'pnet/input:0': img})
            self.rnet_fun = lambda img: self.sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'),
                                                      feed_dict={'rnet/input:0': img})
            self.onet_fun = lambda img: self.sess.run(
                ('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0', 'onet/conv5/conv5:0'), feed_dict={'onet/input:0': img})

        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor

    def detect_human(self, img):
        bounding_boxes, points, features = detect_face.detect_face(img, self.minsize, self.pnet_fun, self.rnet_fun, self.onet_fun,
                                                         self.threshold, self.factor)
        ret0 = bounding_boxes[0][0]
        ret1 = bounding_boxes[0][1]
        ret2 = bounding_boxes[0][2]
        ret3 = bounding_boxes[0][3]
        return bounding_boxes, points, features



if __name__ == '__main__':

    #cam = SXCamera().get(id=5)  # type: SXCamera
    address = 'rtsp://admin:sxwl12345678@192.168.199.45/h264/ch1/main/av_stream'
    video = cv2.VideoCapture(address, cv2.CAP_GSTREAMER)
    wind_name = 'displayer'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)
    face_detector = human_face_detector()
    angles = [-1,-1,-1,-1]

    #pt3d = np.load('./pt3d.npy')
    pose_estimator = face_pose_estimator('./pt3d.npy')

    while True:
        success, im = video.read()
        if not success: continue
        try:
            im = cv2.resize(im, (1080,720))
            start = time.time()
            bboxs, landmarks, _ = face_detector.detect_human(im)
            print("MTCNN cost : ", time.time() - start)
        except:
            continue
        
        max_idx =0
        max_size = 0
        
        for idx, box in enumerate(bboxs):
            if max_size < box[2]-box[0]:
                max_idx = idx
                max_size = box[2] - box[0]

        bbox = map(int, bboxs[max_idx])
        bbox[0] = max(0,bbox[0]-20)
        bbox[1] = max(0,bbox[1]-20)
        bbox[2] = min(1080,bbox[2]+20)
        bbox[3] = min(720,bbox[3]+20)
        cv2.rectangle(im, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,255,0), 2)
        
        scale = 1
 
        face = im[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        
        resized_face = face
        resized_face = cv2.resize(face, (face.shape[1]/scale,face.shape[0]/scale), cv2.INTER_AREA)
        
        start = time.time()
        
        sess = None
        #angles, lmarks = get_angle(resized_face, pt3d, sess)
        angles, lmarks = pose_estimator.get_angles(resized_face)
        print ("angles :", angles)
        print("get lmarks cost : ", time.time() - start)

        # show result
        if not lmarks == None:
            lmarks = lmarks * scale
            x_list = lmarks[0]
            y_list = lmarks[1]
            for i in xrange(len(x_list)):
                point = (int(bbox[0] + x_list[i]), int(bbox[1] + y_list[i]))
                cv2.circle(im, point, 1, (255,0,0),1)
        if not angles == None:
            cv2.putText(im, 'X: %.2f, Y: %.2f, Z: %.2f '%(angles[0], angles[1], angles[2]), (30, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('123', im)
            pass
        else:
            print("do not find face")
            cv2.imshow('123', im)
            pass
        cv2.waitKey(1)

