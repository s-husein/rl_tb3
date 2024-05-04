#!/usr/bin/env python

from sensor_msgs.msg import CompressedImage, Image
import rospy
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import message_filters as msg_f

def add_noise(img, intensity=12):

        noise = np.random.randint(-intensity, intensity, img.shape, dtype=np.int8)


        noisy_img = (img+noise).clip(0, 255).astype(np.uint8)
        return noisy_img, noise

rgb_img = None

# def callback_rgb(ros_img):
#     rgb_img = CvBridge().imgmsg_to_cv2(ros_img)
#     rgb_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2GRAY)
#     cv.imshow('rgb', rgb_img)
#     cv.waitKey(1)

# def callback_depth(ros_img):
#         cv_img = CvBridge().imgmsg_to_cv2(ros_img)
#         cv_img = cv_img/7.0
#         cv_img = np.nan_to_num(cv_img)
#         cv_img = (cv_img*255).astype(np.uint8)
#         cv.imshow('depth', cv_img)
#         cv.waitKey(1)

r_img = None
de_img = None

def callback(ros_rgb_img, ros_d_img):
        global r_img, de_img
        rgb_img = CvBridge().imgmsg_to_cv2(ros_rgb_img)
        rgb_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2GRAY)
        d_img = CvBridge().imgmsg_to_cv2(ros_d_img)
        d_img = d_img/7.0
        d_img = np.nan_to_num(d_img)
        d_img = (d_img*255).astype(np.uint8)
        r_img, de_img = rgb_img, d_img
        cv.waitKey(1)




rospy.init_node('img_ros', anonymous=True)
rgb_img = rospy.wait_for_message('/camera/color/image_raw', Image,10)

img = rospy.topics.Subscriber()

# rgb_sub = msg_f.Subscriber('/camera/color/image_raw', Image)
# d_sub = msg_f.Subscriber('/camera/depth/image_rect_raw', Image)

# rospy.Subscriber('/camera/color/image_raw', Image, callback_rgb, queue_size=1)
# rospy.Subscriber('/camera/depth/image_rect_raw', Image, callback_depth, queue_size=1)
# ts = msg_f.TimeSynchronizer([rgb_sub, d_sub], 10)
# ts.registerCallback(callback)

# print(r_img, de_img)
# rospy.spin()

# print(rgb_img)

# cv.imshow('x', rgb_img)
# cv.waitKey()
# cv.destroyAllWindows()
    
#     cv_img = CvBridge().imgmsg_to_cv2(ros_img)
#     cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)

#     cv_img = cv_img/7.0

#     cv_img = np.nan_to_num(cv_img)
#     cv_img = (cv_img*255).astype(np.uint8)
#     cv_img = cv_img[:250, :]
#     cv_img = cv.resize(cv_img, (0, 0), fx = 0.05, fy = 0.05)
#     cv_img, noise = add_noise(cv_img)
#     cv.imshow('state', cv_img)
#     cv.waitKey()
#     print(cv_img.shape)
#     print(f'average {np.average(cv_img)}, black pixels: {np.sum(cv_img < 7)}')
#     cv.destroyAllWindows()