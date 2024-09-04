#!/usr/bin/env python

from nav_msgs.msg import OccupancyGrid
import rospy
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import torch
from sensor_msgs.msg import Image

# for i in range(20):
rospy.init_node('x')
ros_msg = rospy.wait_for_message('/camera/depth/image_rect_raw', Image)

cv_img = CvBridge().imgmsg_to_cv2(ros_msg)
cv_img = cv_img/8.0
cv_img = (cv_img*255).astype(np.uint8)
cv_img = np.nan_to_num(cv_img, nan=0.0)
cv.imwrite('depth_image.png', cv_img)

cv_img = cv.resize(cv_img, (0, 0), fx = 0.1, fy = 0.1)
cv_img_s = cv.resize(cv_img, (0, 0), fx = 10, fy = 10)

cv.imwrite('rescaled_depth_image.png', cv_img_s)

cv_img = cv_img[:27, :]
cv_img_c = cv.resize(cv_img, (0, 0), fx = 10, fy = 10)

cv.imwrite('cropped_depth_image.png', cv_img_c)

noise = np.random.randint(-10, 10, cv_img.shape, dtype=np.int8)
noisy_img = (cv_img+noise).clip(0, 255).astype(np.uint8)

cv_img_n = cv.resize(noisy_img, (0, 0), fx = 10, fy = 10)

cv.imwrite('noisy_depth_image.png', cv_img_n)

