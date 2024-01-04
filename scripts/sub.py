#!/usr/bin/env python

from sensor_msgs.msg import Image
import rospy
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
        

if __name__ == '__main__':
    file = open('/home/user/fyp/src/custom_turtle/images/image_data.txt', 'w')
    rospy.init_node('img_ros', anonymous=True)
    
    ros_img = rospy.wait_for_message('/camera/depth/image_raw', Image,10)
    cv_img = CvBridge().imgmsg_to_cv2(ros_img)
    # cv.normalize(cv_img, cv_img, 0, 1, norm_type=cv.NORM_MINMAX)
    cv_img = cv_img/6.0
    cv_img = np.nan_to_num(cv_img)
    cv_img = (cv_img*255).astype(np.uint8)
    cv_img = cv_img[140:410, 150:1130]
    # noise = np.zeros(cv_img.shape, dtype=np.uint8)
    # noise = cv.randn(noise, 0, 10)
    # cv_img = cv.add(cv_img, noise)
    # cv_img = cv.resize(cv_img, (0, 0), fx = 0.04, fy = 0.06)
    cv.imshow('state', cv_img)
    cv.waitKey()
    cv.imwrite('/home/user/fyp/src/custom_turtle/images/ros_depth_state1.png', cv_img)
    print(cv_img.shape)
    print(f'average {np.average(cv_img)}, black pixels: {np.sum(cv_img < 15)}')
    print(cv_img, file=file)
    cv.destroyAllWindows()