from nets import make_dnn
from gymenv import Gym
import cv2 as cv
import threading as thr
from multiprocessing import Pool
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters as msg_f
import time



# conv_layers = [[16, 3, 3],
#                [32, 1, 1]]
# hid_layers = [64, 64]
# action_space = 'cont'
# net_type = 'actor'
# max_pool = [2, 2]

# env = Gym(obs_scale_factor=0.1,
#            action_space=action_space, conv_layers=conv_layers)
# # model = make_dnn(env, hid_layers=hid_layers, conv_layers=conv_layers,
# #                  action_space=action_space, net_type=net_type,
# #                  max_pool=max_pool)

# rgb, depth = env.get_observation()

# cv.imshow('x', depth)
# cv.waitKey()
# cv.destroyAllWindows()

# rgb, depth = env.get_observation()

# print(rgb.shape)
# print(depth.shape)

rgb = None
depth = None

rospy.init_node('get_img')
def callback(ros_img):
    global rgb
    rgb = CvBridge().imgmsg_to_cv2(ros_img)

def d_callback(ros_img):
    global depth
    depth = CvBridge().imgmsg_to_cv2(ros_img)/7.0

rgb_s = rospy.Subscriber('/camera/color/image_raw', Image, callback)
dpth_s = rospy.Subscriber('/camera/depth/image_rect_raw', Image, d_callback)

while True:
    if rgb is not None and depth is not None:
        cv.imshow('x', rgb)
        cv.imshow('y', depth)
        cv.waitKey(1)

cv.destroyAllWindows()







