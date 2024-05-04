from nets import make_dnn
from gymenv import Gym
import cv2 as cv
import time
import multiprocessing as mp
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters as msg_f



conv_layers = [[16, 3, 3],
               [32, 1, 1]]
hid_layers = [64, 64]
action_space = 'cont'
net_type = 'actor'
max_pool = [2, 2]

env = Gym(obs_scale_factor=0.1,
           action_space=action_space, conv_layers=conv_layers)
# # model = make_dnn(env, hid_layers=hid_layers, conv_layers=conv_layers,
# #                  action_space=action_space, net_type=net_type,
# #                  max_pool=max_pool)

while True:
    rgb, depth = env.get_observation()

    cv.imshow('x', rgb)
    cv.imshow('y', depth)
    cv.waitKey(1)





