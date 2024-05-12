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



conv_layers = [[16, 3, 3],
               [32, 1, 1]]
hid_layers = [64, 64]
action_space = 'cont'
net_type = 'actor'
max_pool = [2, 2]

env = Gym(obs_scale_factor=0.1,
           action_space=action_space, conv_layers=conv_layers)


while True:
    d, r = env.get_observation()
    cv.imshow('d', d)
    cv.imshow('r',r)
    cv.waitKey(1)


# model = make_dnn(env, hid_layers=hid_layers, conv_layers=conv_layers,
#                  action_space=action_space, net_type=net_type,
#                  max_pool=max_pool, img_type='rgb')

# print(model)

# state = env.observation_space.sample()

# cv.imshow('x', state[1])
# cv.waitKey()
# cv.destroyAllWindows()


# rgb, depth = env.get_observation()


# eni da sequence michal
# rgb_img = None
# depth_img = None




# dpth_s = rospy.Subscriber('/camera/depth/image_rect_raw', Image, depth_cb)
# rgb_s = rospy.Subscriber('/camera/color/image_raw', Image, rgb_cb)

# while True:
#     cv.imshow('x', rgb_img)
#     cv.imshow('y', depth_img)
#     cv.waitKey(1)


# e shi da sequence run musha
# while True:



# rgb, depth = env.get_observation()

# # cv.imshow('x', rgb)
# # cv.waitKey()
# # cv.destroyAllWindows()
# import torch
# rgb_ = torch.tensor(rgb/255.0, dtype = torch.float32).unsqueeze(0)
# print(rgb_.shape)

# out = model(rgb_)

# print(out)

# print(rgb.shape)
# print(depth.shape)










