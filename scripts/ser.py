#!/usr/bin/env python

import rospy
import tf.transformations as tft

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import math

if __name__ == '__main__':
    theta = 180
    quad = tft.quaternion_from_euler(0, 0, theta*(math.pi/180))

    model_state = ModelState()
    model_state.model_name='fyp_bot'
    model_state.pose.position.x = 0.5
    model_state.pose.position.y = 0
    model_state.pose.orientation.x = quad[0]
    model_state.pose.orientation.y = quad[1]
    model_state.pose.orientation.z = quad[2]
    model_state.pose.orientation.w = quad[3]


    set_model = rospy.ServiceProxy('/gazebo/reset_simulation', SetModelState)
    set_model(model_state)