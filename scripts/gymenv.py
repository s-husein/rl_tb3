import gymnasium as gym
from std_srvs.srv import Empty
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import cv2 as cv
from cv_bridge import CvBridge
import numpy as np
import tf.transformations as tft
import random
import math


class Gym(gym.Env):

    def __init__(self, positions = [(0, 0)], angles = [0], action_space = 'disc', bins=7, obs_scale_factor=1, conv_layers=None):
        self._action_space = action_space
        self._bins = bins
        self.POS = positions
        self.ANGLES = angles
        self.scal_fac = obs_scale_factor
        self.conv_layers = conv_layers
        rospy.init_node("gym_node", anonymous=True)
        img_shape = (int(360*obs_scale_factor), int(640*obs_scale_factor), 1)
        self.img_area = np.prod(img_shape)
        self.observation_space = gym.spaces.Box(0, 255, shape=img_shape, dtype=np.uint8) #a grayscale depth image
        if self._action_space == 'disc':
            self.action_space = gym.spaces.Discrete(3)
        else: self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape = (2,), dtype=np.float32)
        self.action_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1, latch=True)

    def step(self, _action):
        if self._action_space=='disc':
            action = self.act_d(_action)
        elif self._action_space == 'cont':
            action = self.conv_action(_action[0], _action[1])
            self.act_c(action)
        elif self._action_space == 'discretize':
            encode = np.linspace(-1, 1, self._bins)
            action = self.conv_action(encode[_action[0]], encode[_action[1]])
            self.act_c(action)
        observation = self.get_observation()
        reward, done = self.get_reward(action, observation)
        return (observation/255.0).astype(np.float32), reward, done, False, {}

    def reset(self, seed=None):
        super().reset(seed=seed)
        pos = random.choice(self.POS)
        angle = random.choice(self.ANGLES)

        rospy.ServiceProxy('/gazebo/reset_simulation', Empty)()        
        self.set_model_state(pos, angle)
        observation = self.get_observation()
        return (observation/255.0).astype(np.float32), {}
    
    def get_reward(self, action, state):#contin.. action space rewards
        done = False
        reward = 0

        if self._action_space == 'disc':
            reward = 0.03
        else:
            reward = (action[0])/(abs(action[1]) + 0.1) - 0.05
        
        if (np.sum(state < 7) > 0.05*self.img_area):
            reward = -100
            done = True
        
        return reward, done

    def act_d(self, action): #for discrete action space
        act = Twist()
        act.linear.x = 0.15
            
        if action == 1:
            act.angular.z = -0.5
        elif action == 2:
            act.angular.z = 0.5
        self.action_pub.publish(act)
        return act

    def act_c(self, action):#contin.. action space
        pub_act = Twist()
        pub_act.linear.x, pub_act.angular.z = action[0], action[1]
        self.action_pub.publish(pub_act)

    def conv_action(self, lin_act, ang_act):
            return np.clip((1/(1 + np.exp(-7*lin_act)))*0.22, 0.0, 0.22), np.clip(np.tanh(2.5*ang_act)*0.5, -0.5, 0.5)


    def get_observation(self):
        ros_img = rospy.wait_for_message('/camera/depth/image_rect_raw', Image, 10)
        cv_img = CvBridge().imgmsg_to_cv2(ros_img)
        cv_img = cv_img/10.0
        cv_img = (cv_img*255).astype(np.uint8)
        cv_img = np.nan_to_num(cv_img, nan=0.0)
        cv_img = cv.resize(cv_img, (0, 0), fx = self.scal_fac, fy=self.scal_fac)
        if self.conv_layers is not None:
            cv_img = np.expand_dims(cv_img, 0)
        return cv_img


    def set_model_state(self, pos, angle):
        quat = tft.quaternion_from_euler(0, 0, angle*(math.pi/180))
        bot = ModelState()
        bot.model_name = 'fyp_bot'
        bot.pose.position.x = pos[0]
        bot.pose.position.y = pos[1]

        bot.pose.orientation.x = quat[0]
        bot.pose.orientation.y = quat[1]
        bot.pose.orientation.z = quat[2]
        bot.pose.orientation.w = quat[3]
        rospy.wait_for_service('/gazebo/set_model_state', 3)
        rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)(bot)
    
    def render(self):
        state = self.get_observation()
        cv.imshow('state', state.squeeze())
        cv.waitKey(1)