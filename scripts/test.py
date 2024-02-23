import cv2 as cv
import numpy as np
import subprocess
import time


proc = subprocess.Popen(['gnome-terminal', '--tab', '--', 'bash', '-c','roslaunch rl_tb3 tb3gazebo.launch'])

proc.wait()

print('waiting..find')


