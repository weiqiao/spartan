import unittest
import subprocess
import psutil
import sys
import os
import numpy as np
import time
import socket

# ROS
import rospy
import actionlib
import sensor_msgs.msg
import geometry_msgs.msg
import trajectory_msgs.msg
import rosgraph
import tf2_ros


# spartan
from spartan.utils.ros_utils import JointStateSubscriber
from spartan.utils.ros_utils import RobotService
import spartan.utils.ros_utils as rosUtils
import spartan.utils.utils as spartan_utils
import robot_control.control_utils as control_utils
# spartan ROS
import robot_msgs.msg



def quaternion_to_rotation_matrix(q):
    res = np.zeros((3,3))
    res[0][0] = q[0]**2-q[1]**2-q[2]**2+q[3]**2
    res[0][1] = 2*(q[0]*q[1]-q[2]*q[3])
    res[0][2] = 2*(q[0]*q[2]+q[1]*q[3])
    res[1][0] = 2*(q[0]*q[1]+q[2]*q[3])
    res[1][1] = -q[0]**2+q[1]**2-q[2]**2+q[3]**2
    res[1][2] = 2*(q[1]*q[2]-q[0]*q[3])
    res[2][0] = 2*(q[0]*q[2]-q[1]*q[3])
    res[2][1] = 2*(q[1]*q[2]+q[0]*q[3])
    res[2][2] = -q[0]**2-q[1]**2+q[2]**2+q[3]**2
    return res


def axis_angle_to_rotation_matrix(u, theta):
    def skew_sym(u):
        res = np.zeros((3, 3))
        res[0][1] = -u[2]
        res[0][2] = u[1]
        res[1][0] = u[2]
        res[1][2] = -u[0]
        res[2][0] = -u[1]
        res[2][1] = u[0]
        return res
    res = np.zeros((3,3))
    res += np.cos(theta)*np.identity(3)
    res += np.sin(theta)*skew_sym(u)
    res += (1-np.cos(theta))*(u.reshape(-1,1)).dot(u.reshape(1,-1))
    return res

def rotation_matrix_to_axis_angle(R):
    # only return theta
    return np.arccos((R[0][0]+R[1][1]+R[2][2]-1)/2)

def quaternion_to_axis_angle(q):
    theta = np.arccos(q[3])*2
    u = np.zeros(3)
    sin_theta_over_2 = np.sin(theta/2)
    u[0] = q[0]/sin_theta_over_2
    u[1] = q[1]/sin_theta_over_2
    u[2] = q[2]/sin_theta_over_2
    return theta, u

def axis_angle_to_quaternion(u, theta):
    q = np.zeros(4)
    first_three_entries = np.sin(theta/2)*u
    q[0] = first_three_entries[0]
    q[1] = first_three_entries[1]
    q[2] = first_three_entries[2]
    q[3] = np.cos(theta/2)
    return q