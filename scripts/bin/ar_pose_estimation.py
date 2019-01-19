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
import std_msgs

# spartan
from spartan.utils.ros_utils import JointStateSubscriber
from spartan.utils.ros_utils import RobotService
import spartan.utils.ros_utils as rosUtils
import spartan.utils.utils as spartan_utils
import robot_control.control_utils as control_utils
import spartan.utils.transformations as transformations
import transform_util as tf_util
from spartan.manipulation.schunk_driver import SchunkDriver
# spartan ROS
import robot_msgs.msg

from ar_track_alvar_msgs.msg import AlvarMarkers
from geometry_msgs.msg import Twist


class ARPoseEst():
	def __init__(self):
		rospy.init_node("ar_pose_est")
		# get initial pose
		self.pose = []
		f = open('./initial_pose2.csv', 'r')
		for line in f:
			for word in line.split(','):
				self.pose.append(float(word))
		print(self.pose)
		rospy.wait_for_message('ar_pose_marker', AlvarMarkers)
		# Subscribe to the ar_pose_marker topic to get the image width and height
		rospy.Subscriber('ar_pose_marker', AlvarMarkers, self.pose_est_call_back)
		self.pub = rospy.Publisher('carrot_pose_pub',std_msgs.msg.Float32MultiArray,queue_size=1)

	def pose_est_call_back(self, msg): 
		try:
			marker = msg.markers[0]
			pos_x = marker.pose.pose.position.x
			pos_y = marker.pose.pose.position.y
			pos_z = marker.pose.pose.position.z
			ori_x = marker.pose.pose.orientation.x
			ori_y = marker.pose.pose.orientation.y
			ori_z = marker.pose.pose.orientation.z
			ori_w = marker.pose.pose.orientation.w
			quat = np.array([ori_w,ori_x,ori_y,ori_z])
			mat = transformations.quaternion_matrix(quat)
			y_axis = mat[:3,1] 
			quat_init = np.array([self.pose[6],self.pose[3],self.pose[4],self.pose[5]])
			mat_init = transformations.quaternion_matrix(quat_init)
			y_axis_init = mat_init[:3,1]
			angle = np.arccos((y_axis.dot(y_axis_init)/(np.linalg.norm(y_axis)*np.linalg.norm(y_axis_init))))
			if np.cross(y_axis_init,y_axis).dot(np.array([0,0,1])) > 0:
				angle = -angle
			print('degree = %.2f'%(angle/np.pi*180))
			new_msg = std_msgs.msg.Float32MultiArray(data=[angle,pos_x])
			self.pub.publish(new_msg)

		except:
			return



if __name__ == '__main__':
    try:
        ARPoseEst()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("AR reader node terminated.")
