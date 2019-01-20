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
import spartan.utils.transformations as transformations
import transform_util as tf_util
from spartan.manipulation.schunk_driver import SchunkDriver
# spartan ROS
import robot_msgs.msg

from ar_track_alvar_msgs.msg import AlvarMarkers
from geometry_msgs.msg import Twist


class ARReader():
	def __init__(self):
		rospy.init_node("ar_reader")
		self.status = 0
		rospy.wait_for_message('ar_pose_marker', AlvarMarkers)
		# Subscribe to the ar_pose_marker topic to get the image width and height
		rospy.Subscriber('ar_pose_marker', AlvarMarkers, self.reader_call_back)

	def reader_call_back(self, msg): 
		if self.status == 0:
			try:
				marker = msg.markers[0]
				pos_x = marker.pose.pose.position.x
				pos_y = marker.pose.pose.position.y
				pos_z = marker.pose.pose.position.z
				ori_x = marker.pose.pose.orientation.x
				ori_y = marker.pose.pose.orientation.y
				ori_z = marker.pose.pose.orientation.z
				ori_w = marker.pose.pose.orientation.w
				print('recording initial pose\n')
				f = open('./initial_pose.csv', 'w')
				f.write('%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n'%(pos_x,pos_y,pos_z,ori_x,ori_y,ori_z,ori_w))
				f.close()
				self.status = 1
				print('finish recording initial pose\n')
			except:
				return



if __name__ == '__main__':
    try:
        ARReader()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("AR reader node terminated.")
