import argparse
import time

import numpy as np

import rospy
import actionlib
import robot_msgs.msg
import robot_msgs.srv
import trajectory_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
import std_srvs.srv

import robot_control.control_utils as control_utils
import spartan.utils.ros_utils as ros_utils

def make_cartesian_trajectory_goal_world_frame(pos, quat, duration):

    # (array([0.588497  , 0.00716426, 0.5159925 ]), array([ 0.70852019, -0.15500524,  0.67372875,  0.1416407 ]))


    goal = robot_msgs.msg.CartesianTrajectoryGoal()
    traj = goal.trajectory

    # frame_id = "iiwa_link_ee"
    frame_id = "base"
    ee_frame_id = "iiwa_link_ee"
    
    xyz_knot = geometry_msgs.msg.PointStamped()
    xyz_knot.header.frame_id = frame_id
    xyz_knot.point.x = 0
    xyz_knot.point.y = 0
    xyz_knot.point.z = 0
    traj.xyz_points.append(xyz_knot)

    xyz_knot = geometry_msgs.msg.PointStamped()
    xyz_knot.header.frame_id = frame_id
    xyz_knot.point.x = pos[0]
    xyz_knot.point.y = pos[1]
    xyz_knot.point.z = pos[2]

    traj.xyz_points.append(xyz_knot)

    traj.ee_frame_id = ee_frame_id

    traj.time_from_start.append(rospy.Duration(0.0))
    traj.time_from_start.append(rospy.Duration(duration))

    quat_msg = geometry_msgs.msg.Quaternion()
    quat_msg.w = quat[0]
    quat_msg.x = quat[1]
    quat_msg.y = quat[2]
    quat_msg.z = quat[3]

    traj.quaternions.append(quat_msg)

    return goal

def make_cartesian_gains_msg(kp_rot = 5,kp_trans = 10):
    msg = robot_msgs.msg.CartesianGain()

    
    msg.rotation.x = kp_rot
    msg.rotation.y = kp_rot
    msg.rotation.z = kp_rot

    
    msg.translation.x = kp_trans
    msg.translation.y = kp_trans
    msg.translation.z = kp_trans

    return msg

def test_task_space_streaming():
    rospy.wait_for_service("plan_runner/init_task_space_streaming")
    sp = rospy.ServiceProxy('plan_runner/init_task_space_streaming',
        robot_msgs.srv.StartStreamingPlan)
    init = robot_msgs.srv.StartStreamingPlanRequest()
    #init.force_guard.append(make_force_guard_msg())
    print sp(init)
    pub = rospy.Publisher('plan_runner/task_space_streaming_setpoint',
        robot_msgs.msg.CartesianGoalPoint, queue_size=1)
    robotSubscriber = ros_utils.JointStateSubscriber("/joint_states")
    print("Waiting for full kuka state...")
    while len(robotSubscriber.joint_positions.keys()) < 3:
        rospy.sleep(0.1)
    print("got full state")

    start_time = time.time()
    new_msg = robot_msgs.msg.CartesianGoalPoint()

    pos = [0.63, 0.0, 0.15]
    quat = [ 0.71390524,  0.12793277,  0.67664775, -0.12696589] # straight down position
    '''
    goal = make_cartesian_trajectory_goal_world_frame(
        pos,
        quat,
        duration = 5.)
    '''
    #for i in range(10):
    new_msg.xyz_point.header.frame_id = "iiwa_link_ee"
    new_msg.xyz_point.point.x = 0 #pos[0]
    new_msg.xyz_point.point.y = 0.01 #pos[1]
    new_msg.xyz_point.point.z = 0 #pos[2]
    new_msg.xyz_d_point.x = 0.
    new_msg.xyz_d_point.y = 0.
    new_msg.xyz_d_point.z = 0.0
    new_msg.quaternion.w = 1 #quat[0]
    new_msg.quaternion.x = 0 #quat[1]
    new_msg.quaternion.y = 0 #quat[2]
    new_msg.quaternion.z = 0 #quat[3]
    new_msg.gain = make_cartesian_gains_msg(50,10)
    new_msg.ee_frame_id = "iiwa_link_ee"
    
    while (time.time() - start_time < 10):
        pub.publish(new_msg)
        time.sleep(0.001)
        #print "goal",i
        print time.time() - start_time 
    start_time = time.time()
    #time.sleep(3)
    rospy.wait_for_service("plan_runner/stop_plan")
    sp = rospy.ServiceProxy('plan_runner/stop_plan',
        std_srvs.srv.Trigger)
    init = std_srvs.srv.TriggerRequest()
    print sp(init)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--movement", type=str,
        help="(optional) type of movement, can be gripper_frame or world_frame", default="gripper_frame")
    rospy.init_node("test_plan_runner")
    args = parser.parse_args()
    test_task_space_streaming()