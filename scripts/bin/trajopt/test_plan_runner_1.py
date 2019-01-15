import argparse
import time

import numpy as np

import rospy
import tf2_ros
import tf
import actionlib
import robot_msgs.msg
import robot_msgs.srv
import trajectory_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
import std_srvs.srv
import spartan.utils.transformations as transformations
from pyquaternion import Quaternion
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

def tf_matrix_from_pose(pose):
    trans, quat = pose
    mat = transformations.quaternion_matrix(quat)
    mat[:3, 3] = trans
    return mat

def get_relative_tf_between_poses(pose_1, pose_2):
    tf_1 = tf_matrix_from_pose(pose_1)
    tf_2 = tf_matrix_from_pose(pose_2)
    return np.linalg.inv(tf_1).dot(tf_2)

def test_task_space_streaming():
    #rospy.init_node('sandbox', anonymous=True)
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

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
    quat_interpolation_number = 1000
    quat_interpolation_array = np.linspace(2.0, 3.0, num=quat_interpolation_number)
    quat_interpolation_cnt = 0
    while (time.time() - start_time < 1):

        pos = [0.63, 0.0, 0.15]
        quat = [ 0.71390524,  0.12793277,  0.67664775, -0.12696589] # straight down position

        pose_2 = tuple((pos,quat))
        frame_name = "iiwa_link_ee"
        try:
            current_pose_ee = ros_utils.poseFromROSTransformMsg(
                tfBuffer.lookup_transform("base", frame_name, rospy.Time()).transform)
            print current_pose_ee
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("Troubling looking up tf...")
            rate.sleep()

        rel_pose = get_relative_tf_between_poses(current_pose_ee,pose_2)
        rel_q = transformations.quaternion_from_matrix(rel_pose)
        rel_trans = rel_pose[:3,3]
        rel_trans = np.clip(rel_trans,-0.01, 0.01)
        #print tuple((rel_trans,rel_q))
        q0 = Quaternion(quat)
        q1 = Quaternion(current_pose_ee[1])
        q = Quaternion.slerp(q0,q1,quat_interpolation_array[quat_interpolation_cnt])
        #quat_interpolation_cnt += 1
        #quat_interpolation_cnt = min(quat_interpolation_cnt,quat_interpolation_number-1)
        #for i in range(10):
        new_msg.xyz_point.header.frame_id = "base"
        new_msg.xyz_point.point.x = current_pose_ee[0][0]+rel_trans[0]
        new_msg.xyz_point.point.y = current_pose_ee[0][1]+rel_trans[1]
        new_msg.xyz_point.point.z = current_pose_ee[0][2]+rel_trans[2]
        new_msg.xyz_d_point.x = 0.
        new_msg.xyz_d_point.y = 0.
        new_msg.xyz_d_point.z = 0.0
        new_msg.quaternion.w = q[0] #rel_q[0]
        new_msg.quaternion.x = q[1] #rel_q[1]
        new_msg.quaternion.y = q[2] #rel_q[2]
        new_msg.quaternion.z = q[3] #rel_q[3]
        new_msg.gain = make_cartesian_gains_msg(50,10)
        new_msg.ee_frame_id = "iiwa_link_ee"
    
        pub.publish(new_msg)
        time.sleep(0.1)
        #print "goal",i
        #print time.time() - start_time 
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