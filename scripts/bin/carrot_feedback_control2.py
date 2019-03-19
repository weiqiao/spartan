import unittest
import subprocess
import psutil
import sys
import os
import numpy as np
import time
import socket
import pickle

# ROS
import rospy
import actionlib
import sensor_msgs.msg
import geometry_msgs.msg
import trajectory_msgs.msg
import rosgraph
import tf2_ros
import std_srvs.srv
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
import robot_msgs.srv


data = [0.0,0.0]
file_name = "trajopt_example11_latest"
# file_name = "trajopt_example9_latest"
# file_name = "trajopt_example11_latest_artificial_data_output"

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

def make_cartesian_trajectory_goal_gripper_frame(pos, quat, duration):
    # (array([0.588497  , 0.00716426, 0.5159925 ]), array([ 0.70852019, -0.15500524,  0.67372875,  0.1416407 ]))

    # Barely touching tabletop

    goal = robot_msgs.msg.CartesianTrajectoryGoal()
    traj = goal.trajectory

    # frame_id = "iiwa_link_ee"
    frame_id = "iiwa_link_ee"
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

def make_cartesian_gains_msg(kp_rot, kp_trans):
    msg = robot_msgs.msg.CartesianGain()

    msg.rotation.x = kp_rot
    msg.rotation.y = kp_rot
    msg.rotation.z = kp_rot

    msg.translation.x = kp_trans
    msg.translation.y = kp_trans
    msg.translation.z = kp_trans

    return msg

def make_force_guard_msg(scale):
    msg = robot_msgs.msg.ForceGuard()
    external_force = robot_msgs.msg.ExternalForceGuard()

    body_frame = "iiwa_link_ee"
    expressed_in_frame = "iiwa_link_ee"
    force_vec = scale*np.array([-1,0,0])

    external_force.force.header.frame_id = expressed_in_frame
    external_force.body_frame = body_frame
    external_force.force.vector.x = force_vec[0]
    external_force.force.vector.y = force_vec[1]
    external_force.force.vector.z = force_vec[2]

    msg.external_force_guards.append(external_force)

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

def safety_check(pose_cur,phi,alpha,l=0.16):
    safe = 1
    if phi >= np.pi*3.0/4.0 or phi <= np.pi/4.0:
        safe = 0
        print("phi out of range: phi=",phi)
        return safe 
    z_max = pose_cur[0][2] + 0.05 # TODO
    alpha += 0.015 # the width of a finger
    if phi <= np.pi/2:
        z = alpha*np.cos(phi) + l*np.sin(phi)
    else:
        z = - alpha*np.cos(phi) + l*np.sin(phi)
    if z >= z_max:
        print("z is larger than z_max: z=",z,"z_max=",z_max)
        safe = 0
    return safe 

def translate_gripper_yz(pose_cur,dx=[0.0,0.0]):
    trans, quat = pose_cur
    trans_new = trans
    dx_norm = np.sqrt(dx[0]*dx[0]+dx[1]*dx[1])
    if dx_norm > 0.02:
        dx[0] *= 0.02/dx_norm
        dx[1] *= 0.02/dx_norm
    trans_new[1] += dx[0]
    trans_new[2] += dx[1]

    pose_new = [trans_new,quat]

    new_msg = robot_msgs.msg.CartesianGoalPoint()
    new_msg.xyz_point.header.frame_id = "base"
    new_msg.xyz_point.point.x = pose_new[0][0]
    new_msg.xyz_point.point.y = pose_new[0][1]
    new_msg.xyz_point.point.z = pose_new[0][2]
    new_msg.xyz_d_point.x = 0.
    new_msg.xyz_d_point.y = 0.
    new_msg.xyz_d_point.z = 0.0
    new_msg.quaternion.w = pose_new[1][0]
    new_msg.quaternion.x = pose_new[1][1]
    new_msg.quaternion.y = pose_new[1][2]
    new_msg.quaternion.z = pose_new[1][3]
    new_msg.gain = make_cartesian_gains_msg(50,10)
    new_msg.ee_frame_id = "iiwa_link_ee"

    return new_msg, pose_new

def rotate_around_left_finger_tip(pose_cur,dphi,phi,alpha=0.05,dx=[0.0,0.0],l=0.16):
    # rotate around left finger tip for dphi degrees
    # pose_cur = [[trans],[quat]]: current pose
    # phi: current degree of gripper
    # dx: horizontal and vertical translation of the left finger tip
    # alpha: distance between to fingers
    # l: height between left finger tip and the ee frame origin 

    # for safety, restrict length of dx
    dx_norm = np.sqrt(dx[0]*dx[0]+dx[1]*dx[1])
    if dx_norm > 0.02:
        dx[0] *= 0.02/dx_norm
        dx[1] *= 0.02/dx_norm

    L = np.sqrt(alpha**2 + l**2)
    dy = L*np.cos(phi+dphi) - L*np.cos(phi)
    dz = L*np.sin(phi+dphi) - L*np.sin(phi)

    trans, quat = pose_cur
    trans_new = trans
    trans_new[1] += dy + dx[0]
    trans_new[2] += dz + dx[1]
    trans_new[2] += 0.001*np.abs(dphi)/(np.pi/180)

    # rotate phi degrees counter-clockwise
    mat = transformations.quaternion_matrix(quat)
    mat = mat[:3,:3]
    rot_axis = np.array([1,0,0])
    rot_theta = dphi
    rot_mat = tf_util.axis_angle_to_rotation_matrix(rot_axis, rot_theta)
    mat = rot_mat.dot(mat)
    quat = transformations.quaternion_from_matrix(mat)
    
    pose_new = [trans_new,quat.tolist()]

    new_msg = robot_msgs.msg.CartesianGoalPoint()
    new_msg.xyz_point.header.frame_id = "base"
    new_msg.xyz_point.point.x = pose_new[0][0]
    new_msg.xyz_point.point.y = pose_new[0][1]
    new_msg.xyz_point.point.z = pose_new[0][2]
    new_msg.xyz_d_point.x = 0.
    new_msg.xyz_d_point.y = 0.
    new_msg.xyz_d_point.z = 0.0
    new_msg.quaternion.w = pose_new[1][0]
    new_msg.quaternion.x = pose_new[1][1]
    new_msg.quaternion.y = pose_new[1][2]
    new_msg.quaternion.z = pose_new[1][3]
    new_msg.gain = make_cartesian_gains_msg(50,10)
    new_msg.ee_frame_id = "iiwa_link_ee"

    return new_msg, pose_new, phi + dphi 

def rotate_around_left_finger_tip2(pose_cur,dphi,phi,alpha=0.05,dx=[0.0,0.0],l=0.16):
    # rotate around left finger tip for dphi degrees
    # pose_cur = [[trans],[quat]]: current pose
    # phi: current degree of gripper
    # dx: horizontal and vertical translation of the left finger tip
    # alpha: distance between to fingers
    # l: height between left finger tip and the ee frame origin 

    # for safety, restrict length of dx
    dx_norm = np.sqrt(dx[0]*dx[0]+dx[1]*dx[1])
    if dx_norm > 0.02:
        dx[0] *= 0.02/dx_norm
        dx[1] *= 0.02/dx_norm

    L = np.sqrt(alpha**2 + l**2)
    dy = L*np.cos(phi+dphi) - L*np.cos(phi)
    dz = L*np.sin(phi+dphi) - L*np.sin(phi)

    trans, quat = pose_cur
    trans_new = trans
    trans_new[1] += dy + dx[0]
    trans_new[2] += dz + dx[1]
    trans_new[2] += -0.001*np.abs(dphi)/(np.pi/180)

    # rotate phi degrees counter-clockwise
    mat = transformations.quaternion_matrix(quat)
    mat = mat[:3,:3]
    rot_axis = np.array([1,0,0])
    rot_theta = dphi
    rot_mat = tf_util.axis_angle_to_rotation_matrix(rot_axis, rot_theta)
    mat = rot_mat.dot(mat)
    quat = transformations.quaternion_from_matrix(mat)
    
    pose_new = [trans_new,quat.tolist()]

    new_msg = robot_msgs.msg.CartesianGoalPoint()
    new_msg.xyz_point.header.frame_id = "base"
    new_msg.xyz_point.point.x = pose_new[0][0]
    new_msg.xyz_point.point.y = pose_new[0][1]
    new_msg.xyz_point.point.z = pose_new[0][2]
    new_msg.xyz_d_point.x = 0.
    new_msg.xyz_d_point.y = 0.
    new_msg.xyz_d_point.z = 0.0
    new_msg.quaternion.w = pose_new[1][0]
    new_msg.quaternion.x = pose_new[1][1]
    new_msg.quaternion.y = pose_new[1][2]
    new_msg.quaternion.z = pose_new[1][3]
    new_msg.gain = make_cartesian_gains_msg(50,10)
    new_msg.ee_frame_id = "iiwa_link_ee"

    return new_msg, pose_new, phi + dphi 


def change_finger_distance_while_keeping_left_finger_tip_unmoved(pose_cur,phi,dalpha,dx=[0.0,0.0]):
    trans, quat = pose_cur
    trans_new = trans
    dy = dalpha*np.sin(phi)
    dz = -dalpha*np.cos(phi)
    trans_new[1] += dy + dx[0]
    trans_new[2] += dz + dx[1]

    pose_new = [trans_new,quat]

    new_msg = robot_msgs.msg.CartesianGoalPoint()
    new_msg.xyz_point.header.frame_id = "base"
    new_msg.xyz_point.point.x = pose_new[0][0]
    new_msg.xyz_point.point.y = pose_new[0][1]
    new_msg.xyz_point.point.z = pose_new[0][2]
    new_msg.xyz_d_point.x = 0.
    new_msg.xyz_d_point.y = 0.
    new_msg.xyz_d_point.z = 0.0
    new_msg.quaternion.w = pose_new[1][0]
    new_msg.quaternion.x = pose_new[1][1]
    new_msg.quaternion.y = pose_new[1][2]
    new_msg.quaternion.z = pose_new[1][3]
    new_msg.gain = make_cartesian_gains_msg(50,10)
    new_msg.ee_frame_id = "iiwa_link_ee"

    return new_msg, pose_new

def prepose_for_closed_loop():
    #pos = [0.63, 0., 0.164] # this is for example 8 trajectory
    #pos = [0.63, 0., 0.162] # this is for example 9 trajectory, before breaking gripper
    #pos = [0.63, 0., 0.24] # just to be safe
    #pos = [0.63, 0., 0.164] # this is for example 9 trajectory -- open loop works
    #pos = [0.63, 0., 0.166] # this is for example 9 trajectory (right finger almost touches table)
    #pos = [0.63, 0., 0.170] # added a bench: pass safety check
    #pos = [0.63, 0., 0.176] # added a bench: pass safety check


    #pos = [0.63, 0., 0.17] # added a bench: pass safety check -- traj 11 open loop works (with red cutting board)
    pos = [0.63, 0., 0.18] # added a bench: pass safety check -- traj 11 open loop works (with red cutting board)
    quat = [ 0.71390524,  0.12793277,  0.67664775, -0.12696589] # straight down position
    
    # # rotate -pi/4 degrees
    # mat = transformations.quaternion_matrix(quat)
    # mat = mat[:3,:3]
    # rot_axis = np.array([0,1,0])
    # rot_theta = -np.pi/4
    # rot_mat = tf_util.axis_angle_to_rotation_matrix(rot_axis, rot_theta)
    # mat = rot_mat.dot(mat)
    # quat = transformations.quaternion_from_matrix(mat)
    # print quat
    # make goal
    goal = make_cartesian_trajectory_goal_world_frame(
        pos,
        quat,
        duration = 5.)
    goal.gains.append(make_cartesian_gains_msg(50., 10.))
    goal.force_guard.append(make_force_guard_msg(15.))
    return goal

def postpose_for_closed_loop():
    #pos = [0.63, 0., 0.164] # this is for example 8 trajectory
    #pos = [0.63, 0., 0.162] # this is for example 9 trajectory, before breaking gripper
    #pos = [0.63, 0., 0.24] # just to be safe
    #pos = [0.63, 0., 0.164] # this is for example 9 trajectory -- open loop works
    #pos = [0.63, 0., 0.166] # this is for example 9 trajectory (right finger almost touches table)
    #pos = [0.63, 0., 0.170] # added a bench: pass safety check
    #pos = [0.63, 0., 0.176] # added a bench: pass safety check
    

    pos = [0.63, 0., 0.135] # added a bench: pass safety check
    #pos = [0.63, 0., 0.125] # added a bench: pass safety check
    #pos = [0.63, -0.17, 0.12] # added a bench: pass safety check
    quat = [ 0.71390524,  0.12793277,  0.67664775, -0.12696589] # straight down position
    
    # rotate pi/3 degrees
    rot_theta = np.pi/5
    mat = transformations.quaternion_matrix(quat)
    mat = mat[:3,:3]
    rot_axis = np.array([1,0,0])
    rot_mat = tf_util.axis_angle_to_rotation_matrix(rot_axis, rot_theta)
    mat = rot_mat.dot(mat)
    quat = transformations.quaternion_from_matrix(mat)

    goal = make_cartesian_trajectory_goal_world_frame(
        pos,
        quat,
        duration = 5.)
    goal.gains.append(make_cartesian_gains_msg(50., 10.))
    goal.force_guard.append(make_force_guard_msg(15.))
    return goal

def postpose_for_closed_loop2():
    #pos = [0.63, 0., 0.164] # this is for example 8 trajectory
    #pos = [0.63, 0., 0.162] # this is for example 9 trajectory, before breaking gripper
    #pos = [0.63, 0., 0.24] # just to be safe
    #pos = [0.63, 0., 0.164] # this is for example 9 trajectory -- open loop works
    #pos = [0.63, 0., 0.166] # this is for example 9 trajectory (right finger almost touches table)
    #pos = [0.63, 0., 0.170] # added a bench: pass safety check
    #pos = [0.63, 0., 0.176] # added a bench: pass safety check
    

    #pos = [0.63, 0., 0.19] # added a bench: pass safety check
    pos = [0.63, -0.2, 0.135] # added a bench: pass safety check
    quat = [ 0.71390524,  0.12793277,  0.67664775, -0.12696589] # straight down position
    
    # rotate pi/3 degrees
    rot_theta = np.pi/5
    mat = transformations.quaternion_matrix(quat)
    mat = mat[:3,:3]
    rot_axis = np.array([1,0,0])
    rot_mat = tf_util.axis_angle_to_rotation_matrix(rot_axis, rot_theta)
    mat = rot_mat.dot(mat)
    quat = transformations.quaternion_from_matrix(mat)

    goal = make_cartesian_trajectory_goal_world_frame(
        pos,
        quat,
        duration = 5.)
    goal.gains.append(make_cartesian_gains_msg(50., 10.))
    goal.force_guard.append(make_force_guard_msg(15.))
    return goal

def carrot_closed_loop_initial_pose():
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rospy.wait_for_service("plan_runner/init_task_space_streaming")
    sp = rospy.ServiceProxy('plan_runner/init_task_space_streaming', robot_msgs.srv.StartStreamingPlan)
    init = robot_msgs.srv.StartStreamingPlanRequest()
    rospy.sleep(1)
    #init.force_guard.append(make_force_guard_msg())
    print sp(init)
    pub = rospy.Publisher('plan_runner/task_space_streaming_setpoint', robot_msgs.msg.CartesianGoalPoint, queue_size=1)
    robotSubscriber = rosUtils.JointStateSubscriber("/joint_states")
    print("Waiting for full kuka state...")
    while len(robotSubscriber.joint_positions.keys()) < 3:
        rospy.sleep(0.1)
    print("got full state")

    new_msg = robot_msgs.msg.CartesianGoalPoint()
    frame_name = "iiwa_link_ee"
    try:
        current_pose_ee = rosUtils.poseFromROSTransformMsg(tfBuffer.lookup_transform("base", frame_name, rospy.Time()).transform)
        print current_pose_ee
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        print("Troubling looking up tf...")
        rate.sleep()

    # load offline controller
    state_and_control = pickle.load(open(file_name + ".p","rb"))
    pos_over_time = state_and_control["state"]
    F_over_time = state_and_control["control"]
    params = state_and_control["params"]

    phi = np.pi/2
    alpha = 0.05
    idx = int(params[0])
    T = int(params[idx+27])
    r0 = params[3]
    r = params[4]
    omega_offset = 0.008

    # go to initial state
    t = 0
    x,y,theta,x_dot,y_dot,theta_dot,d = pos_over_time[t,:]
    F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi0, omega = F_over_time[t,:]
    x_centroid = x - r0*np.sin(theta)
    y_centroid = y + r0*np.cos(theta)
    x_F1 = x_centroid-d*np.cos(theta)
    y_F1 = y_centroid-d*np.sin(theta)

    frame_name = "iiwa_link_ee"
    # alpha    
    omega += omega_offset
    dalpha = omega / 2.0 - alpha 
    print "dalpha = ", dalpha
    try:
        current_pose_ee = rosUtils.poseFromROSTransformMsg(tfBuffer.lookup_transform("base", frame_name, rospy.Time()).transform)
        #print current_pose_ee
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        print("Troubling looking up tf...")
        rate.sleep()
    new_msg, pose_new = change_finger_distance_while_keeping_left_finger_tip_unmoved(current_pose_ee,phi,dalpha)
    safe1 = safety_check(pose_new,phi,alpha)
    safe2 = safety_check(pose_new,phi,alpha+dalpha)
    if safe1 and safe2:
        start_time = time.time()
        while (time.time() - start_time < 2):
            pub.publish(new_msg)
            handDriver.sendGripperCommand(omega, force=80, speed=0.05)
            time.sleep(.1)
        alpha += dalpha
    else:
        print("unsafe initial alpha",safe1,safe2)

        rospy.wait_for_service("plan_runner/stop_plan")
        sp = rospy.ServiceProxy('plan_runner/stop_plan',std_srvs.srv.Trigger)
        init = std_srvs.srv.TriggerRequest()
        print sp(init)
        return 

    # phi
    while (np.abs(phi0-phi)>np.pi/180.0*0.1):
        try:
            current_pose_ee = rosUtils.poseFromROSTransformMsg(tfBuffer.lookup_transform("base", frame_name, rospy.Time()).transform)
            #print current_pose_ee
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("Troubling looking up tf...")
            rate.sleep()
        dphi = phi0 - phi 
        dphi = min(dphi,np.pi/180.0*2.0)
        dphi = max(dphi,-np.pi/180.0*2.0)
        print "dphi=",dphi
        new_msg, pose_new, phi = rotate_around_left_finger_tip2(current_pose_ee,dphi,phi,alpha)
        safe = safety_check(pose_new,phi,alpha)
        if safe:
            start_time = time.time()
            while (time.time() - start_time < 0.5):        
                pub.publish(new_msg)
                time.sleep(0.1)
        else:
            print("unsafe initial phi",safe)
            phi -= dphi 

            rospy.wait_for_service("plan_runner/stop_plan")
            sp = rospy.ServiceProxy('plan_runner/stop_plan',std_srvs.srv.Trigger)
            init = std_srvs.srv.TriggerRequest()
            print sp(init)
            return 

def carrot_closed_loop():
    #rospy.init_node('sandbox', anonymous=True)
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rospy.wait_for_service("plan_runner/init_task_space_streaming")
    sp = rospy.ServiceProxy('plan_runner/init_task_space_streaming', robot_msgs.srv.StartStreamingPlan)
    init = robot_msgs.srv.StartStreamingPlanRequest()
    rospy.sleep(1)
    #init.force_guard.append(make_force_guard_msg())
    print sp(init)
    pub = rospy.Publisher('plan_runner/task_space_streaming_setpoint', robot_msgs.msg.CartesianGoalPoint, queue_size=1)
    robotSubscriber = rosUtils.JointStateSubscriber("/joint_states")
    print("Waiting for full kuka state...")
    while len(robotSubscriber.joint_positions.keys()) < 3:
        rospy.sleep(0.1)
    print("got full state")

    # new_msg = robot_msgs.msg.CartesianGoalPoint()
    frame_name = "iiwa_link_ee"
    try:
        current_pose_ee = rosUtils.poseFromROSTransformMsg(tfBuffer.lookup_transform("base", frame_name, rospy.Time()).transform)
        print current_pose_ee
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        print("Troubling looking up tf...")
        rate.sleep()

    # load offline controller
    state_and_control = pickle.load(open(file_name + ".p","rb"))
    pos_over_time = state_and_control["state"]
    F_over_time = state_and_control["control"]
    params = state_and_control["params"]

    phi = np.pi/2
    alpha = 0.05
    idx = int(params[0])
    T = int(params[idx+27])
    r0 = params[3]
    r = params[4]
    omega_offset = 0.008 # from experiment. previously was 0.008

    # go to initial state
    t = 0
    x,y,theta,x_dot,y_dot,theta_dot,d = pos_over_time[t,:]
    F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi0, omega = F_over_time[t,:]
    x_centroid = x - r0*np.sin(theta)
    y_centroid = y + r0*np.cos(theta)
    x_F1 = x_centroid-d*np.cos(theta)
    y_F1 = y_centroid-d*np.sin(theta)

    # frame_name = "iiwa_link_ee"
    # alpha    
    omega += omega_offset
    dalpha = omega / 2.0 - alpha 
    print "dalpha = ", dalpha
    new_msg, pose_new = change_finger_distance_while_keeping_left_finger_tip_unmoved(current_pose_ee,phi,dalpha)
    safe1 = safety_check(pose_new,phi,alpha)
    safe2 = safety_check(pose_new,phi,alpha+dalpha)
    if safe1 and safe2:
        start_time = time.time()
        while (time.time() - start_time < 2):
            pub.publish(new_msg)
            handDriver.sendGripperCommand(omega, force=80, speed=0.05)
            time.sleep(.1)
        alpha += dalpha
    else:
        print("unsafe initial alpha",safe1,safe2)

        rospy.wait_for_service("plan_runner/stop_plan")
        sp = rospy.ServiceProxy('plan_runner/stop_plan',std_srvs.srv.Trigger)
        init = std_srvs.srv.TriggerRequest()
        print sp(init)
        return 

    # phi
    while (np.abs(phi0-phi)>np.pi/180.0*0.1):
        current_pose_ee = pose_new 
        dphi = phi0 - phi 
        dphi = min(dphi,np.pi/180.0*2.0)
        dphi = max(dphi,-np.pi/180.0*2.0)
        print "dphi=",dphi
        new_msg, pose_new, phi = rotate_around_left_finger_tip2(current_pose_ee,dphi,phi,alpha)
        safe = safety_check(pose_new,phi,alpha)
        if safe:
            start_time = time.time()
            while (time.time() - start_time < 0.5):        
                pub.publish(new_msg)
                time.sleep(0.1)
        else:
            print("unsafe initial phi",safe)
            phi -= dphi 

            rospy.wait_for_service("plan_runner/stop_plan")
            sp = rospy.ServiceProxy('plan_runner/stop_plan',std_srvs.srv.Trigger)
            init = std_srvs.srv.TriggerRequest()
            print sp(init)
            return 

    # Finish initial pose. Start control...

    rospy.sleep(20)
    print("10 seconds left")
    rospy.sleep(10)
    print("5 seconds left")
    rospy.sleep(5)
    print("start...")

    gripper_state = [x_F1,y_F1,phi,alpha]
    dx_cur = 0 # if the initial state is perturbed this is not the case
    #dx_cur_array = np.zeros(T)
    # output = pickle.load(open("trajopt_example11_latest_artificial_data_output" + ".p","rb"))
    # ttt = 0
    while (t < T-1):
        # # compute current state from data
        # # data = [pos_over_time[t,2],pos_over_time[t,0]]
        # # data = output["data"][ttt,:]
        # # ttt += 1
        theta_cur, x_cur = data 
        tol = np.pi/180.0*5
        theta_diff = np.abs(theta_cur - pos_over_time[t,2])
        if theta_diff < tol:
            print("normal t = ",t)
        else:
            t_new = t 
            for t0 in range(T-1):
                if theta_diff > np.abs(pos_over_time[t0,2] - theta_cur):
                    t_new = t0
                    theta_diff = np.abs(pos_over_time[t0,2] - theta_cur)
            if t_new != t:
                print("t changes from",t,"to",t_new)
                t = t_new
                continue 

        x = pos_over_time[t,0]
        dx_cur = x_cur - x # compensation for deviation in x
        print("deviation in x is %.2f"%dx_cur) 
        #dx_cur_array[t] = dx_cur 

        t += 1

        x,y,theta,x_dot,y_dot,theta_dot,d = pos_over_time[t,:]
        F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi_next, omega = F_over_time[t,:]
        x_centroid = x - r0*np.sin(theta)
        y_centroid = y + r0*np.cos(theta)
        x_F1_next = x_centroid-d*np.cos(theta)
        y_F1_next = y_centroid-d*np.sin(theta)
        # x_F1_next += dx_cur 

        x_F1, y_F1, phi, alpha = gripper_state

        dx = x_F1_next - x_F1 
        dy = y_F1_next - y_F1 

        # phi, dy
        
        current_pose_ee = pose_new 

        dphi = phi_next - phi 
        omega += omega_offset
        dalpha = omega / 2.0 - alpha 

        print "dphi=",dphi/np.pi * 180.0, "dalpha=",dalpha, "dx=",dx,"dy=",dy
        # do while loop or do this?? 
        dphi = min(dphi,np.pi/180.0*2.0)
        dphi = max(dphi,-np.pi/180.0*2.0)
        if dphi > 0:
            new_msg, pose_new, phi = rotate_around_left_finger_tip(current_pose_ee,dphi,phi,alpha,[dx,dy])
        else:
            new_msg, pose_new, phi = rotate_around_left_finger_tip2(current_pose_ee,dphi,phi,alpha,[dx,dy])
        safe = safety_check(pose_new,phi,alpha)
        if safe:
            start_time = time.time()
            while (time.time() - start_time < 0.5):        
                pub.publish(new_msg)
                time.sleep(0.1)
        else:
            print("unsafe phi, t = %.2f",t)
            phi -= dphi 

            rospy.wait_for_service("plan_runner/stop_plan")
            sp = rospy.ServiceProxy('plan_runner/stop_plan',std_srvs.srv.Trigger)
            init = std_srvs.srv.TriggerRequest()
            print sp(init)
            return 

        # alpha    
        current_pose_ee = pose_new
        new_msg, pose_new = change_finger_distance_while_keeping_left_finger_tip_unmoved(current_pose_ee,phi,dalpha)
        safe1 = safety_check(pose_new,phi,alpha)
        safe2 = safety_check(pose_new,phi,alpha+dalpha)
        if safe1 and safe2:
            start_time = time.time()
            while (time.time() - start_time < 1):
                pub.publish(new_msg)
                handDriver.sendGripperCommand(omega, force=80, speed=0.05)
                time.sleep(1)
            alpha += dalpha
        else:
            print("unsafe alpha, t=",t,safe1,safe2)

            rospy.wait_for_service("plan_runner/stop_plan")
            sp = rospy.ServiceProxy('plan_runner/stop_plan',std_srvs.srv.Trigger)
            init = std_srvs.srv.TriggerRequest()
            print sp(init)
            return 

        gripper_state[0] += dx 
        gripper_state[1] += dy
        gripper_state[2] = phi
        gripper_state[3] = alpha 

    #output = {"dx_cur_array":dx_cur_array}
    #pickle.dump( output, open(file_name+"_dx.p","wb"))

    rospy.wait_for_service("plan_runner/stop_plan")
    sp = rospy.ServiceProxy('plan_runner/stop_plan',std_srvs.srv.Trigger)
    init = std_srvs.srv.TriggerRequest()
    print sp(init)


def pregrasp():
    pos = [0.63, 0., 0.165]
    quat = [ 0.71390524,  0.12793277,  0.67664775, -0.12696589] # straight down position
    # load offline controller
    state_and_control = pickle.load(open(file_name + ".p","rb"))
    F_over_time = state_and_control["control"]

    phi = np.pi/2.0
    phi_next = F_over_time[0,-2]
    dphi = phi_next - phi  
    print "phi=",phi,"phi_next=",phi_next,"dphi=",dphi
    # rotate phi degrees counter-clockwise
    mat = transformations.quaternion_matrix(quat)
    mat = mat[:3,:3]
    rot_axis = np.array([1,0,0])
    rot_theta = dphi
    rot_mat = tf_util.axis_angle_to_rotation_matrix(rot_axis, rot_theta)
    mat = rot_mat.dot(mat)
    quat = transformations.quaternion_from_matrix(mat)
    
    # # rotate -pi/4 degrees
    # mat = transformations.quaternion_matrix(quat)
    # mat = mat[:3,:3]
    # rot_axis = np.array([0,1,0])
    # rot_theta = -np.pi/4
    # rot_mat = tf_util.axis_angle_to_rotation_matrix(rot_axis, rot_theta)
    # mat = rot_mat.dot(mat)
    # quat = transformations.quaternion_from_matrix(mat)

    goal = make_cartesian_trajectory_goal_world_frame(
        pos,
        quat,
        duration = 5.)
    goal.gains.append(make_cartesian_gains_msg(50., 10.))
    goal.force_guard.append(make_force_guard_msg(15.))
    return goal

def carrot_pose_call_back(msg): 
    try:
        data[0] = msg.data[0]
        data[1] = msg.data[1]
        #print data[0], data[1]
    except:
        print "no data"
        return

if __name__ == "__main__":
    rospy.init_node('sandboxx')
    
    rospy.Subscriber('carrot_pose_pub', std_msgs.msg.Float32MultiArray, carrot_pose_call_back)

    robotSubscriber = JointStateSubscriber("/joint_states")
    rospy.sleep(1.0)
    
    print("Moving to start position")

    above_table_pre_grasp = [0.04486168762069299, 0.3256606458812486, -0.033502080520812445, -1.5769091802934694, 0.05899249087322813, 1.246379583616529, 0.38912999977004026]
    targetPosition = above_table_pre_grasp

    robotService = rosUtils.RobotService.makeKukaRobotService()
    success = robotService.moveToJointPosition(targetPosition, timeout=5)

    while len(robotSubscriber.joint_positions.keys()) < 3:
        rospy.sleep(0.1)
    print("Got full state, starting control")

    # EE CONTROL VERSION
    client = actionlib.SimpleActionClient("plan_runner/CartesianTrajectory", robot_msgs.msg.CartesianTrajectoryAction)
    print "waiting for EE action server"
    client.wait_for_server()
    print "connected to EE action server"

    handDriver = SchunkDriver()
    rospy.sleep(1)
    handDriver.sendOpenGripperCommand()
    rospy.sleep(1)


    # gripper_goal_pos = 0.02
    # handDriver.sendGripperCommand(gripper_goal_pos, speed=0.1, timeout=0.01)
    # rospy.sleep(1)
    # handDriver.sendCloseGripperCommand()
    # rospy.sleep(1)

    for goal in [prepose_for_closed_loop()]:
        print "sending goal"
        client.send_goal(goal)
        rospy.loginfo("waiting for CartesianTrajectory action result")
        client.wait_for_result()
        result = client.get_result()
    # carrot_closed_loop_initial_pose()

    carrot_closed_loop()
    
    ### this part is to grab the carrot
    # rospy.sleep(3)
    # handDriver.sendGripperCommand(0.042, force=80, speed=0.05)
    # rospy.sleep(1)

    ### final movement to flip the carrot another 90 degrees
    i = 0
    for goal in [postpose_for_closed_loop(),
                    postpose_for_closed_loop2()]:
        print "sending goal"
        client.send_goal(goal)
        rospy.loginfo("waiting for CartesianTrajectory action result")
        client.wait_for_result()
        result = client.get_result()
        if i==0:
            i+=1
            handDriver.sendGripperCommand(0.075, force=80, speed=0.05)


