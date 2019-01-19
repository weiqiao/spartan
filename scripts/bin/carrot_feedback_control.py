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

def rotate_around_left_finger_tip(pose_cur,dphi,phi,alpha=0.05,dx=[0.0,0.0],l=0.19):
    # rotate around left finger tip for dphi degrees
    # pose_cur = [[trans],[quat]]: current pose
    # phi: current degree of gripper
    # dx: horizontal and vertical translation of the left finger tip
    # alpha: distance between to fingers
    # l: height between left finger tip and the ee frame origin 
    L = np.sqrt(alpha**2 + l**2)
    #print "L=",L
    eta = np.arctan2(alpha,l)
    eta2 = phi-eta 
    eta2_new = eta2 + dphi
    #print "eta = ", eta, "eta2 = ", eta2, "eta2_new = ", eta2_new, "dphi = ", dphi
    dy = L*np.cos(eta2_new) - L*np.cos(eta2)
    dz = L*np.sin(eta2_new) - L*np.sin(eta2)
    #print dy,dz

    trans, quat = pose_cur
    trans_new = trans
    trans_new[1] += dy + dx[0]
    trans_new[2] += dz + dx[1]

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

    return new_msg, phi + dphi 

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

    return new_msg

def carrot_open_loop():
    #rospy.init_node('sandbox', anonymous=True)
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rospy.wait_for_service("plan_runner/init_task_space_streaming")
    sp = rospy.ServiceProxy('plan_runner/init_task_space_streaming',
        robot_msgs.srv.StartStreamingPlan)
    init = robot_msgs.srv.StartStreamingPlanRequest()
    rospy.sleep(1)
    #init.force_guard.append(make_force_guard_msg())
    print sp(init)
    pub = rospy.Publisher('plan_runner/task_space_streaming_setpoint',
        robot_msgs.msg.CartesianGoalPoint, queue_size=1)
    robotSubscriber = rosUtils.JointStateSubscriber("/joint_states")
    print("Waiting for full kuka state...")
    while len(robotSubscriber.joint_positions.keys()) < 3:
        rospy.sleep(0.1)
    print("got full state")

    new_msg = robot_msgs.msg.CartesianGoalPoint()
    frame_name = "iiwa_link_ee"
    try:
        current_pose_ee = rosUtils.poseFromROSTransformMsg(
            tfBuffer.lookup_transform("base", frame_name, rospy.Time()).transform)
        print current_pose_ee
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        print("Troubling looking up tf...")
        rate.sleep()

    # load offline controller
    state_and_control = pickle.load(open("example_8sol.p","rb"))
    pos_over_time = state_and_control["state"]
    F_over_time = state_and_control["control"]
    params = state_and_control["params"]

    phi = np.pi/2
    alpha = 0.05
    idx = int(params[0])
    T = int(params[idx+27])
    r0 = params[3]

    # fix initial state
    t = 0
    x,y,theta,x_dot,y_dot,theta_dot,d = pos_over_time[t,:]
    F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi0, omega = F_over_time[t,:]
    x_centroid = x - r0*np.sin(theta)
    y_centroid = y + r0*np.cos(theta)
    x_F1 = x_centroid-d*np.cos(theta)
    y_F1 = y_centroid-d*np.sin(theta)

    phi = phi0 
    frame_name = "iiwa_link_ee"
    # try:
    #     current_pose_ee = rosUtils.poseFromROSTransformMsg(
    #         tfBuffer.lookup_transform("base", frame_name, rospy.Time()).transform)
    #     #print current_pose_ee
    # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
    #     print("Troubling looking up tf...")
    #     rate.sleep()
    # dphi = phi0 - phi 
    # print "dphi=",dphi
    # new_msg, phi = rotate_around_left_finger_tip(current_pose_ee,dphi,phi,alpha)
    # start_time = time.time()
    # while (time.time() - start_time < 0.5):        
    #     pub.publish(new_msg)
    #     time.sleep(0.1)
    # alpha    
    dalpha = omega / 2.0 - alpha 
    print "dalpha = ", dalpha
    try:
        current_pose_ee = rosUtils.poseFromROSTransformMsg(
            tfBuffer.lookup_transform("base", frame_name, rospy.Time()).transform)
        #print current_pose_ee
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        print("Troubling looking up tf...")
        rate.sleep()
    new_msg = change_finger_distance_while_keeping_left_finger_tip_unmoved(current_pose_ee,phi,dalpha)
    start_time = time.time()
    while (time.time() - start_time < 1):
        pub.publish(new_msg)
        handDriver.sendGripperCommand(omega, force=80, speed=0.05)
        time.sleep(1)
    alpha += dalpha

    for t in range(1,T):
        x,y,theta,x_dot,y_dot,theta_dot,d = pos_over_time[t,:]
        F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi_next, omega = F_over_time[t,:]
        x_centroid = x - r0*np.sin(theta)
        y_centroid = y + r0*np.cos(theta)
        x_F1_next = x_centroid-d*np.cos(theta)
        y_F1_next = y_centroid-d*np.sin(theta)
        dx = x_F1_next - x_F1
        dy = y_F1_next - y_F1 
        x_F1 = x_F1_next
        y_F1 = y_F1_next

        #frame_name = "iiwa_link_ee"
        # phi 
        try:
            current_pose_ee = rosUtils.poseFromROSTransformMsg(
                tfBuffer.lookup_transform("base", frame_name, rospy.Time()).transform)
            #print current_pose_ee
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("Troubling looking up tf...")
            rate.sleep()
        dphi = phi_next - phi 
        dalpha = omega / 2.0 - alpha 
        print "dphi=",dphi/np.pi * 180.0, "dalpha=",dalpha, "dx=",dx,"dy=",dy
        new_msg, phi = rotate_around_left_finger_tip(current_pose_ee,dphi,phi,alpha,[dx,dy])
        start_time = time.time()
        while (time.time() - start_time < 0.5):        
            pub.publish(new_msg)
            time.sleep(0.1)
        # alpha    
        try:
            current_pose_ee = rosUtils.poseFromROSTransformMsg(
                tfBuffer.lookup_transform("base", frame_name, rospy.Time()).transform)
            #print current_pose_ee
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("Troubling looking up tf...")
            rate.sleep()
        new_msg = change_finger_distance_while_keeping_left_finger_tip_unmoved(current_pose_ee,phi,dalpha)
        start_time = time.time()
        while (time.time() - start_time < 1):
            pub.publish(new_msg)
            handDriver.sendGripperCommand(omega, force=80, speed=0.05)
            time.sleep(1)
        alpha += dalpha

    rospy.wait_for_service("plan_runner/stop_plan")
    sp = rospy.ServiceProxy('plan_runner/stop_plan',
        std_srvs.srv.Trigger)
    init = std_srvs.srv.TriggerRequest()
    print sp(init)

def test_task_space_streaming():
        #rospy.init_node('sandbox', anonymous=True)
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rospy.wait_for_service("plan_runner/init_task_space_streaming")
    sp = rospy.ServiceProxy('plan_runner/init_task_space_streaming',
        robot_msgs.srv.StartStreamingPlan)
    init = robot_msgs.srv.StartStreamingPlanRequest()
    rospy.sleep(1)
    #init.force_guard.append(make_force_guard_msg())
    print sp(init)
    pub = rospy.Publisher('plan_runner/task_space_streaming_setpoint',
        robot_msgs.msg.CartesianGoalPoint, queue_size=1)
    robotSubscriber = rosUtils.JointStateSubscriber("/joint_states")
    print("Waiting for full kuka state...")
    while len(robotSubscriber.joint_positions.keys()) < 3:
        rospy.sleep(0.1)
    print("got full state")

    new_msg = robot_msgs.msg.CartesianGoalPoint()
    frame_name = "iiwa_link_ee"
    try:
        current_pose_ee = rosUtils.poseFromROSTransformMsg(
            tfBuffer.lookup_transform("base", frame_name, rospy.Time()).transform)
        print current_pose_ee
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        print("Troubling looking up tf...")
        rate.sleep()

    # load offline controller
    state_and_control = pickle.load(open("example_8sol.p","rb"))
    pos_over_time = state_and_control["state"]
    F_over_time = state_and_control["control"]
    params = state_and_control["params"]

    phi = np.pi/2
    alpha = 0.05
    idx = int(params[0])
    T = int(params[idx+27])
    r0 = params[3]

    # fix initial state
    omega = 2*(alpha-0.025)
    handDriver.sendGripperCommand(omega, force=80, speed=0.05)
    time.sleep(1)
    print "initial omega = ", omega 

    frame_name = "iiwa_link_ee"
    try:
        current_pose_ee = rosUtils.poseFromROSTransformMsg(
            tfBuffer.lookup_transform("base", frame_name, rospy.Time()).transform)
        #print current_pose_ee
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        print("Troubling looking up tf...")
        rate.sleep()
    dphi = -2.0 /180*np.pi
    print "dphi=",dphi
    new_msg, phi = rotate_around_left_finger_tip(current_pose_ee,dphi,phi,alpha)
    start_time = time.time()
    while (time.time() - start_time < 0.5):        
        pub.publish(new_msg)
        time.sleep(0.1)

    rospy.wait_for_service("plan_runner/stop_plan")
    sp = rospy.ServiceProxy('plan_runner/stop_plan',
        std_srvs.srv.Trigger)
    init = std_srvs.srv.TriggerRequest()
    print sp(init)

def grip_to_d(schunk_driver,d):
       schunk_driver.sendGripperCommand(d, force=80, speed=0.05)

def pregrasp():
    pos = [0.63, 0., 0.165]
    quat = [ 0.71390524,  0.12793277,  0.67664775, -0.12696589] # straight down position
    # load offline controller
    state_and_control = pickle.load(open("example_8sol.p","rb"))
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
    # print quat
    # make goal
    goal = make_cartesian_trajectory_goal_world_frame(
        pos,
        quat,
        duration = 5.)
    goal.gains.append(make_cartesian_gains_msg(50., 10.))
    goal.force_guard.append(make_force_guard_msg(15.))
    return goal


def right_finger_move(driver, d, force=80, speed=0.05):
    # d: in target configuration, distance between two fingers
    d = min(d,0.1)
    driver.sendGripperCommand(d, force=80, speed=0.05)


data = [0.0,0.0]
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
    handDriver.reset_and_home()
    rospy.sleep(1)
    handDriver.sendOpenGripperCommand()
    rospy.sleep(1)
    i=0
    for goal in [pregrasp()]:
        print "sending goal"
        client.send_goal(goal)
        #handDriver.sendGripperCommand(0.054, force=80, speed=0.05)
        rospy.loginfo("waiting for CartesianTrajectory action result")
        client.wait_for_result()
        result = client.get_result()
    carrot_open_loop()
    #test_task_space_streaming()

