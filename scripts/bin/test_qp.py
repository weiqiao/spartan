from __future__ import absolute_import, division, print_function
import unittest
import subprocess
import psutil
import sys
import os
import numpy as np
import time
import socket
import pickle
import scipy
from scipy.linalg import block_diag
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

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


file_name = "trajopt_example11_latest"
EPS = 0.0001


def print_pos(cur_pos,cur_F):
	print("x=%f,y=%f,theta=%f,phi=%f,omega=%f"%(cur_pos[0],cur_pos[1],cur_pos[2],cur_F[10],cur_F[11]))

def print_pos_data(cur_data):
	print("x=%f,theta=%f"%(cur_data[1],cur_data[0]))

def display_data():
    output2 = pickle.load(open("trajopt_example11_latest_artificial_disturbance.p","rb"))
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

    for ttt in range(T):
    	print_pos(pos_over_time[ttt,:],F_over_time[ttt,:])
    	if ttt < len(output2):
    		print_pos_data(output2[ttt,:])    	


def get_back_controller(cur_linear_cell,x,x_ref,G_inv):
    # cur_linear_cell: current linear cell
    # x: current state
    # f(x,u) + delta = Ax+Bu+c+delta is in x_ref + G*P
    A = cur_linear_cell.A 
    B = cur_linear_cell.B
    c = cur_linear_cell.c 
    p = cur_linear_cell.p
    H = p.H
    h = p.h 
    n = len(x)
    m = int(B.shape[1])

    def loss(x):
        u = x[:m]
        delta = x[m:]
        return 0.5*delta.dot(delta)

    def jac(x):
        delta = x[m:]
        ans = np.zeros(n+m)
        ans[m:] = delta 
        return ans     

    x0 = np.random.randn(n+m)
    # u = t[m:], delta = t[m:] = t[m:m+n]
    cons = []
    cons1 = {'type':'ineq',
        'fun':lambda t: 1 + EPS - (G_inv.dot(A.dot(x) + B.dot(t[:m]) + c + t[m:].reshape(-1,1)-x_ref))[:,0], # >= 0
        'jac':lambda t: -np.hstack((G_inv.dot(B),G_inv))}
    cons2 = {'type':'ineq',
        'fun':lambda t: 1 + EPS + (G_inv.dot(A.dot(x) + B.dot(t[:m]) + c + t[m:].reshape(-1,1)-x_ref))[:,0], # >= 0
        'jac':lambda t: np.hstack((G_inv.dot(B),G_inv))}
    cons.append(cons1)
    cons.append(cons2)
    num_constraints = H.shape[0]
    jac_H = np.zeros((num_constraints,n+m))
    jac_H[:,:m] = -H[:,n:]
    cons3 = {'type':'ineq',
        'fun':lambda t: (EPS - (H[:,:n].dot(x) + H[:,n:].dot(t[:m].reshape((-1,1))) - h))[:,0],
        'jac':lambda t: jac_H}
    opt = {'disp':False}
    cons.append(cons3)
    res_cons = optimize.minimize(loss, x0, jac=jac,constraints=cons,
                                 method='SLSQP', options=opt)
    x_res = res_cons['x']

    return x_res[:m], x_res[m:]

DistanceCentroidToCoM = None
r = None
def carrot_closed_loop_simulation2_visualize():
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
    global DistanceCentroidToCoM
    global r
    m, I, DistanceCentroidToCoM, r, dt, DynamicsConstraintEps, PositionConstraintEps, mu_ground,mu_finger, MaxInputForce, MaxRelVel = params[1:idx+1] # mass, inertia

    # load linearization
    polytube_controller = pickle.load(open("trajopt_example11_latest"+"_tube_output.p","rb"))
    polytube_controller_x = polytube_controller['x']
    polytube_controller_u = polytube_controller['u']
    polytube_controller_G = polytube_controller['G']
    polytube_controller_G_inv = polytube_controller['G_inv']
    polytube_controller_theta = polytube_controller['theta']
    polytube_controller_list_of_cells = polytube_controller['list_of_cells']
    
    # stack linearization matrices to simplify computation.
    l = len(polytube_controller_u)
    n = len(polytube_controller_x[0])
    x_stack = polytube_controller_x[0]
    u_stack = polytube_controller_u[0]
    G_stack = polytube_controller_G[0]
    G_inv_stack = polytube_controller_G_inv[0]
    theta_stack = polytube_controller_theta[0]
    for i in range(1,l):
        x_stack = np.vstack((x_stack, polytube_controller_x[i]))
        u_stack = np.vstack((u_stack, polytube_controller_u[i]))
        G_stack = scipy.linalg.block_diag(G_stack, polytube_controller_G[i])
        G_inv_stack = scipy.linalg.block_diag(G_inv_stack, polytube_controller_G_inv[i])
        theta_stack = np.vstack((theta_stack, polytube_controller_theta[i]))

    output1 = pickle.load(open("trajopt_example11_latest_artificial_no_disturbance.p","rb"))
    output2 = pickle.load(open("trajopt_example11_latest_artificial_disturbance.p","rb"))

    t = 80
	# the reference state
    x,y,theta,x_dot,y_dot,theta_dot,d = pos_over_time[t,:]
    F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi0, omega = F_over_time[t,:]
    x_centroid = x - r0*np.sin(theta)
    y_centroid = y + r0*np.cos(theta)
    x_F1 = x_centroid-d*np.cos(theta)
    y_F1 = y_centroid-d*np.sin(theta)
    gripper_state = [x_F1,y_F1,phi0,alpha]
    print("t=%f,x=%f,y=%f,phi=%f,theta=%f"%(t,x,y,phi,theta))

    # current state is loaded from artificial data
    data = output2[t,:]
    ### get controller
    theta_cur, x_cur = data 
    print("data:theta=%f,x=%f"%(x_cur,theta_cur))
    phi_cur = gripper_state[2]
    tol = np.pi/180.0*5
    theta_diff = np.abs(theta_cur - pos_over_time[t,2])
    phi_diff = np.abs(phi_cur - F_over_time[t,10])
    print("theta_diff=%f,phi_diff=%f"%(theta_diff,phi_diff))
    if theta_diff < tol and phi_diff < tol:
        print("normal t = %f"%t)
        t += 1
        x,y,theta,x_dot,y_dot,theta_dot,d = pos_over_time[t,:]
        F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi_next, omega = F_over_time[t,:]
        visualize(pos_over_time[t,:], F_over_time[t,:],t)
    else:
        # construct cur_x
        cur_x = np.array([x_cur,r-r0*np.cos(theta_cur),theta_cur,0,0,0,d])
        cur_x = cur_x.reshape((-1,1))
        # compute distance from cur_x to the nearest polytope
        cur_x_stack = np.tile(cur_x,(l,1)) # equivalent to repmat
        print(cur_x_stack.shape, x_stack.shape)
        px = G_inv_stack.dot(cur_x_stack-x_stack)
        px_star = np.maximum(np.minimum(px,1),-1)
        d_signed = G_stack.dot(px-px_star) 
        d_signed = d_signed.reshape((l,n))
        d = np.linalg.norm(d_signed,axis = 1)
        idx_min = np.argmin(d)
        d_min = d[idx_min]
        
        if d_min > 0:
            print("d_min>0")
            # outside nearest polytope, need solve QP
            idx_min += 1 # the child of the node
            cur_linear_cell = polytube_controller_list_of_cells[idx_min] # linear_cell(A,B,c,polytope(H,h))
            
            # solver = GurobiSolver()
            # prog1 = GetbackController()
            # cur_u_var, cur_delta_var = prog1.formulate_optimization_problem(cur_linear_cell,cur_x,polytube_controller_x[idx_min],polytube_controller_G_inv[idx_min])
            # result = solver.Solve(prog1)
            # assert result == mp.SolutionResult.kSolutionFound
            # cur_u_lin = prog1.get_solution(cur_u_var)
            # print cur_u_lin

            cur_u_lin, cur_delta = get_back_controller(cur_linear_cell,cur_x,polytube_controller_x[idx_min],polytube_controller_G_inv[idx_min])
            # the next state is computed from linear dynamics
            # or the full nonlinear dynamics?
            next_x_lin = (cur_linear_cell.A.dot(cur_x) + cur_linear_cell.B.dot(cur_u_lin).reshape((-1,1)) + cur_linear_cell.c)[:,0]
            x,y,theta,x_dot,y_dot,theta_dot,d = next_x_lin
            F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi_next, omega = cur_u_lin
            
            dx = x - pos_over_time[t,0]
            print(pos_over_time[t,:].shape, next_x_lin.shape)
            next_x_lin[-1] = pos_over_time[t,-1] + np.minimum(np.maximum((next_x_lin[-1]-pos_over_time[t,-1]),-0.1),0.1)
            cur_u_lin = F_over_time[t,:] + np.minimum(np.maximum((cur_u_lin-F_over_time[t,:]),-0.1),0.1)

            visualize(next_x_lin,cur_u_lin,t)
        else:
            print("d_min=0")
            # inside nearest polytope, use polytopic control law
            cur_u_lin = polytube_controller_u[idx_min]
            cur_linear_cell = polytube_controller_list_of_cells[idx_min] # linear_cell(A,B,c,polytope(H,h))
            next_x_lin = cur_linear_cell.A.dot(cur_x) + cur_linear_cell.B.dot(cur_u_lin) + cur_linear_cell.c
            x,y,theta,x_dot,y_dot,theta_dot,d = next_x_lin
            F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi_next, omega = cur_u_lin

        if idx_min != t:
            print("t changes from",t,"to",idx_min)
            t = idx_min


"""
Visualization Tools
"""    
def visualize(X,F,t):
    fig,ax1 = plt.subplots()
    ax1.set_xlabel("x",fontsize=20)
    ax1.set_ylabel("y",fontsize=20)
    ax1.set_xlim([-0.2,0.2])
    ax1.set_ylim([-0.2,0.4])
    fig.gca().set_aspect('equal')
    p_list=[]
    v=vertices(X)
    p_list.append(patches.Polygon(v, True))
    p=PatchCollection(p_list,color=(1,0,0),alpha=0.31,edgecolor=(1,0,0))
    ax1.add_collection(p)
    ax1.grid(color=(0,0,0), linestyle='--', linewidth=0.3)
    ax1.set_title("carrot %d"%t)
    ax1.plot([-12,12],[0,0],'black')
    ax1.plot(X[0],X[1],'+',color=(1,0,0))# draw CoM
    #draw_force(ax1,X,F) # draw forces
    draw_left_finger(ax1,X,F)
    draw_right_finger(ax1,X,F)
    t += 1
    fig.savefig(file_name+'_fig_latest/carrot_%d.png'%t, dpi=100)
    plt.close()
    return fig

def vertices(X,N=50):
    x,y,theta=X[:3]
    x_centroid = x - DistanceCentroidToCoM*np.sin(theta)
    y_centroid = y + DistanceCentroidToCoM*np.cos(theta)
    v=np.empty((N,2))
    for k in range(50):
        phi=-np.pi/2+np.pi/(N-1)*k
        v[k,0]=x_centroid+r*np.sin(phi+theta)
        v[k,1]=y_centroid-r*np.cos(phi+theta)
    return v

def draw_force(ax,X,F):
	force_scaling_factor = 0.01 # TODO
	x,y,theta=X[:3]
	d = X[-1]
	F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi, omega = F
	x_centroid = x - DistanceCentroidToCoM*np.sin(theta)
	y_centroid = y + DistanceCentroidToCoM*np.cos(theta)

	x_F1 = x_centroid-d*np.cos(theta)
	y_F1 = y_centroid-d*np.sin(theta)
	dx_F1 = np.sin(theta)*F1*force_scaling_factor
	dy_F1 = -np.cos(theta)*F1*force_scaling_factor

	x_F1_tp = x_F1
	y_F1_tp = y_F1
	dx_F1_tp = np.cos(theta)*F1_tp*force_scaling_factor
	dy_F1_tp = np.sin(theta)*F1_tp*force_scaling_factor

	x_F1_tm = x_F1
	y_F1_tm = y_F1
	dx_F1_tm = -np.cos(theta)*F1_tm*force_scaling_factor
	dy_F1_tm = -np.sin(theta)*F1_tm*force_scaling_factor

	x_Fn = x_centroid
	y_Fn = 0
	dx_Fn = 0
	dy_Fn = Fn*force_scaling_factor

	x_Ft = x_centroid
	y_Ft = 0
	dx_Ft = Ft*force_scaling_factor
	dy_Ft = 0

	x_F2 = x_centroid + r*np.sin(phi)
	y_F2 = y_centroid - r*np.cos(phi)
	dx_F2 = -np.sin(phi)*F2*force_scaling_factor
	dy_F2 = np.cos(phi)*F2*force_scaling_factor

	x_F2_tp = x_F2 
	y_F2_tp = y_F2
	dx_F2_tp = F2_tp*np.cos(phi)*force_scaling_factor
	dy_F2_tp = F2_tp*np.sin(phi)*force_scaling_factor

	x_F2_tm = x_F2 
	y_F2_tm = y_F2 
	dx_F2_tm = -F2_tm*np.cos(phi)*force_scaling_factor
	dy_F2_tm = -F2_tm*np.sin(phi)*force_scaling_factor

	x_G = x 
	y_G = y
	dx_G = 0
	dy_G = -1*mass*g*force_scaling_factor

	ax.arrow(x_F1,y_F1,dx_F1,dy_F1,color=(1,0,1),head_width=0.0005, head_length=0.001)
	ax.arrow(x_F1_tp,y_F1_tp,dx_F1_tp,dy_F1_tp,color=(1,0,1),head_width=0.0005, head_length=0.001)
	ax.arrow(x_F1_tm,y_F1_tm,dx_F1_tm,dy_F1_tm,color=(1,0,1),head_width=0.0005, head_length=0.001)
	ax.arrow(x_Fn,y_Fn,dx_Fn,dy_Fn,color=(1,0,1),head_width=0.0005, head_length=0.001)
	ax.arrow(x_Ft,y_Ft,dx_Ft,dy_Ft,color=(1,0,1),head_width=0.0005, head_length=0.001)
	ax.arrow(x_F2,y_F2,dx_F2,dy_F2,color=(1,0,1),head_width=0.0005, head_length=0.001)
	ax.arrow(x_F2_tp,y_F2_tp,dx_F2_tp,dy_F2_tp,color=(1,0,1),head_width=0.0005, head_length=0.001)
	ax.arrow(x_F2_tm,y_F2_tm,dx_F2_tm,dy_F2_tm,color=(1,0,1),head_width=0.0005, head_length=0.001)
	ax.arrow(x_G,y_G,dx_G,dy_G,color=(1,0,1),head_width=0.0005, head_length=0.001)
	# ax.arrow(x_F1,y_F1,dx_F1,dy_F1,color=(1,0,1))
	# ax.arrow(x_F1_tp,y_F1_tp,dx_F1_tp,dy_F1_tp,color=(1,0,1))
	# ax.arrow(x_F1_tm,y_F1_tm,dx_F1_tm,dy_F1_tm,color=(1,0,1))
	# ax.arrow(x_Fn,y_Fn,dx_Fn,dy_Fn,color=(1,0,1))
	# ax.arrow(x_Ft,y_Ft,dx_Ft,dy_Ft,color=(1,0,1))
	# ax.arrow(x_F2,y_F2,dx_F2,dy_F2,color=(1,0,1))
	# ax.arrow(x_F2_tp,y_F2_tp,dx_F2_tp,dy_F2_tp,color=(1,0,1))
	# ax.arrow(x_F2_tm,y_F2_tm,dx_F2_tm,dy_F2_tm,color=(1,0,1))
	# ax.arrow(x_G,y_G,dx_G,dy_G,color=(1,0,1))

def draw_left_finger(ax, X, F):
	Width = 0.008 # TODO
	Length = .1 # TODO

	x,y,theta=X[:3]
	d = X[-1]
	F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi, omega = F

	x_centroid = x - DistanceCentroidToCoM*np.sin(theta)
	y_centroid = y + DistanceCentroidToCoM*np.cos(theta)
	x_F1 = x_centroid-d*np.cos(theta)
	y_F1 = y_centroid-d*np.sin(theta)

	# counter clock wise counting vertices
	v_x = [x_F1, x_F1 + Length*np.cos(phi), x_F1 + Length*np.cos(phi) + Width*np.cos(phi+np.pi/2), x_F1 + Width*np.cos(phi+np.pi/2)]
	v_y = [y_F1, y_F1 + Length*np.sin(phi), y_F1 + Length*np.sin(phi) + Width*np.sin(phi+np.pi/2), y_F1 + Width*np.sin(phi+np.pi/2)]

	left_finger = [patches.Polygon(np.array([v_x+v_y]).reshape(2,4).T, True)]
	ax.add_collection(PatchCollection(left_finger, color=(0,0,0),alpha=0.8,edgecolor=(0,0,0)))

def draw_right_finger(ax, X, F):
	Width = 0.008 # TODO
	Length = .1 # TODO

	x,y,theta=X[:3]
	d = X[-1]
	F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi, omega = F

	x_centroid = x - DistanceCentroidToCoM*np.sin(theta)
	y_centroid = y + DistanceCentroidToCoM*np.cos(theta)
	x_F1 = x_centroid-d*np.cos(theta)
	y_F1 = y_centroid-d*np.sin(theta)
	x_F1 += (Width+omega)*np.cos(phi-np.pi/2)
	y_F1 += (Width+omega)*np.sin(phi-np.pi/2)

	# counter clock wise counting vertices
	v_x = [x_F1, x_F1 + Length*np.cos(phi), x_F1 + Length*np.cos(phi) + Width*np.cos(phi+np.pi/2), x_F1 + Width*np.cos(phi+np.pi/2)]
	v_y = [y_F1, y_F1 + Length*np.sin(phi), y_F1 + Length*np.sin(phi) + Width*np.sin(phi+np.pi/2), y_F1 + Width*np.sin(phi+np.pi/2)]

	right_finger = [patches.Polygon(np.array([v_x+v_y]).reshape(2,4).T, True)]
	ax.add_collection(PatchCollection(right_finger, color=(0,0,0),alpha=0.8,edgecolor=(0,0,0)))

if __name__ == "__main__":
	#display_data()
	carrot_closed_loop_simulation2_visualize()