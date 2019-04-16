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
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

file_name = "trajopt_example13_latest"
EPS = 0.0001
def get_back_controller(cur_linear_cell,x,x_ref,G_inv):
    # solve LP using scipy.optimize.linprog
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

    # u = t[:m], delta = t[m:] = t[m:m+n]

    A_ub1 = np.hstack((G_inv.dot(B),G_inv))
    b_ub1 = (1+EPS-G_inv.dot(A.dot(x) + c - x_ref))[:,0]

    print("constraint 1", A_ub1.shape, b_ub1.shape)

    A_ub2 = - np.hstack((G_inv.dot(B),G_inv))
    b_ub2 = (1+EPS+G_inv.dot(A.dot(x) + c - x_ref))[:,0]

    print("constraint 2", A_ub2.shape, b_ub2.shape)

    num_constraints = H.shape[0]
    A_ub3 = np.zeros((num_constraints,n+m))
    A_ub3[:,:m] = H[:,n:]
    b_ub3 = (-H[:,:n].dot(x) + h + EPS)[:,0]

    print("constraint 3", A_ub3.shape, b_ub3.shape)
    #A_ub = np.vstack((A_ub1,A_ub2,A_ub3))
    #b_ub = np.hstack((b_ub1,b_ub2,b_ub3))
    # A_ub = np.vstack((A_ub3))
    # b_ub = np.hstack((b_ub3))
    A_ub = np.vstack((A_ub1,A_ub2))
    b_ub = np.hstack((b_ub1,b_ub2))

    cost = np.zeros(n+m)
    cost[m:] = 1
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, options={"disp":True})
    print(res)
    return res[:m], res[m:]

def run():
	state_and_control = pickle.load(open(file_name + ".p","rb"))
	pos_over_time = state_and_control["state"]
	F_over_time = state_and_control["control"]
	params = state_and_control["params"]

	idx = int(params[0])
	global DistanceCentroidToCoM
	global r
	m, I, DistanceCentroidToCoM, r, dt, DynamicsConstraintEps,PositionConstraintEps,mu_ground,mu_finger,MaxInputForce,MaxRelVel = params[1:idx+1] # mass, inertia
	T = int(params[idx+25])


	# load linearization
	polytube_controller = pickle.load(open("trajopt_example13_latest"+"_tube_output.p","rb"))
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


	__ = 100 # total time steps
	init_state = pos_over_time[10,:]
	# init_state[0] -= 0.1
	cur_x = init_state
	for t in range(__):
		print('t=%d',t)
		# get current controller
		cur_x = cur_x.reshape((-1,1))
		cur_x_stack = np.tile(cur_x,(l,1)) # equivalent to repmat
		px = G_inv_stack.dot(cur_x_stack-x_stack)
		px_star = np.maximum(np.minimum(px,1),-1)
		d_signed = G_stack.dot(px-px_star)
		d_signed = d_signed.reshape((l,n))
		d = np.linalg.norm(d_signed,axis = 1)
		idx_min = np.argmin(d)
		d_min = d[idx_min]
		if d_min > 0:
			idx_min += 1 # the child of the node
			print('idx_min=%d'%idx_min)
			cur_linear_cell = polytube_controller_list_of_cells[idx_min] # linear_cell(A,B,c,polytope(H,h))
			cur_u_lin, cur_delta = get_back_controller(cur_linear_cell,cur_x,polytube_controller_x[idx_min],polytube_controller_G_inv[idx_min])
			next_x_lin = (cur_linear_cell.A.dot(cur_x) + cur_linear_cell.B.dot(cur_u_lin).reshape((-1,1)) + cur_linear_cell.c)[:,0]
		else:
			# inside nearest polytope, use polytopic control law
			cur_u_lin = polytube_controller_u[idx_min]
			cur_linear_cell = polytube_controller_list_of_cells[idx_min] # linear_cell(A,B,c,polytope(H,h))
			next_x_lin = cur_linear_cell.A.dot(cur_x) + cur_linear_cell.B.dot(cur_u_lin) + cur_linear_cell.c
		visualize(cur_x,cur_u_lin)
		cur_x = next_x_lin
	visualize(cur_x,cur_u_lin)
    	

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
    draw_force(ax1,X,F) # draw forces
    t += 1
    fig.savefig(file_name+'_pwa_fig_latest/carrot_%d.png'%t, dpi=100)
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
	F1, F1t, Fn, Ft = F
	x_centroid = x - DistanceCentroidToCoM*np.sin(theta)
	y_centroid = y + DistanceCentroidToCoM*np.cos(theta)

	x_F1 = x_centroid-d*np.cos(theta)
	y_F1 = y_centroid-d*np.sin(theta)
	dx_F1 = np.sin(theta)*F1*force_scaling_factor
	dy_F1 = -np.cos(theta)*F1*force_scaling_factor

	x_F1_tp = x_F1
	y_F1_tp = y_F1
	dx_F1_tp = np.cos(theta)*F1t*force_scaling_factor
	dy_F1_tp = np.sin(theta)*F1t*force_scaling_factor

	x_Fn = x_centroid
	y_Fn = 0
	dx_Fn = 0
	dy_Fn = Fn*force_scaling_factor

	x_Ft = x_centroid
	y_Ft = 0
	dx_Ft = Ft*force_scaling_factor
	dy_Ft = 0

	x_G = x 
	y_G = y
	dx_G = 0
	dy_G = -1*mass*g*force_scaling_factor

	ax.arrow(x_F1,y_F1,dx_F1,dy_F1,color=(1,0,1),head_width=0.0005, head_length=0.001)
	ax.arrow(x_F1_tp,y_F1_tp,dx_F1_tp,dy_F1_tp,color=(1,0,1),head_width=0.0005, head_length=0.001)
	ax.arrow(x_Fn,y_Fn,dx_Fn,dy_Fn,color=(1,0,1),head_width=0.0005, head_length=0.001)
	ax.arrow(x_Ft,y_Ft,dx_Ft,dy_Ft,color=(1,0,1),head_width=0.0005, head_length=0.001)
	ax.arrow(x_G,y_G,dx_G,dy_G,color=(1,0,1),head_width=0.0005, head_length=0.001)


if __name__ == "__main__":
	run()