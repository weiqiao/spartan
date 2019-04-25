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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import osqp
import scipy.sparse as sparse

file_name = "trajopt_example14_latest"
EPS = 0.0001
DynamicsConstraintEps = 0.00001
PositionConstraintEps = 0.01
FingerWidth = 0.015
mu_ground = 0.5 # frictional coefficient between ground and carrot
mu_finger = 0.2 # frictional coefficient between finger and carrot
g = 9.8
r = 0.036
mass = 0.006565
inertia = (np.pi/4-8/(9*np.pi))*(2*mass*r**2/np.pi)
DistanceCentroidToCoM = 4*r/(3*np.pi)
MaxInputForce = 100
MaxRelVel = 0.005
StateBound = np.array([[-4,-1,-np.pi,-2,-2,-2],[4,1,np.pi,2,2,2]])
OptimizationSlackEps = 0.01
VISUALIZE = 1
USE_GOOD_INITIAL_GUESS = 0 
SAVE_STATE_AND_CONTROL = 1
d = 0.018 # hard code
def get_back_controller(cur_linear_cell,x,cur_phi,x_ref,G,G_inv):
	# solve LP using osqp
	# cur_linear_cell: current linear cell
	# x: current state
	# f(x,u) + delta = Ax+Bu+c+delta is in x_ref + G*p, -1 <= p(i) <= 1, for all i
	A = cur_linear_cell.A 
	B = cur_linear_cell.B
	c = cur_linear_cell.c 
	p = cur_linear_cell.p
	H = p.H
	h = p.h 
	n = len(x)
	m = int(B.shape[1])
	#print("n=%d,m=%d"%(n,m))
	# t = [u_1,...,u_m|delta_1,...,delta_n|p_1,...,p_n|w]
	A_eq1 = np.hstack((B,np.eye(n),-G,np.zeros((n,1))))
	b_eq1 = (-A.dot(x)-c+x_ref)[:,0]

	A_ub1 = np.hstack((np.zeros((n,m+n)),np.eye(n),np.zeros((n,1))))
	b_ub1 = np.ones(n)

	#print("constraint 1", A_ub1.shape, b_ub1.shape)

	A_ub2 = np.hstack((np.zeros((n,m+n)),-np.eye(n),np.zeros((n,1))))
	b_ub2 = np.ones(n)

	A_ub3 = np.hstack((np.zeros((n,m)),np.eye(n),np.zeros((n,n)),-np.ones((n,1))))
	b_ub3 = np.zeros(n)

	A_ub4 = np.hstack((np.zeros((n,m)),-np.eye(n),np.zeros((n,n)),-np.ones((n,1))))
	b_ub4 = np.zeros(n)

	num_constraints = H.shape[0]
	A_ub5 = np.zeros((num_constraints,2*n+m+1))
	A_ub5[:,:m] = H[:,n:]
	b_ub5 = (-H[:,:n].dot(x) + h + EPS)[:,0]

	# phi cannot change too much
	# phi <= cur_phi + 1.0/180.0*np.pi
	# -phi <= -(cur_phi - 1.0/180.0*np.pi)
	prob_dim = 2*n+m+1
	A_ub6 = np.zeros((2,prob_dim))
	A_ub6[0,m-1] = 1
	A_ub6[1,m-1] = -1
	b_ub6 = np.array([cur_phi+1.0/180.0*np.pi, -(cur_phi - 1.0/180.0*np.pi)])

	#print("constraint 3", A_ub3.shape, b_ub3.shape)
	A_ub = np.vstack((A_ub1,A_ub2,A_ub3,A_ub4,A_ub5,A_ub6,A_eq1,-A_eq1))
	A_ub = sparse.csc_matrix(A_ub)
	b_ub = np.hstack((b_ub1,b_ub2,b_ub3,b_ub4,b_ub5,b_ub6,b_eq1,-b_eq1))
	b_lb = -np.inf*np.ones(len(b_ub))
	# A_ub = []#np.vstack((A_ub2))
	# b_ub = []#np.hstack((b_ub2))

	
	P = np.zeros((prob_dim,prob_dim))
	P = sparse.csc_matrix(P)
	q = np.zeros(prob_dim)
	q[prob_dim-1] = 1
	#print("A upper bound")
	#print(A_ub)
	#print("b upper bound")
	#print(b_ub)

	# Create an OSQP object
	prob = osqp.OSQP()

	# Setup workspace and change alpha parameter
	prob.setup(P, q, A_ub, b_lb, b_ub,verbose = False)

	# Solve problem
	res = prob.solve()
	# print(res.info.status)
	#print(res.x)
	#assert(1 == 0)
	return res.x[:m], res.x[m:m+n], res.x[2*n+m]


def get_back_controller_linprog(cur_linear_cell,x,x_ref,G,G_inv):
    # solve LP using scipy.optimize.linprog
    # cur_linear_cell: current linear cell
    # x: current state
    # f(x,u) + delta = Ax+Bu+c+delta is in x_ref + G*p, -1 <= p(i) <= 1, for all i
    A = cur_linear_cell.A 
    B = cur_linear_cell.B
    c = cur_linear_cell.c 
    p = cur_linear_cell.p
    H = p.H
    h = p.h 
    n = len(x)
    m = int(B.shape[1])
    print("n=%d,m=%d"%(n,m))
    # t = [u_1,...,u_m|delta_1,...,delta_n|p_1,...,p_n|w]
    A_eq1 = np.hstack((B,np.eye(n),-G,np.zeros((n,1))))
    b_eq1 = (-A.dot(x)-c+x_ref)[:,0]

    A_ub1 = np.hstack((np.zeros((n,m+n)),np.eye(n),np.zeros((n,1))))
    b_ub1 = np.ones(n)

    #print("constraint 1", A_ub1.shape, b_ub1.shape)

    A_ub2 = np.hstack((np.zeros((n,m+n)),-np.eye(n),np.zeros((n,1))))
    b_ub2 = np.ones(n)

    A_ub3 = np.hstack((np.zeros((n,m)),np.eye(n),np.zeros((n,n)),-np.ones((n,1))))
    b_ub3 = np.zeros(n)

    A_ub4 = np.hstack((np.zeros((n,m)),-np.eye(n),np.zeros((n,n)),-np.ones((n,1))))
    b_ub4 = np.zeros(n)

    num_constraints = H.shape[0]
    A_ub5 = np.zeros((num_constraints,2*n+m+1))
    A_ub5[:,:m] = H[:,n:]
    b_ub5 = (-H[:,:n].dot(x) + h + EPS)[:,0]

    #print("constraint 3", A_ub3.shape, b_ub3.shape)
    A_ub = np.vstack((A_ub1,A_ub2,A_ub3,A_ub4,A_ub5))
    b_ub = np.hstack((b_ub1,b_ub2,b_ub3,b_ub4,b_ub5))

    # A_ub = []#np.vstack((A_ub2))
    # b_ub = []#np.hstack((b_ub2))

    cost = np.zeros(2*n+m+1)
    cost[2*n+m] = 1

    print("A eq")
    print(A_eq1)
    print("b eq")
    print(b_eq1)
    print("A upper bound")
    print(A_ub)
    print("b upper bound")
    print(b_ub)

    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq1, b_eq=b_eq1, options={"disp":True})
    print(res.x)
    return res.x[:m], res.x[m:m+n], res.x[2*n+m]

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
	polytube_controller = pickle.load(open("trajopt_example14_latest"+"_tube_output.p","rb"))
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
	## sanity check
	# for t in range(__):
	# 	print('t=%d'%t)
	# 	print('x=',pos_over_time[t,:])
	# 	print('u=',F_over_time[t,:])
	init_state = pos_over_time[0,:]
	init_state[2] = 10.0*np.pi/180.0
	init_state[0] = DistanceCentroidToCoM*np.sin(init_state[2])-r*init_state[2]
	init_state[1] = r - DistanceCentroidToCoM*np.cos(init_state[2])
	cur_x = init_state
	init_phi = F_over_time[0,-1]
	cur_phi = init_phi
	for t in range(__):
		# print('t=%d,angle=%f'%(t,cur_x[2]/np.pi*180.0))
		# # get current controller
		# cur_x = cur_x.reshape((-1,1))
		# print('cur_x=',cur_x)
		# cur_x_stack = np.tile(cur_x,(l,1)) # equivalent to repmat
		# print('cur_x_stack=',cur_x_stack)
		# px = G_inv_stack.dot(cur_x_stack-x_stack)
		# print('px=',px)
		# px_star = np.maximum(np.minimum(px,1),-1)
		# print('px_star=',px_star)
		# d_signed = G_stack.dot(px-px_star)
		# print('d_signed=',d_signed)
		# d_signed = d_signed.reshape((l,n))
		# print('d_signed=',d_signed)
		# d = np.linalg.norm(d_signed,axis = 1)
		# print('d=',d)
		# idx_min = np.argmin(d)
		# print('idx_min=',idx_min)
		# d_min = d[idx_min]
		# print('d_min=',d_min)
		# if d_min > 0:
		# 	idx_min += 1 # the child of the node
		# 	print('idx_min=%d'%idx_min)
		# 	cur_linear_cell = polytube_controller_list_of_cells[idx_min] # linear_cell(A,B,c,polytope(H,h))
		# 	cur_u_lin, cur_delta = get_back_controller(cur_linear_cell,cur_x,polytube_controller_x[idx_min],polytube_controller_G[idx_min],polytube_controller_G_inv[idx_min])
		# 	print('cur_u_lin=',cur_u_lin)
		# 	print('cur_delta=',cur_delta)
		# 	next_x_lin = (cur_linear_cell.A.dot(cur_x) + cur_linear_cell.B.dot(cur_u_lin).reshape((-1,1)) + cur_linear_cell.c)[:,0]
		# 	print('next_x_lin=',next_x_lin)
		# else:
		# 	# inside nearest polytope, use polytopic control law
		# 	cur_u_lin = polytube_controller_u[idx_min]
		# 	print('cur_u_lin=',cur_u_lin)
		# 	cur_linear_cell = polytube_controller_list_of_cells[idx_min] # linear_cell(A,B,c,polytope(H,h))
		# 	next_x_lin = cur_linear_cell.A.dot(cur_x) + cur_linear_cell.B.dot(cur_u_lin) + cur_linear_cell.c
		# 	print('next_x_lin=',next_x_lin)
		
		cur_x = cur_x.reshape((-1,1))
		idx_min = -1
		w_min = None
		for idx in range(T):
			#print('idx=%d'%idx)
			cur_linear_cell = polytube_controller_list_of_cells[idx] # linear_cell(A,B,c,polytope(H,h))
			cur_u_lin, cur_delta, cur_w = get_back_controller(cur_linear_cell,cur_x,cur_phi,polytube_controller_x[idx],polytube_controller_G[idx],polytube_controller_G_inv[idx])
			if w_min == None or cur_w < w_min:
				idx_min = idx
				u_lin_min = cur_u_lin
				w_min = cur_w
		assert(idx_min > -1)
		EPS = 1e-6
		if idx_min >= T - 1:
			print('success')
			break
		print('idx=%d'%idx_min)
		if w_min < EPS: # inside nearest polytope, use polytopic control law
			cur_u_lin = polytube_controller_u[idx_min]
		cur_linear_cell = polytube_controller_list_of_cells[idx_min]
		next_x_lin = (cur_linear_cell.A.dot(cur_x) + cur_linear_cell.B.dot(cur_u_lin) + cur_linear_cell.c)[:,0]
		visualize(cur_x,cur_u_lin,t)
		cur_x = next_x_lin
		cur_x[0] = DistanceCentroidToCoM*np.sin(cur_x[2])-r*cur_x[2]
		cur_x[1] = r - DistanceCentroidToCoM*np.cos(cur_x[2])
	visualize(cur_x,cur_u_lin,t)
    	

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
	F1, F1t, F2, F2t, Fn, Ft, phi = F
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


def draw_left_finger(ax, X, F):
	Width = 0.008 # TODO
	Length = .1 # TODO

	x,y,theta=X[:3]
	F1, F1t, F2, F2t, Fn, Ft, phi = F

	omega = d*np.sin(phi-theta)+r


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
	F1, F1t, F2, F2t, Fn, Ft, phi = F

	omega = d*np.sin(phi-theta)+r

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
	run()
