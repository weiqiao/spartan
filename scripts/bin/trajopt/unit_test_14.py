import numpy as np
import scipy as sp
import scipy.linalg
from gurobipy import Model,GRB,LinExpr
import pickle

import time as time

from pypolycontain.lib.containment_encodings import subset_LP,subset_zonotopes
from pypolycontain.lib.polytope import polytope
from pypolycontain.lib.zonotope import zonotope

import carrot_lin_14 as calin1 
import poly_trajectory_14 as polytraj 
from pwa_system import system, linear_cell
import pickle

def test1():
	file_name = "trajopt_example14_latest"
	state_and_control = pickle.load(open(file_name + ".p","rb"))
	pos_over_time = state_and_control["state"]
	F_over_time = state_and_control["control"]
	params = state_and_control["params"]
	x = pos_over_time
	u = F_over_time
	'''
		params_save = np.array([11])                                                                    # 0: idx
		params_save = np.append(params_save,np.array([mass, inertia, DistanceCentroidToCoM, r, dt, 
			DynamicsConstraintEps,PositionConstraintEps,mu_ground,mu_finger,MaxInputForce,MaxRelVel]))  # 1~idx
		params_save = np.append(params_save,StateBound[0,:])											# idx+1~idx+6
		params_save = np.append(params_save,StateBound[1,:])											# idx+7~idx+12
		params_save = np.append(params_save,pos_init)												    # idx+13~idx+18
		params_save = np.append(params_save,pos_final)												 	# idx+19~idx+24
		params_save = np.append(params_save,[T])														# idx+25
	'''
	idx = int(params[0])
	T = int(params[idx+25])
	t0 = 0
	# t0 = int(T/4*3)
	#t0 = 0
	list_of_cells2 = []
	n = 6
	m = 7
	for t in range(t0,T):
		A,B,c,H,h = calin1.linearize(pos_over_time[t,:], F_over_time[t,:], params)
		print('xt')
		print(pos_over_time[t,:])
		print('ut')
		print(F_over_time[t,:])
		print('x(t+1)')
		print(pos_over_time[t+1,:])
		print(A.dot(x[t,:]).shape)
		print(B.dot(u[t,:]).shape)
		print(c.shape)
		print((A.dot(x[t,:])-B.dot(u[t,:])-c.reshape(1,-1)).shape)
		print(A.dot(x[t,:])-B.dot(u[t,:])-c.reshape(1,-1))
		x_t=np.array([x[t,j] for j in range(n)]).reshape(n,1)
		u_t=np.array([u[t,j] for j in range(m)]).reshape(m,1)
		assert(np.sum(np.abs(x[t+1,:]-A.dot(x[t,:])-B.dot(u[t,:])-c.reshape(1,-1))) < 1e-4)

def test2():
	file_name = "trajopt_example14_latest"
	state_and_control = pickle.load(open(file_name + ".p","rb"))
	pos_over_time = state_and_control["state"]
	F_over_time = state_and_control["control"]
	params = state_and_control["params"]
	x = pos_over_time
	u = F_over_time

	idx = int(params[0])
	T = int(params[idx+25])
	t0 = 0
	# t0 = int(T/4*3)
	#t0 = 0
	list_of_cells2 = []
	n = 6
	m = 7
	for t in range(t0,T):
		A,B,c,H,h = calin1.linearize(pos_over_time[t,:], F_over_time[t,:], params)
		print(H.shape)
		print(h.shape)
		print(np.hstack((x[t,:],u[t,:])).shape)
		print(H.dot(np.hstack((x[t,:],u[t,:]))).shape)
		print((H.dot(np.hstack((x[t,:],u[t,:])))-h.reshape(1,-1)).shape)
		print((H.dot(np.hstack((x[t,:],u[t,:])))-h.reshape(1,-1)))	
		tmp = H.dot(np.hstack((x[t,:],u[t,:])))-h.reshape(1,-1)	
		tmp = tmp[0,:]
		print(tmp.shape)
		all(tmp<=0e-14)

def test3():
	file_name = "trajopt_example14_latest"
	state_and_control = pickle.load(open(file_name + ".p","rb"))
	pos_over_time = state_and_control["state"]
	F_over_time = state_and_control["control"]
	params = state_and_control["params"]
	x = pos_over_time
	u = F_over_time

	polytube_controller = pickle.load(open("trajopt_example14_latest"+"_tube_output.p","rb"))
	polytube_controller_x = polytube_controller['x']
	polytube_controller_u = polytube_controller['u']
	polytube_controller_theta = polytube_controller['theta']
	polytube_controller_list_of_cells = polytube_controller['list_of_cells']

	idx = int(params[0])
	T = int(params[idx+25])
	t0 = 0

	cur_x = pos_over_time[0,:]
	print('cur_x')
	print(cur_x)
	for t in range(t0,2):
		A,B,c,H,h = calin1.linearize(pos_over_time[t,:], F_over_time[t,:], params)
		cur_u_lin = F_over_time[t]
		next_x_lin = (A.dot(cur_x) + B.dot(cur_u_lin) + c[:,0])
		assert(np.sum(np.abs(next_x_lin - pos_over_time[t+1,:])) < 1e-4)

		cur_linear_cell = polytube_controller_list_of_cells[t]
		next_x_lin2 = (cur_linear_cell.A.dot(cur_x) + cur_linear_cell.B.dot(cur_u_lin) + cur_linear_cell.c[:,0])

		print('A')
		print(A)
		print('B')
		print(B)
		print('c')
		print(c)
		print('cell_A')
		print(cur_linear_cell.A)
		print('cell_B')
		print(cur_linear_cell.B)
		print('cell_c')
		print(cur_linear_cell.c)
		print('cur_u_lin')
		print(cur_u_lin)
		print('next_x_lin')
		print(next_x_lin)
		print('next_x_lin2')
		print(next_x_lin2)

		cur_x = next_x_lin2

	# cur_x = pos_over_time[0]
	# for t in range(t0,T):
	# 	cur_linear_cell = polytube_controller_list_of_cells[t]
	# 	cur_u_lin = F_over_time[t]
	# 	next_x_lin = (cur_linear_cell.A.dot(cur_x) + cur_linear_cell.B.dot(cur_u_lin) + cur_linear_cell.c[:,0])
	# 	print('next_x_lin')
	# 	print(next_x_lin)
	# 	print('pos_over_time')
	# 	print(pos_over_time[t+1])
	# 	assert(np.sum(np.abs(next_x_lin - pos_over_time[t+1,:])) < 1e-3)
	# 	#cur_x = next_x_lin
	# 	cur_x = pos_over_time[t+1]


if __name__=="__main__":
	test3()