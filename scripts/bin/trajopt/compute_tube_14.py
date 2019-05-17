import pickle
import numpy as np
import scipy as sp
from gurobipy import Model,GRB,LinExpr
import time as time
from pypolycontain.lib.containment_encodings import subset_LP,subset_zonotopes
from pypolycontain.lib.polytope import polytope
from pypolycontain.lib.zonotope import zonotope

import carrot_lin_14 as calin1 
import poly_trajectory_14 as polytraj 
from pwa_system import system, linear_cell
import pickle

# compute tube for example 14.

SAVE_OUTPUT = 0
if __name__=="__main__":
	file_name = "trajopt_example14_latest"
	state_and_control = pickle.load(open(file_name + ".p","rb"))
	pos_over_time = state_and_control["state"]
	F_over_time = state_and_control["control"]
	params = state_and_control["params"]
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
	list_of_cells = []
	for t in range(t0,T):
		A,B,c,H,h = calin1.linearize(pos_over_time[t,:], F_over_time[t,:], params)
		print('A',A)
		print('B',B)
		print('c',c)
		list_of_cells.append(linear_cell(A,B,c,polytope(H,h)))
		print(list_of_cells[t].A)
		print(list_of_cells[t].B)
		print(list_of_cells[t].c)

	pos_init = pos_over_time[t0,:]
	x0 = pos_init.reshape((-1,1))
	pos_final = params[idx+19:idx+25]
	n = 6
	epsilon = 0.01
	goal=zonotope(pos_final.reshape(-1,1),np.eye(n)*epsilon)

	(x,u,G,theta)= polytraj.polytopic_trajectory_given_modes(x0,list_of_cells,goal,eps=1,order=1)

	epsilon2 = 0.001
	g_add = epsilon2 * np.identity(n)
	G_inv = []
	for g0 in G:
		g0 += g_add
		G_inv.append(np.linalg.inv(g0))


	if SAVE_OUTPUT:
		output = {"x": x, "u":u, "G": G, "theta": theta, "G_inv":G_inv, "list_of_cells":list_of_cells}
		pickle.dump( output, open(file_name+"_tube_output.p","wb"))
