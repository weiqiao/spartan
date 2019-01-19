import pickle
import numpy as np
import scipy as sp
from gurobipy import Model,GRB,LinExpr
import time as time
from pypolycontain.lib.containment_encodings import subset_LP,subset_zonotopes
from pypolycontain.lib.polytope import polytope
from pypolycontain.lib.zonotope import zonotope

import carrot_linearization_1 as calin1 
import carrot_linearization_2 as calin2
import carrot_linearization_3 as calin3
import poly_trajectory as polytraj 
from pwa_system import system, linear_cell
import pickle


SAVE_OUTPUT = 1
if __name__=="__main__":
	file_name = "example_8sol_T200_nolowerboundondphi"
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
		params_save = np.append(params_save,pos_init)												    # idx+13~idx+19
		params_save = np.append(params_save,pos_final)												 	# idx+20~idx+26
		params_save = np.append(params_save,[T])														# idx+27
	'''
	idx = int(params[0])
	T = int(params[idx+27])
	t0 = int(T/4*2)
	list_of_cells = []
	for t in range(t0,T):
		v1 = F_over_time[t,4]
		# if np.abs(v1) < 0.001:
		# 	A,B,c,H,h = calin1.linearize(pos_over_time[t,:], F_over_time[t,:], params)
		# elif v1 > 0.001:
		# 	A,B,c,H,h = calin2.linearize(pos_over_time[t,:], F_over_time[t,:], params)
		# else:
		# 	A,B,c,H,h = calin3.linearize(pos_over_time[t,:], F_over_time[t,:], params)
		if v1 > 0:
			A,B,c,H,h = calin2.linearize(pos_over_time[t,:], F_over_time[t,:], params)
		else:
			A,B,c,H,h = calin3.linearize(pos_over_time[t,:], F_over_time[t,:], params)
		list_of_cells.append(linear_cell(A,B,c,polytope(H,h)))
		# A1,B1,c1,H1,h1 = calin1.linearize(pos_over_time[0,:], F_over_time[0,:], params)
		# A2,B2,c2,H2,h2 = calin2.linearize(pos_over_time[0,:], F_over_time[0,:], params)
		# A3,B3,c3,H3,h3 = calin3.linearize(pos_over_time[0,:], F_over_time[0,:], params)
		# sys = system()
		# sys.A[0,0] = A1
		# sys.B[0,0] = B1
		# sys.c[0,0] = c1
		# sys.C[0,0] = polytope(H1,h1)

		# sys.A[0,1] = A2
		# sys.B[0,1] = B2
		# sys.c[0,1] = c2
		# sys.C[0,1] = polytope(H2,h2)

		# sys.A[0,2] = A3
		# sys.B[0,2] = B3
		# sys.c[0,2] = c3
		# sys.C[0,2] = polytope(H3,h3)

		# sys.build_cells()
		
	pos_init = pos_over_time[t0,:]
	x0 = pos_init.reshape((-1,1))
	pos_final = params[idx+20:idx+27]
	n = 7
	epsilon = 0.01
	goal=zonotope(pos_final.reshape(-1,1),np.eye(n)*epsilon)

	(x,u,G,theta)= polytraj.polytopic_trajectory_given_modes(x0,list_of_cells,goal,eps=1,order=1)

	if SAVE_OUTPUT:
		output = {"x": x, "u":u, "G": G, "theta": theta}
		pickle.dump( output, open(file_name+"_tube_output.p","wb"))
