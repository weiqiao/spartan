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
from pwa_system import system 

if __name__=="__main__":

	state_and_control = pickle.load(open("example_7_sol.p","rb"))
	pos_over_time = state_and_control["state"]
	F_over_time = state_and_control["control"]
	params = state_and_control["state_and_control"]
	A1,B1,c1,H1,h1 = calin1.linearize(pos_over_time[0,:], F_over_time[0,:], params)
	A2,B2,c2,H2,h2 = calin2.linearize(pos_over_time[0,:], F_over_time[0,:], params)
	A3,B3,c3,H3,h3 = calin3.linearize(pos_over_time[0,:], F_over_time[0,:], params)
	sys = system()
	sys.A[0,0] = A1
	sys.B[0,0] = B1
	sys.c[0,0] = c1
	sys.C[0,0] = polytope(H1,h1)

	sys.A[0,1] = A2
	sys.B[0,1] = B2
	sys.c[0,1] = c2
	sys.C[0,1] = polytope(H2,h2)

	sys.A[0,2] = A3
	sys.B[0,2] = B3
	sys.c[0,2] = c3
	sys.C[0,2] = polytope(H3,h3)

	(x,u,G,theta)= polytraj.polytopic_trajectory_given_modes(x0,list_of_cells,sys.goal,eps=1,order=1,scale=sys.scale)
