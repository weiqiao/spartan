import numpy as np
import scipy as sp
import scipy.linalg
from gurobipy import Model,GRB,LinExpr
import pickle

import time as time

from pypolycontain.lib.containment_encodings import subset_LP,subset_zonotopes
from pypolycontain.lib.polytope import polytope
from pypolycontain.lib.zonotope import zonotope

file_name = "trajopt_example13_latest"
state_and_control = pickle.load(open(file_name + ".p","rb"))
pos_over_time = state_and_control["state"]
F_over_time = state_and_control["control"]
params = state_and_control["params"]

aaa = pickle.load(open(file_name+"_tube_output.p","rb"))
x = aaa['x']
u = aaa['u']

T = 50

for t in range(T):
	print(pos_over_time[t,:])
	print(x[t].reshape(1,-1))	
	assert(np.sum(np.abs(pos_over_time[t,:]-x[t].reshape(1,-1)))<=6e-4)
	assert(np.sum(np.abs(F_over_time[t,:]-u[t].reshape(1,-1)))<=4e-4)
