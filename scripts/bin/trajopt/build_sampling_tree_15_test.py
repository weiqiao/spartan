import numpy as np
from gurobipy import *
from GurobiModel import GurobiModel
from operator import le, ge, eq

from pwa_system_class import AffineSystem, Domain, PiecewiseAffineSystem, TreeStates

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

import carrot_pwa_15_mode1 as calin1 
import carrot_pwa_15_mode2 as calin2
import poly_trajectory_15 as polytraj 
from pwa_system import system, linear_cell
import pickle



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


if __name__ == "__main__":
	file_name = "trajopt_example15_latest"
	tree_states_read = pickle.load(open(file_name+"_tree_states.p","rb"))
	tree_states = tree_states_read["tree_states"]
	N = len(tree_states)
	for t in range(N):
		print(tree_states[t].x)
		print(tree_states[t].modeseq)