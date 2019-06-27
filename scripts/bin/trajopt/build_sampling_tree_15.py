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



def build_mip_ch(S, T, Q, R, P, x_goal, norm):
	# input S with S.affine_systems, S.domains
	# S.affine_systems.A, S.affine_systems.B, S.affine_systems.c
	# S.domains.A, S.domains.b

	# create a new model
	model = GurobiModel()

	# 
	[nx, nu, nm] = [S.nx, S.nu, S.nm]
	# nx = 6
	# nu = 7
	# nm = 4
	# T = 7

	obj = 0
	for t in range(T):
		if t == 0:
			x = model.add_variables(nx, name='x0')
		else:
			x = x_next
		x_next = model.add_variables(nx, name='x%d'%(t+1))
		u = model.add_variables(nu, name='u%d'%t)
		d = model.add_variables(nm, vtype=GRB.BINARY, name='d%d'%t)

		# auxiliary continuous variables for the convex hull method
		y = [model.add_variables(nx) for i in range(nm)]
		v = [model.add_variables(nu) for i in range(nm)]
		z = [model.add_variables(nx) for i in range(nm)]

		# constrained dynamics
		for i in range(nm):

			# enforce dynamics
			Si = S.affine_systems[i]
			model.add_linear_constraints(z[i], eq, Si.A.dot(y[i]) + Si.B.dot(v[i]) + Si.c*d[i])

			# enforce state and input constraints
			Di = S.domains[i]
			yvi = np.concatenate((y[i],v[i]))
			model.add_linear_constraints(Di.A.dot(yvi), le, Di.b*d[i])

		# recompose variables
		model.add_linear_constraints(x, eq, sum(y))
		model.add_linear_constraints(u, eq, sum(v))
		model.add_linear_constraints(x_next, eq, sum(z))

		# constraints on the binaries
		model.addConstr(sum(d) == 1)

		# stage cost
		obj += model.add_stage_cost(Q, R, x-x_goal, u, norm)

	# NO TERMINAL CONSTRAINT
	# # terminal constraint
	# for i in range(nm):
	# 	model.add_linear_constraints(X_N.A.dot(z[i]), le, X_N.b * d[i])

	# terminal cost
	obj += model.add_terminal_cost(P, x_next-x_goal, norm)
	model.setObjective(obj)

	return model 

def get_sample_state():
	file_name = "trajopt_example15_latest"
	state_and_control = pickle.load(open(file_name + ".p","rb"))
	pos_over_time = state_and_control["state"]
	t0 = 20 
	x_s = pos_over_time[t0,:]
	x_s[-1] += 0.02
	return x_s 

def weightedL2Square(a,b,w):
    q = a-b
    return (w*q*q).sum()

def closest_state_on_traj(x_s):
	### let me try weighted l2 distance  
	weight = np.array([1.,1.,20.,.01,.01,.01,20.,5.])
	file_name = "trajopt_example15_latest"
	state_and_control = pickle.load(open(file_name + ".p","rb"))
	pos_over_time = state_and_control["state"]
	F_over_time = state_and_control["control"]
	params = state_and_control["params"]

	idx = int(params[0])
	T = int(params[idx+33])
	min_cost = None
	min_idx = None
	for t in range(T):
		xt = pos_over_time[t,:]
		cur_cost = weightedL2Square(x_s,xt,weight)
		if min_cost == None or cur_cost < min_cost:
			min_cost = cur_cost
			min_idx = t 
	return min_idx 

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
    file_name = "trajopt_example15"
    fig.savefig(file_name+'_sampling_tree/carrot_%d.png'%t, dpi=100)
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
	x,y,theta,x_dot,y_dot,theta_dot, phi, omega = X
	F1, F1t, F2, F2t, Fn, Ft, vphi, vomega = F

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

	x_F2 = x_centroid + r*np.sin(phi)
	y_F2 = y_centroid - r*np.cos(phi)
	dx_F2 = -np.sin(phi)*F2*force_scaling_factor
	dy_F2 = np.cos(phi)*F2*force_scaling_factor

	x_F2_tp = x_F2 
	y_F2_tp = y_F2
	dx_F2_tp = F2t*np.cos(phi)*force_scaling_factor
	dy_F2_tp = F2t*np.sin(phi)*force_scaling_factor

	x_G = x 
	y_G = y
	dx_G = 0
	dy_G = -1*mass*g*force_scaling_factor

	ax.arrow(x_F1,y_F1,dx_F1,dy_F1,color=(1,0,1),head_width=0.0005, head_length=0.001)
	ax.arrow(x_F1_tp,y_F1_tp,dx_F1_tp,dy_F1_tp,color=(1,0,1),head_width=0.0005, head_length=0.001)
	ax.arrow(x_Fn,y_Fn,dx_Fn,dy_Fn,color=(1,0,1),head_width=0.0005, head_length=0.001)
	ax.arrow(x_Ft,y_Ft,dx_Ft,dy_Ft,color=(1,0,1),head_width=0.0005, head_length=0.001)
	ax.arrow(x_F2,y_F2,dx_F2,dy_F2,color=(1,0,1),head_width=0.0005, head_length=0.001)
	ax.arrow(x_F2_tp,y_F2_tp,dx_F2_tp,dy_F2_tp,color=(1,0,1),head_width=0.0005, head_length=0.001)
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

	x,y,theta,x_dot,y_dot,theta_dot, phi, omega = X
	F1, F1t, F2, F2t, Fn, Ft, vphi, vomega = F

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

	x,y,theta,x_dot,y_dot,theta_dot, phi, omega = X
	F1, F1t, F2, F2t, Fn, Ft, vphi, vomega = F

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


if __name__ == '__main__':
	# problem parameters
	TT = 12
	nx = 8
	nu = 8
	nm = 2

	file_name = "trajopt_example15_latest"
	state_and_control = pickle.load(open(file_name + ".p","rb"))
	pos_over_time = state_and_control["state"]
	F_over_time = state_and_control["control"]
	params = state_and_control["params"]

	idx = int(params[0])
	T = int(params[idx+33])
	# pick the pwa system at time t
	# sample and compute the closest state
	tree_states = [] 
	nsample = 1
	for i in range(nsample):
		# generate a new sample
		x_s = get_sample_state() # 1-dim vector
		# find the closest state on the trajectory
		t0 = closest_state_on_traj(x_s)

		# use linearization at t0
		A1,B1,c1,H1,h1 = calin1.linearize(pos_over_time[t0,:], F_over_time[t0,:], params)
		c1 = c1[:,0]
		affsys1 = AffineSystem(A1,B1,c1)
		h1 = h1[:,0]
		dom1 = Domain(H1,h1)
		A2,B2,c2,H2,h2 = calin2.linearize(pos_over_time[t0,:], F_over_time[t0,:], params)
		c2 = c2[:,0]
		affsys2 = AffineSystem(A2,B2,c2)
		h2 = h2[:,0]
		dom2 = Domain(H2,h2)
		# and build a pwa system
		S = PiecewiseAffineSystem([affsys1, affsys2],[dom1,dom2])
		Q = np.eye(nx)
		R = np.eye(nu)
		P = 100*np.eye(nx)

		min_cost = None
		min_cost_idx = None
		# try closest goal states
		for tj in range(-10,11):
			tf = np.maximum(np.minimum(t0+tj, T),0)
			x_goal = pos_over_time[tf,:]

			model = build_mip_ch(S, TT, Q, R, P, x_goal, 'two')
			# set initial state
			for k in range(S.nx):
				model.getVarByName('x0[%d]'%k).LB = x_s[k]
				model.getVarByName('x0[%d]'%k).UB = x_s[k]
			model.update()
			# optimize
			model.optimize()

			# for t in range(TT):
			# 	for k in range(2):
			# 		v = model.getVarByName('d'+str(t)+'[%d]'%k)
			# 		print('d'+str(t)+'[%d]=%d'%(k,v.x))

			# print optimal objective
			if min_cost == None or model.objVal < min_cost:
				min_cost = model.objVal
				min_cost_idx = tf 
			print('Obj: %g' % model.objVal)
		# re-solve for the best index
		tf = min_cost_idx
		x_goal = pos_over_time[tf,:]
		# build model
		model = build_mip_ch(S, TT, Q, R, P, x_goal, 'two')
		# set initial state
		for k in range(S.nx):
			model.getVarByName('x0[%d]'%k).LB = x_s[k]
			model.getVarByName('x0[%d]'%k).UB = x_s[k]
		model.update()
		# optimize
		model.optimize()
		# record result
		mode_seq = []
		for t in range(TT):
			k = 0
			v = model.getVarByName('d'+str(t)+'[%d]'%k)
			if v.x == 0:
				mode_seq.append(1)
			else:
				mode_seq.append(0)

		x_save = x_s 
		for t in range(TT):
			tree_states.append(TreeStates(x_save, mode_seq[t:]))
			# update state
			x_save = np.zeros(nx)
			for kk in range(nx):
				v = model.getVarByName('x'+str(t)+'[%d]'%kk)
				x_save[kk] = v.x 
			visualize(x_save,np.zeros(8),t)

	SAVE_OUTPUT = 1
	if SAVE_OUTPUT:
		output = {"tree_states": tree_states}
		pickle.dump( output, open(file_name+"_tree_states.p","wb"))
