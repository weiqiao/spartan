from __future__ import absolute_import, division, print_function
from itertools import islice, chain
from collections import namedtuple
import time
import numpy as np
import pydrake.solvers.mathematicalprogram as mp
from pydrake.solvers.ipopt import IpoptSolver
import pydrake.symbolic as sym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import pickle

# use two fingers to rotate carrot while considering friction between both fingers and carrot
# two contact points can move
# finger 1 initial and final contact points are the same 
# different from example 8: add cost function

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

class TrajectoryOptimization(mp.MathematicalProgram):
	def add_dynamics_constraints(self, params, pos_init, pos_final):
		T,dt = params
		T = int(T)
		n = len(pos_init)
		# F = [F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi, omega]
		# F1_tp: friction of finger 1 in positive (roughly x) direction
		# F1_tm: friction of finger 2 in negative (roughly x) direction
		# F2_tp: friction of finger 1 in positive (roughly y) direction
		# F2_tm: friction of finger 2 in negative (roughly y) direction
		# gamma1: slack variable for v1
		# v1: velocity of d (negative direction)
		# d: distance from finger 1 contact point to geometric center of semicircle
		F = self.NewContinuousVariables(12,'F_%d' % 0) 
		F_over_time = F
		for t in range(1,T+1):
			F = self.NewContinuousVariables(12,'F_%d' % t) 
			F_over_time = np.vstack((F_over_time,F)) 

		pos = self.NewContinuousVariables(7, 'pos_%d' % 0) # pos = [x, y, theta, x_dot, y_dot, theta_dot, d]
		pos_over_time = pos
		for t in range(1,T+1):
			pos = self.NewContinuousVariables(7,'pos_%d' % t)
			pos_over_time = np.vstack((pos_over_time,pos))

		for t in range(T):
			x,y,theta,x_dot,y_dot,theta_dot,d = pos_over_time[t,:]
			x_next, y_next, theta_next, x_dot_next, y_dot_next, theta_dot_next,d_next = pos_over_time[t+1,:]
			F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi, omega = F_over_time[t,:]
			
			# position update constraints
			self.AddLinearConstraint(x_next - (x+x_dot*dt) <= DynamicsConstraintEps)
			self.AddLinearConstraint(x_next - (x+x_dot*dt) >= -DynamicsConstraintEps)
			self.AddLinearConstraint(y_next - (y+y_dot*dt) <= DynamicsConstraintEps)
			self.AddLinearConstraint(y_next - (y+y_dot*dt) >= -DynamicsConstraintEps)
			self.AddLinearConstraint(theta_next - (theta+theta_dot*dt) <= DynamicsConstraintEps)
			self.AddLinearConstraint(theta_next - (theta+theta_dot*dt) >= -DynamicsConstraintEps)
			
			# force constraints
			x_ddot = (F1*sym.sin(theta) + Ft) / mass
			x_ddot += (F1_tp*sym.cos(theta) - F1_tm*sym.cos(theta)) / mass # friction on finger 1
			x_ddot += - F2*sym.sin(phi) / mass # F2, angle = pi/2 - phi
			x_ddot += (F2_tp*sym.cos(phi) - F2_tm*sym.cos(phi)) / mass # angle = phi 
			y_ddot = (-F1*sym.cos(theta) + Fn - mass*g) / mass
			y_ddot += (F1_tp*sym.sin(theta) - F1_tm*sym.sin(theta)) / mass # friction on finger 1
			y_ddot += F2*sym.cos(phi) / mass
			y_ddot += (F2_tp*sym.sin(phi) - F2_tm*sym.sin(phi)) / mass
			self.AddConstraint(x_dot_next - (x_dot + x_ddot*dt) <= DynamicsConstraintEps)
			self.AddConstraint(x_dot_next - (x_dot + x_ddot*dt) >= -DynamicsConstraintEps)
			self.AddConstraint(y_dot_next - (y_dot + y_ddot*dt) <= DynamicsConstraintEps)
			self.AddConstraint(y_dot_next - (y_dot + y_ddot*dt) >= -DynamicsConstraintEps)

			# torque constraints
			tor_F1 = F1*d 
			tor_Fn = -Fn*(DistanceCentroidToCoM*sym.sin(theta))
			tor_Ft = Ft*(r-DistanceCentroidToCoM*sym.cos(theta))
			tor_F1t = -F1_tp*DistanceCentroidToCoM+F1_tm*DistanceCentroidToCoM
			tor_F2 = F2*DistanceCentroidToCoM*sym.sin(phi-theta)
			tor_F2t = (F2_tp-F2_tm)*(r-DistanceCentroidToCoM*sym.cos(theta-phi))
			theta_ddot = (tor_F1 + tor_Fn + tor_Ft) / inertia
			theta_ddot += tor_F1t / inertia
			theta_ddot += tor_F2 / inertia 
			theta_ddot += tor_F2t / inertia
			self.AddConstraint(theta_dot_next - (theta_dot + theta_ddot*dt) <= DynamicsConstraintEps)
			self.AddConstraint(theta_dot_next - (theta_dot + theta_ddot*dt) >= -DynamicsConstraintEps)
			
			# not penetrating ground constraints
			self.AddConstraint(y + DistanceCentroidToCoM*sym.cos(theta) - r <= DynamicsConstraintEps)
			self.AddConstraint(y + DistanceCentroidToCoM*sym.cos(theta) - r >= -DynamicsConstraintEps)

			# carrot-ground rolling only constraints
			self.AddConstraint(x - DistanceCentroidToCoM*sym.sin(theta) + r*theta <= DynamicsConstraintEps)
			self.AddConstraint(x - DistanceCentroidToCoM*sym.sin(theta) + r*theta >= -DynamicsConstraintEps)

			# ground friction cone constraints 
			self.AddLinearConstraint(Ft <= mu_ground*Fn)
			self.AddLinearConstraint(Ft >= -mu_ground*Fn)

			# basic force constraints
			self.AddLinearConstraint(F1 >= 0)
			self.AddLinearConstraint(Fn >= 0)
			self.AddLinearConstraint(F1 <= MaxInputForce)
			self.AddLinearConstraint(Fn <= MaxInputForce)

			# finger 1 contact point position update
			# note that positive d direction and positive x axis are opposite, so d+ = d - v1*dt
			self.AddLinearConstraint(d_next - (d-v1*dt) <= DynamicsConstraintEps)
			self.AddLinearConstraint(d_next - (d-v1*dt) >= -DynamicsConstraintEps)

			# finger 1 contact point constraint 
			self.AddLinearConstraint(d >= PositionConstraintEps)
			self.AddLinearConstraint(d <= r-PositionConstraintEps)

			# finger 2 contact point constraint
			self.AddConstraint(d*sym.sin(phi-theta)+r-omega <= DynamicsConstraintEps)
			self.AddConstraint(d*sym.sin(phi-theta)+r-omega >= -DynamicsConstraintEps)

			# finger 2 friction cone constraint
			self.AddLinearConstraint(F2_tp>=0)
			self.AddLinearConstraint(F2_tm>=0)
			self.AddLinearConstraint(F2_tp <= mu_finger*F2)
			self.AddLinearConstraint(F2_tm <= mu_finger*F2)

			# LCP on contact between finger 1 and carrot
			alpha = self.NewContinuousVariables(3, 'alpha_%d' % t)
			self.AddLinearConstraint(alpha[0] == mu_finger*F1 - F1_tp - F1_tm)
			self.AddLinearConstraint(alpha[1] == gamma1+v1)
			self.AddLinearConstraint(alpha[2] == gamma1-v1)
			self.AddConstraint(alpha[0]*gamma1 <= OptimizationSlackEps)
			self.AddConstraint(alpha[1]*F1_tp <= OptimizationSlackEps)
			self.AddConstraint(alpha[2]*F1_tm <= OptimizationSlackEps)
			self.AddLinearConstraint(alpha[0] >= 0)
			self.AddLinearConstraint(alpha[1] >= 0)
			self.AddLinearConstraint(alpha[2] >= 0)
			self.AddLinearConstraint(gamma1 >= 0)
			self.AddLinearConstraint(F1_tp >= 0)
			self.AddLinearConstraint(F1_tm >= 0)

			# state bounds
			for i in range(6):
				self.AddLinearConstraint(pos_over_time[t,i] >= StateBound[0,i])
				self.AddLinearConstraint(pos_over_time[t,i] <= StateBound[1,i])

			# control bounds
			self.AddLinearConstraint(F1_tm <= MaxInputForce)
			self.AddLinearConstraint(F1_tp <= MaxInputForce)
			self.AddLinearConstraint(F2_tm <= MaxInputForce)
			self.AddLinearConstraint(F2_tp <= MaxInputForce)
			self.AddLinearConstraint(gamma1 <= MaxRelVel)
			self.AddLinearConstraint(gamma1 >= -MaxRelVel)
			self.AddLinearConstraint(v1 <= MaxRelVel)
			self.AddLinearConstraint(v1 >= -MaxRelVel)
			self.AddLinearConstraint(F2 <= MaxInputForce)
			self.AddLinearConstraint(F2 >= 0)
			self.AddLinearConstraint(phi >= theta)
			self.AddLinearConstraint(phi - theta <= np.pi/2)
			self.AddLinearConstraint(phi >= np.pi/3)
			self.AddLinearConstraint(phi <= np.pi*2/3)

			# phi shouldn't change too much
			PhiChangeMax = np.pi*2/180
			PhiChangeMin = np.pi*0.3/180
			if t > 0:
				phi_prev = F_over_time[t-1,-2]
				self.AddLinearConstraint(phi-phi_prev <= PhiChangeMax)
				self.AddLinearConstraint(phi_prev-phi <= PhiChangeMax)
				#self.AddConstraint((phi-phi_prev)*(phi-phi_prev) >= PhiChangeMin**2)
			if t == T-1:
				self.AddLinearConstraint(phi - pos_final[2] <= DynamicsConstraintEps)
			# if t == 0:
			# 	self.AddLinearConstraint(phi <= np.pi/2.0+2.0/180.0*np.pi)
			# 	self.AddLinearConstraint(phi >= np.pi/2.0-2.0/180.0*np.pi)

			if t > 0:
				theta_prev = pos_over_time[t-1,2]
				self.AddLinearConstraint(theta - theta_prev >= 0)


			# the right finger does not touch the ground
			y_centroid = y + DistanceCentroidToCoM*sym.cos(theta)
			y_F1 = y_centroid-d*sym.sin(theta)
			self.AddConstraint(y_F1 - (omega+FingerWidth)*sym.cos(phi) >= 0)

			# add cost
			self.AddLinearCost((phi-theta))

		# initial state constraint
		for i in range(n):
			self.AddLinearConstraint(pos_over_time[0,i]==pos_init[i])
		# final state constraint
		for i in range(n):
			self.AddLinearConstraint(pos_over_time[-1,i]==pos_final[i])
					

		# initial guess
		if USE_GOOD_INITIAL_GUESS:
			pass 
		else:
			for t in range(T+1):
				for i in range(n):
					self.SetInitialGuess(pos_over_time[t,i], pos_init[i])
			for t in range(T):
				for i in range(7):
					self.SetInitialGuess(F_over_time[t,i],0)

		return pos_over_time, F_over_time

	def get_solution(self,x):
		try:
			return self.GetSolution(x)
		except TypeError:
			return x



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
    fig.savefig('trajopt_example9_fig_latest/carrot_%d.png'%t, dpi=100)
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


if __name__=="__main__":
	T = 100
	dt = 0.01

	prog1 = TrajectoryOptimization()
	params = np.array([T,dt])
	pos_init = np.array([0,r-DistanceCentroidToCoM,0,0,0,0,0.018])
	theta_final = np.pi/2
	d_final = 0.018 # 0.012 also works, took 22 sec to solve
	pos_final = np.array([DistanceCentroidToCoM*np.sin(theta_final)-r*theta_final,r-DistanceCentroidToCoM*np.cos(theta_final),theta_final,0,0,0,d_final])
	initial_guess = {}
	pos_over_time_var, F_over_time_var = prog1.add_dynamics_constraints(params, pos_init, pos_final)
	solver = IpoptSolver()
	# # Ipopt solver options: https://www.coin-or.org/Ipopt/documentation/node40.html
	# prog1.SetSolverOption(mp.SolverType.kIpopt, "max_iter", 10000)
	start_time = time.time()
	result = solver.Solve(prog1)
	# # if use snopt solver
	# # Snopt solver options: http://www.ccom.ucsd.edu/~peg/papers/sndoc7.pdf
	# prog1.SetSolverOption(mp.SolverType.kSnopt, "Major iterations limit", 10000)
	# prog1.SetSolverOption(mp.SolverType.kSnopt, "Print file", "iterations_info_log.txt")
	# result = prog1.Solve()
	solve_time = time.time() - start_time
	print(result)
	assert result == mp.SolutionResult.kSolutionFound
	print(solve_time)
	pos_over_time = prog1.get_solution(pos_over_time_var)
	F_over_time = prog1.get_solution(F_over_time_var)
	if VISUALIZE:
		for t in range(T):
			visualize(pos_over_time[t,:],F_over_time[t,:],t)

	if SAVE_STATE_AND_CONTROL:
		params_save = np.array([11])
		params_save = np.append(params_save,np.array([mass, inertia, DistanceCentroidToCoM, r, dt, 
			DynamicsConstraintEps,PositionConstraintEps,mu_ground,mu_finger,MaxInputForce,MaxRelVel]))
		params_save = np.append(params_save,StateBound[0,:])
		params_save = np.append(params_save,StateBound[1,:])
		params_save = np.append(params_save,pos_init)
		params_save = np.append(params_save,pos_final)
		params_save = np.append(params_save,[T])
		state_and_control = {"state": pos_over_time, "control":F_over_time, "params": params_save}
		pickle.dump( state_and_control, open("trajopt_example9_latest.p","wb"))
