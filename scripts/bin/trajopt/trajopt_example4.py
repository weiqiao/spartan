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

# use two fingers to rotate carrot 30 degrees without considering friction between finger and carrot
# assume that the two contact points are both fixed

DynamicsConstraintEps = 0.0001
mu_ground = 0.2
g = 9.8
r = 1
DistanceCentroidToCoM = 4*r/(3*np.pi)
MaxInputForce = 100
VISUALIZE = 0
StateBound = np.array([[-4,-1,-np.pi,-2,-2,-2],[4,1,np.pi,2,2,2]])
class TrajectoryOptimization(mp.MathematicalProgram):
	def add_dynamics_constraints(self, params, pos_init, pos_final):
		T,dt,d,mass,inertia = params
		T = int(T)
		
		n = len(pos_init)
		F = self.NewContinuousVariables(3,'F_%d' % 0) # F = [F1, Fn, Ft]
		F_over_time = F
		for t in range(1,T+1):
			F = self.NewContinuousVariables(3,'F_%d' % t) 
			F_over_time = np.vstack((F_over_time,F)) 

		pos = self.NewContinuousVariables(6, 'pos_%d' % 0) # pos = [x, y, theta, x_dot, y_dot, theta_dot]
		pos_over_time = pos
		for t in range(1,T+1):
			pos = self.NewContinuousVariables(6,'pos_%d' % t)
			pos_over_time = np.vstack((pos_over_time,pos))

		for t in range(T):
			x,y,theta,x_dot,y_dot,theta_dot = pos_over_time[t,:]
			x_next, y_next, theta_next, x_dot_next, y_dot_next, theta_dot_next = pos_over_time[t+1,:]
			F1, Fn, Ft = F_over_time[t,:]
			
			# position update constraints
			self.AddLinearConstraint(x_next - (x+x_dot*dt) <= DynamicsConstraintEps)
			self.AddLinearConstraint(x_next - (x+x_dot*dt) >= -DynamicsConstraintEps)
			self.AddLinearConstraint(y_next - (y+y_dot*dt) <= DynamicsConstraintEps)
			self.AddLinearConstraint(y_next - (y+y_dot*dt) >= -DynamicsConstraintEps)
			self.AddLinearConstraint(theta_next - (theta+theta_dot*dt) <= DynamicsConstraintEps)
			self.AddLinearConstraint(theta_next - (theta+theta_dot*dt) >= -DynamicsConstraintEps)
			
			# force constraints
			x_ddot = (F1*sym.sin(theta) + Ft) / mass
			y_ddot = (-F1*sym.cos(theta) + Fn - mass*g) / mass
			self.AddConstraint(x_dot_next - (x_dot + x_ddot*dt) <= DynamicsConstraintEps)
			self.AddConstraint(x_dot_next - (x_dot + x_ddot*dt) >= -DynamicsConstraintEps)
			
			# torque constraints
			tor_F1 = F1*d 
			tor_Fn = -Fn*(DistanceCentroidToCoM*sym.sin(theta))
			tor_Ft = Ft*(r-DistanceCentroidToCoM*sym.cos(theta))
			theta_ddot = (tor_F1 + tor_Fn + tor_Ft) / inertia
			self.AddConstraint(theta_dot_next - (theta_dot + theta_ddot*dt) <= DynamicsConstraintEps)
			self.AddConstraint(theta_dot_next - (theta_dot + theta_ddot*dt) >= -DynamicsConstraintEps)
			
			# not penetrating ground constraints
			self.AddConstraint(y + DistanceCentroidToCoM*sym.cos(theta) - r <= DynamicsConstraintEps)
			self.AddConstraint(y + DistanceCentroidToCoM*sym.cos(theta) - r >= -DynamicsConstraintEps)

			# carrot-ground rolling only constraints
			self.AddConstraint(x + r*theta <= DynamicsConstraintEps)
			self.AddConstraint(x + r*theta >= -DynamicsConstraintEps)

			# ground friction cone constraints 
			self.AddLinearConstraint(Ft <= mu_ground*Fn)
			self.AddLinearConstraint(Ft >= -mu_ground*Fn)

			# basic force constraints
			self.AddLinearConstraint(F1 >= 0)
			self.AddLinearConstraint(Fn >= 0)
			self.AddLinearConstraint(F1 <= MaxInputForce)
			self.AddLinearConstraint(Fn <= MaxInputForce)

			# state bounds
			for i in range(6):
				self.AddLinearConstraint(pos_over_time[t,i] >= StateBound[0,i])
				self.AddLinearConstraint(pos_over_time[t,i] <= StateBound[1,i])

		# initial state constraint
		for i in range(n):
			self.AddLinearConstraint(pos_over_time[0,i]==pos_init[i])
		# final state constraint
		for i in range(n):
			self.AddLinearConstraint(pos_over_time[-1,i]==pos_final[i])
		
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
    ax1.set_xlim([-4,4])
    ax1.set_ylim([-1,4])
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
    fig.savefig('trajopt_example2_fig/carrot_%d.png' % t, dpi=100)
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
	d = 0.5 # TODO
	force_scaling_factor = 0.01 # TODO
	x,y,theta=X[:3]
	F1,Fn,Ft = F
	x_centroid = x - DistanceCentroidToCoM*np.sin(theta)
	y_centroid = y + DistanceCentroidToCoM*np.cos(theta)

	x_F1 = x_centroid-d*np.cos(theta)
	y_F1 = y_centroid-d*np.sin(theta)
	dx_F1 = np.sin(theta)*F1*force_scaling_factor
	dy_F1 = -np.cos(theta)*F1*force_scaling_factor

	x_Fn = x_centroid
	y_Fn = 0
	dx_Fn = 0
	dy_Fn = Fn*force_scaling_factor

	x_Ft = x_centroid
	y_Ft = 0
	dx_Ft = Ft*force_scaling_factor
	dy_Ft = 0

	ax.arrow(x_F1,y_F1,dx_F1,dy_F1,color=(1,0,1),head_width=0.05, head_length=0.1)
	ax.arrow(x_Fn,y_Fn,dx_Fn,dy_Fn,color=(1,0,1),head_width=0.05, head_length=0.1)
	ax.arrow(x_Ft,y_Ft,dx_Ft,dy_Ft,color=(1,0,1),head_width=0.05, head_length=0.1)

def center_mass(ax,X):
    x,y,theta,x_dot,y_dot,theta_dot=X
    

if __name__=="__main__":
	prog = TrajectoryOptimization()

	T = 50
	dt = 0.01
	d = 0.5
	mass = 1
	inertia = (np.pi/4-8/(9*np.pi))*(2*mass*r**2/np.pi)

	params = np.array([T,dt,d,mass,inertia])

	pos_init = [0,r-DistanceCentroidToCoM,0,0,0,0]
	pos_final = [-r*np.pi/6,r-DistanceCentroidToCoM*np.cos(np.pi/6),np.pi/6,0,0,0]
	pos_over_time_var, F_over_time_var = prog.add_dynamics_constraints(params, pos_init, pos_final)

	solver = IpoptSolver()
	start_time = time.time()
	result = solver.Solve(prog)
	solve_time = time.time() - start_time
	assert result == mp.SolutionResult.kSolutionFound
	print(solve_time)
	
	pos_over_time = prog.get_solution(pos_over_time_var)
	F_over_time = prog.get_solution(F_over_time_var)

	if VISUALIZE:
		for t in range(T):
			visualize(pos_over_time[t,:],F_over_time[t,:],t)
	# plot trajectory
	# plt.figure()
	# plt.plot(pos_over_time[:,0],pos_over_time[:,1])
	# plt.xlabel('x')
	# plt.ylabel('y')
	# plt.show()