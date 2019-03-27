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
from trajopt.forward_dynamics import forward_dynamics


# this is to simulate the system forward, but the integration error is too large
g = 9.8

"""
Visualization Tools
"""    
def visualize(X,F,Y,t,params,file_name):
    fig,ax1 = plt.subplots()
    ax1.set_xlabel("x",fontsize=20)
    ax1.set_ylabel("y",fontsize=20)
    ax1.set_xlim([-0.2,0.2])
    ax1.set_ylim([-0.2,0.4])
    fig.gca().set_aspect('equal')
    p_list=[]
    r0 = params[3]
    r = params[4]
    v=vertices(X,r0,r)
    p_list.append(patches.Polygon(v, True))
    p=PatchCollection(p_list,color=(1,0,0),alpha=0.31,edgecolor=(1,0,0))
    ax1.add_collection(p)
    ax1.grid(color=(0,0,0), linestyle='--', linewidth=0.3)
    ax1.set_title("carrot %d"%t)
    ax1.plot([-12,12],[0,0],'black')
    ax1.plot(X[0],X[1],'+',color=(1,0,0))# draw CoM
    #draw_force(ax1,X,F) # draw forces
    draw_left_finger(ax1,X,F,Y)
    draw_right_finger(ax1,X,F,Y)
    t += 1
    fig.savefig(file_name + '_animation/carrot_%d.png'%t, dpi=100)
    plt.close()
    return fig

def vertices(X,r0,r,N=50):
    x,y,theta=X[:3]
    x_centroid = x - r0*np.sin(theta)
    y_centroid = y + r0*np.cos(theta)
    v=np.empty((N,2))
    for k in range(50):
        phi=-np.pi/2+np.pi/(N-1)*k
        v[k,0]=x_centroid+r*np.sin(phi+theta)
        v[k,1]=y_centroid-r*np.cos(phi+theta)
    return v


def draw_left_finger(ax, X, F, Y):
	Width = 0.008 # TODO
	Length = .1 # TODO

	# x,y,theta=X[:3]
	# d = X[-1]
	# F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi, omega = F

	# x_centroid = x - DistanceCentroidToCoM*np.sin(theta)
	# y_centroid = y + DistanceCentroidToCoM*np.cos(theta)
	# x_F1 = x_centroid-d*np.cos(theta)
	# y_F1 = y_centroid-d*np.sin(theta)
	x_F1, y_F1, phi, omega = Y
	# counter clock wise counting vertices
	v_x = [x_F1, x_F1 + Length*np.cos(phi), x_F1 + Length*np.cos(phi) + Width*np.cos(phi+np.pi/2), x_F1 + Width*np.cos(phi+np.pi/2)]
	v_y = [y_F1, y_F1 + Length*np.sin(phi), y_F1 + Length*np.sin(phi) + Width*np.sin(phi+np.pi/2), y_F1 + Width*np.sin(phi+np.pi/2)]

	left_finger = [patches.Polygon(np.array([v_x+v_y]).reshape(2,4).T, True)]
	ax.add_collection(PatchCollection(left_finger, color=(0,0,0),alpha=0.8,edgecolor=(0,0,0)))

def draw_right_finger(ax, X, F, Y):
	Width = 0.008 # TODO
	Length = .1 # TODO

	# x,y,theta=X[:3]
	# d = X[-1]
	# F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi, omega = F

	# x_centroid = x - DistanceCentroidToCoM*np.sin(theta)
	# y_centroid = y + DistanceCentroidToCoM*np.cos(theta)
	# x_F1 = x_centroid-d*np.cos(theta)
	# y_F1 = y_centroid-d*np.sin(theta)
	x_F1, y_F1, phi, omega = Y
	x_F1 += (Width+omega)*np.cos(phi-np.pi/2)
	y_F1 += (Width+omega)*np.sin(phi-np.pi/2)

	# counter clock wise counting vertices
	v_x = [x_F1, x_F1 + Length*np.cos(phi), x_F1 + Length*np.cos(phi) + Width*np.cos(phi+np.pi/2), x_F1 + Width*np.cos(phi+np.pi/2)]
	v_y = [y_F1, y_F1 + Length*np.sin(phi), y_F1 + Length*np.sin(phi) + Width*np.sin(phi+np.pi/2), y_F1 + Width*np.sin(phi+np.pi/2)]

	right_finger = [patches.Polygon(np.array([v_x+v_y]).reshape(2,4).T, True)]
	ax.add_collection(PatchCollection(right_finger, color=(0,0,0),alpha=0.8,edgecolor=(0,0,0)))


def simulate_forward(file_name):
    state_and_control = pickle.load(open(file_name + ".p","rb"))
    pos_over_time = state_and_control["state"]
    F_over_time = state_and_control["control"]
    params = state_and_control["params"]

    phi = np.pi/2
    alpha = 0.05
    idx = int(params[0])
    T = int(params[idx+27])
    r0 = params[3]
    r = params[4]

    t = 0

    x,y,theta,x_dot,y_dot,theta_dot,d = pos_over_time[t,:]
    F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi0, omega = F_over_time[t,:]
    x_centroid = x - r0*np.sin(theta)
    y_centroid = y + r0*np.cos(theta)
    x_F1 = x_centroid-d*np.cos(theta)
    y_F1 = y_centroid-d*np.sin(theta)

    # Y = np.array([-d,r,phi0,omega]) # pose of gripper: (x,y,phi,omega)
    Y = np.array([x_F1,y_F1,phi0,omega]) # pose of gripper: (x,y,phi,omega)
    X = pos_over_time[t,:]
    for t in range(T):
		F = F_over_time[t,:]
		X_next = forward_dynamics(X,F,params)
		if t > 0:
		    x,y,theta,x_dot,y_dot,theta_dot,d = pos_over_time[t,:]
		    # x,y,theta,x_dot,y_dot,theta_dot,d = X
		    F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi_next, omega = F_over_time[t,:]
		    x_centroid = x - r0*np.sin(theta)
		    y_centroid = y + r0*np.cos(theta)
		    x_F1_next = x_centroid-d*np.cos(theta)
		    y_F1_next = y_centroid-d*np.sin(theta)
		    dx = x_F1_next - x_F1
		    dy = y_F1_next - y_F1 
		    x_F1 = x_F1_next
		    y_F1 = y_F1_next
		    
		    Y[0] += dx
		    Y[1] += dy
		    Y[2] = phi_next
		    Y[3] = omega

		visualize(X,F,Y,t,params,file_name)
		X = X_next


if __name__ == "__main__":
	file_name = "trajopt_example9_latest"
	simulate_forward(file_name)