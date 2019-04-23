# -*- coding: utf-8 -*-

# linearization for trajopt_example13
import numpy as np 
g = 9.8
d = 0.018 

def linearize(X,F,params):
	x,y,theta,x_dot,y_dot,theta_dot = X
	F1, F1t, F2, F2t, Fn, Ft, phi, omega = F
	idx = int(params[0])
	m, I, DistanceCentroidToCoM, r, dt, DynamicsConstraintEps,PositionConstraintEps,mu_ground,mu_finger,MaxInputForce,MaxRelVel = params[1:idx+1] # mass, inertia

	A = np.identity(len(X))
	B = np.zeros((len(X),len(F)))
	c = np.zeros((len(X),1))

	A[0,3] = dt 
	A[1,4] = dt 
	A[2,5] = dt 
	
	# x_ddot
	coeff_x, coeff_u, coeff_c = df(X,F,params)
	A[3,:] += coeff_x*dt 
	B[3,:] += coeff_u*dt 
	c[3] += coeff_c*dt 

	# y_ddot
	coeff_x, coeff_u, coeff_c = dg(X,F,params)
	A[4,:] += coeff_x*dt 
	B[4,:] += coeff_u*dt 
	c[4] += coeff_c*dt 

	# theta_ddot
	coeff_x, coeff_u, coeff_c = dh(X,F,params)
	A[5,:] += coeff_x*dt 
	B[5,:] += coeff_u*dt 
	c[5] += coeff_c*dt 


	H,h = constraints(X,F,params)

	# # change linearization model
	# n_ = len(X)
	# m_ = len(F)
	# A_new = np.zeros((n_+2, n_+2))
	# A_new[:n_,:n_] = A
	# B_new = np.zeros((n_+2,m_))
	# B_new[:n_,:m_] = B 
	# B_new[n_,-2] = 1
	# B_new[n_+1,-1] = 1
	# c_new = np.zeros((n_+2,1))
	# c_new[:n_,:] = c
	# h_n, h_m = H.shape
	# H_new = np.zeros((h_n,h_m+2))
	# H_new[:,:h_m] = H[:,:h_m]
	# H_new[:,h_m+2:] = H[:,h_m:]

	# return A_new, B_new, c_new, H_new, h
	return A, B, c, H, h

def df(X,F,params):
	# f = x_ddot
	# compute the linearization of f around the current point (X,F)
	x,y,theta,x_dot,y_dot,theta_dot = X
	F1, F1t, F2, F2t, Fn, Ft, phi, omega = F
	idx = int(params[0])
	m, I, DistanceCentroidToCoM, r, dt, DynamicsConstraintEps,PositionConstraintEps,mu_ground,mu_finger,MaxInputForce,MaxRelVel = params[1:idx+1] # mass, inertia
	
	df_dx = 0
	df_dy = 0
	df_dtheta = (F1*np.cos(theta)-F1t*np.sin(theta))/m 
	df_dx_dot = 0
	df_dy_dot = 0
	df_dtheta_dot = 0
	coeff_x = np.array([df_dx,df_dy,df_dtheta,df_dx_dot,df_dy_dot,df_dtheta_dot])

	df_dF1 = np.sin(theta) / m
	df_dF1t = np.cos(theta) / m
	df_dF2 = -np.sin(phi) / m
	df_dF2t = np.cos(phi) / m
	df_dFn = 0
	df_dFt = 1 / m
	df_dphi = (-F2*np.cos(phi) - F2t*np.sin(phi)) / m
	df_dw = 0
	coeff_u = np.array([df_dF1,df_dF1t,df_dF2,df_dF2t,df_dFn,df_dFt,df_dphi,df_dw])

	# constant term
	coeff_c = (F1*np.sin(theta) + F1t*np.cos(theta) - F2*np.sin(phi) + F2t*np.cos(phi) + Ft) / m
	coeff_c -= coeff_x.dot(X) + coeff_u.dot(F)

	return coeff_x, coeff_u, coeff_c 

def dg(X,F,params):
	# g = y_ddot
	# compute the linearization of g around the current point (X,F)
	x,y,theta,x_dot,y_dot,theta_dot = X
	F1, F1t, F2, F2t, Fn, Ft, phi, omega = F
	idx = int(params[0])
	m, I, DistanceCentroidToCoM, r, dt, DynamicsConstraintEps,PositionConstraintEps,mu_ground,mu_finger,MaxInputForce,MaxRelVel = params[1:idx+1] # mass, inertia
	
	dg_dx = 0
	dg_dy = 0
	dg_dtheta = (F1*np.sin(theta)+F1t*np.cos(theta))/m 
	dg_dx_dot = 0
	dg_dy_dot = 0
	dg_dtheta_dot = 0
	coeff_x = np.array([dg_dx,dg_dy,dg_dtheta,dg_dx_dot,dg_dy_dot,dg_dtheta_dot])

	dg_dF1 = -np.cos(theta) / m
	dg_dF1t = np.sin(theta) / m
	dg_dF2 = np.cos(phi) / m
	dg_dF2t = np.sin(phi) / m 
	dg_dFn = 1/m
	dg_dFt = 0
	dg_dphi = (-F2*np.sin(phi) + F2t*np.cos(phi)) / m
	dg_dw = 0
	coeff_u = np.array([dg_dF1,dg_dF1t,dg_dF2,dg_dF2t,dg_dFn,dg_dFt,dg_dphi,dg_dw])

	# constant term
	coeff_c = (-F1*np.cos(theta) + F1t*np.sin(theta) + F2*np.cos(phi) + F2t*np.sin(phi) + Fn - m*g) / m
	coeff_c -= coeff_x.dot(X) + coeff_u.dot(F)

	return coeff_x, coeff_u, coeff_c


def dh(X,F,params):
	# h = theta_ddot
	# compute the linearization of h around the current point (X,F)
	x,y,theta,x_dot,y_dot,theta_dot = X
	F1, F1t, F2, F2t, Fn, Ft, phi, omega = F
	idx = int(params[0])
	m, I, DistanceCentroidToCoM, r, dt, DynamicsConstraintEps,PositionConstraintEps,mu_ground,mu_finger,MaxInputForce,MaxRelVel = params[1:idx+1] # mass, inertia
	r0 = DistanceCentroidToCoM

	dh_dx = 0
	dh_dy = 0
	dh_dtheta = (-Fn*r0*np.cos(theta)+Ft*r0*np.sin(theta)-F2*r0*np.cos(phi-theta)+F2t*r0*np.sin(theta-phi))/I 
	dh_dx_dot = 0
	dh_dy_dot = 0
	dh_dtheta_dot = 0
	coeff_x = np.array([dh_dx,dh_dy,dh_dtheta,dh_dx_dot,dh_dy_dot,dh_dtheta_dot])

	dh_dF1 = d/I
	dh_dF1t = -r0/I
	dh_dF2 = r0*np.sin(phi-theta)/I 
	dh_dF2t = (r-r0*np.cos(theta-phi))/I
	dh_dFn = -r0*np.sin(theta)/I 
	dh_dFt = (r-r0*np.cos(theta))/I
	dh_dphi = (F2*r0*np.cos(phi-theta)+F2t*r0*np.sin(phi-theta))/I 
	dh_dw = 0
	coeff_u = np.array([dh_dF1,dh_dF1t,dh_dF2,dh_dF2t,dh_dFn,dh_dFt,dh_dphi,dh_dw])

	# constant term
	coeff_c = (F1*d - F1t*r0 + F2*r0*np.sin(phi-theta) + F2t*(r-r0*np.cos(theta-phi)) - Fn*r0*np.sin(theta) + Ft*(r-r0*np.cos(theta)))/I
	coeff_c -= coeff_x.dot(X) + coeff_u.dot(F)

	return coeff_x, coeff_u, coeff_c

def constraints(X,F,params):
	#0,1,2,   3,    4,    5,       
	x,y,theta,x_dot,y_dot,theta_dot = X
	#6, 7,   8,  9,   10, 11, 12,  13
	F1, F1t, F2, F2t, Fn, Ft, phi, omega = F
	idx = int(params[0])
	m, I, DistanceCentroidToCoM, r, dt, DynamicsConstraintEps,PositionConstraintEps,mu_ground,mu_finger,MaxInputForce,MaxRelVel = params[1:idx+1] # mass, inertia
	StateBound = np.array([np.array(params[idx+1:idx+7]),np.array(params[idx+7:idx+13])])
	r0 = DistanceCentroidToCoM # alias

	# # ground friction cone constraints 
	# self.AddLinearConstraint(Ft <= mu_ground*Fn)
	# self.AddLinearConstraint(Ft >= -mu_ground*Fn)

	# # finger 1 friction cone constraint
	# self.AddLinearConstraint(F1t <= mu_finger*F1)
	# self.AddLinearConstraint(F1t >= -mu_finger*F1)

	# # control bounds
	# self.AddLinearConstraint(F1 >= 0)
	# self.AddLinearConstraint(F1 <= MaxInputForce)
	# self.AddLinearConstraint(Fn >= 0)
	# self.AddLinearConstraint(Fn <= MaxInputForce)

	# # state bounds
	# for i in range(6):
	# 	self.AddLinearConstraint(pos_over_time[t,i] >= StateBound[0,i])
	# 	self.AddLinearConstraint(pos_over_time[t,i] <= StateBound[1,i])

	# the output is of the form H [X,U] <= h
	len_h = len(X)+len(F)
	# constraint 4: general constraints
	# ground friction cone constraints 
	# -mu_g * Fn - Ft <= 0
	cons = np.zeros(len_h)
	cons[10] = -mu_ground
	cons[11] = -1
	h2 = 0
	# H = np.vstack((H, cons))
	# h = np.vstack((h, h2))
	H = cons
	h = h2

	# -mu_g * Fn + Ft <= 0
	cons = np.zeros(len_h)
	cons[10] = -mu_ground
	cons[11] = 1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# finger 1 friction cone constraint
	# -mu_g * F1 - F1t <= 0
	cons = np.zeros(len_h)
	cons[6] = -mu_ground
	cons[7] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# -mu_g * F1 + F1t <= 0
	cons = np.zeros(len_h)
	cons[6] = -mu_ground
	cons[7] = 1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# finger 2 friction cone constraint
	# -mu_g * F2 - F2t <= 0
	cons = np.zeros(len_h)
	cons[8] = -mu_ground
	cons[9] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# -mu_g * F2 + F2t <= 0
	cons = np.zeros(len_h)
	cons[8] = -mu_ground
	cons[9] = 1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# control bounds
	# -F1 <= 0
	cons = np.zeros(len_h)
	cons[6] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# F1 <= MaxInputForce
	cons = np.zeros(len_h)
	cons[6] = 1
	h2 = MaxInputForce
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# -F2 <= 0
	cons = np.zeros(len_h)
	cons[8] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# F2 <= MaxInputForce
	cons = np.zeros(len_h)
	cons[8] = 1
	h2 = MaxInputForce
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# -Fn <= 0
	cons = np.zeros(len_h)
	cons[10] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# Fn <= MaxInputForce
	cons = np.zeros(len_h)
	cons[10] = 1
	h2 = MaxInputForce
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# theta - phi <= 0
	cons = np.zeros(len_h)
	cons[2] = 1
	cons[12] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# phi - theta <= np.pi/2
	cons = np.zeros(len_h)
	cons[2] = -1
	cons[12] = 1
	h2 = np.pi/2
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# -phi <= -np.pi*5/12
	cons = np.zeros(len_h)
	cons[12] = -1
	h2 = -np.pi*5/12
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# phi <= np.pi*2/3
	cons = np.zeros(len_h)
	cons[12] = 1
	h2 = np.pi*2/3
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# omega <= 0.01 (hard coded)
	cons = np.zeros(len_h)
	cons[13] = 1
	h2 = 0.01
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# omega >= 0
	cons = np.zeros(len_h)
	cons[13] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# state bounds
	for i in range(6):
		cons = np.zeros(len_h)
		cons[i] = -1
		h2 = -StateBound[0,i]
		H = np.vstack((H, cons))
		h = np.vstack((h, h2))

		cons = np.zeros(len_h)
		cons[i] = 1
		h2 = StateBound[1,i]
		H = np.vstack((H, cons))
		h = np.vstack((h, h2))

	return H,h