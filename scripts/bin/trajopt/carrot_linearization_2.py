# -*- coding: utf-8 -*-

import numpy as np 
g = 9.8

def linearize(X,F,params):
	x,y,theta,x_dot,y_dot,theta_dot,d = X
	F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi, omega = F
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

	# d+ = d - v1*dt
	B[6,4] += -dt 

	H,h = constraints(X,F,params)

	return A, B, c, H, h  

def df(X,F,params):
	# f = x_ddot
	# compute the linearization of f around the current point (X,F)
	x,y,theta,x_dot,y_dot,theta_dot,d = X
	F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi, omega = F
	idx = int(params[0])
	m, I, DistanceCentroidToCoM, r, dt, DynamicsConstraintEps,PositionConstraintEps,mu_ground,mu_finger,MaxInputForce,MaxRelVel = params[1:idx+1] # mass, inertia
	
	df_dx = 0
	df_dy = 0
	df_dtheta = (F1*np.cos(theta)-F1_tp*np.sin(theta)+F1_tm*np.sin(theta))/m 
	df_dx_dot = 0
	df_dy_dot = 0
	df_dtheta_dot = 0
	df_dd = 0
	coeff_x = np.array([df_dx,df_dy,df_dtheta,df_dx_dot,df_dy_dot,df_dtheta_dot,df_dd])

	df_dF1 = np.sin(theta) / m
	df_dF1_tp = np.cos(theta) / m
	df_dF1_tm = -np.cos(theta) / m
	df_dgamma1 = 0
	df_dv1 = 0
	df_dFn = 0
	df_dFt = 1/m
	df_dF2 = -np.sin(phi) / m
	df_dF2_tp = np.cos(phi) / m
	df_dF2_tm = -np.cos(phi) / m 
	df_dphi = (-F2*np.cos(phi)-F2_tp*np.sin(phi)+F2_tm*np.sin(phi))/m 
	df_domega = 0
	coeff_u = np.array([df_dF1,df_dF1_tp,df_dF1_tm,df_dgamma1,df_dv1,df_dFn,df_dFt,df_dF2,df_dF2_tp,df_dF2_tm,df_dphi,df_domega])

	# constant term
	coeff_c = (F1*np.sin(theta) + Ft + F1_tp*np.cos(theta) - F1_tm*np.cos(theta) - F2*np.sin(phi) + F2_tp*np.cos(phi) - F2_tm*np.cos(phi)) / m
	coeff_c -= coeff_x.dot(X) + coeff_u.dot(F)

	return coeff_x, coeff_u, coeff_c 

def dg(X,F,params):
	# g = y_ddot
	# compute the linearization of g around the current point (X,F)
	x,y,theta,x_dot,y_dot,theta_dot,d = X
	F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi, omega = F
	idx = int(params[0])
	m, I, DistanceCentroidToCoM, r, dt, DynamicsConstraintEps,PositionConstraintEps,mu_ground,mu_finger,MaxInputForce,MaxRelVel = params[1:idx+1] # mass, inertia
	
	dg_dx = 0
	dg_dy = 0
	dg_dtheta = (F1*np.sin(theta)+F1_tp*np.cos(theta)-F1_tm*np.cos(theta))/m 
	dg_dx_dot = 0
	dg_dy_dot = 0
	dg_dtheta_dot = 0
	dg_dd = 0
	coeff_x = np.array([dg_dx,dg_dy,dg_dtheta,dg_dx_dot,dg_dy_dot,dg_dtheta_dot,dg_dd])

	dg_dF1 = -np.cos(theta) / m
	dg_dF1_tp = np.sin(theta) / m
	dg_dF1_tm = -np.sin(theta) / m
	dg_dgamma1 = 0
	dg_dv1 = 0
	dg_dFn = 1/m
	dg_dFt = 0
	dg_dF2 = np.cos(phi) / m
	dg_dF2_tp = np.sin(phi) / m
	dg_dF2_tm = -np.sin(phi) / m 
	dg_dphi = (-F2*np.sin(phi)+F2_tp*np.cos(phi)-F2_tm*np.cos(phi))/m 
	dg_domega = 0
	coeff_u = np.array([dg_dF1,dg_dF1_tp,dg_dF1_tm,dg_dgamma1,dg_dv1,dg_dFn,dg_dFt,dg_dF2,dg_dF2_tp,dg_dF2_tm,dg_dphi,dg_domega])

	# constant term
	coeff_c = (-F1*np.cos(theta) + Fn - m*g + F1_tp*np.sin(theta) - F1_tm*np.sin(theta) + F2*np.cos(phi) + F2_tp*np.sin(phi) - F2_tm*np.sin(phi)) / m
	coeff_c -= coeff_x.dot(X) + coeff_u.dot(F)

	return coeff_x, coeff_u, coeff_c


def dh(X,F,params):
	# h = theta_ddot
	# compute the linearization of h around the current point (X,F)
	x,y,theta,x_dot,y_dot,theta_dot,d = X
	F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi, omega = F
	idx = int(params[0])
	m, I, DistanceCentroidToCoM, r, dt, DynamicsConstraintEps,PositionConstraintEps,mu_ground,mu_finger,MaxInputForce,MaxRelVel = params[1:idx+1] # mass, inertia
	r0 = DistanceCentroidToCoM

	dh_dx = 0
	dh_dy = 0
	dh_dtheta = (-Fn*r0*np.cos(theta)+Ft*r0*np.sin(theta)-F2*r0*np.cos(phi-theta)-(F2_tp - F2_tm)*r0*np.sin(phi-theta))/I 
	dh_dx_dot = 0
	dh_dy_dot = 0
	dh_dtheta_dot = 0
	dh_dd = 0
	coeff_x = np.array([dh_dx,dh_dy,dh_dtheta,dh_dx_dot,dh_dy_dot,dh_dtheta_dot,dh_dd])

	dh_dF1 = d/I
	dh_dF1_tp = -r0/I
	dh_dF1_tm = r0/I
	dh_dgamma1 = 0
	dh_dv1 = 0
	dh_dFn = -r0*np.sin(theta)/I 
	dh_dFt = (r-r0*np.cos(theta))/I
	dh_dF2 = r0*np.sin(phi-theta)/I
	dh_dF2_tp = (r-r0*np.cos(phi-theta))/I
	dh_dF2_tm = -(r-r0*np.cos(phi-theta))/I
	dh_dphi = (F2*r0*np.cos(phi-theta) + (F2_tp - F2_tm)*r0*np.sin(phi-theta))/I 
	dh_domega = 0
	coeff_u = np.array([dh_dF1,dh_dF1_tp,dh_dF1_tm,dh_dgamma1,dh_dv1,dh_dFn,dh_dFt,dh_dF2,dh_dF2_tp,dh_dF2_tm,dh_dphi,dh_domega])

	# constant term
	coeff_c = (F1*d -Fn*(r0*np.sin(theta)) + Ft*(r-r0*np.cos(theta)) -F1_tp*r0+F1_tm*r0 + F2*r0*np.sin(phi-theta) + (F2_tp-F2_tm)*(r-r0*np.cos(theta-phi)))/I
	coeff_c -= coeff_x.dot(X) + coeff_u.dot(F)

	return coeff_x, coeff_u, coeff_c

def constraints(X,F,params):
	#0,1,2,   3,    4,    5,        6
	x,y,theta,x_dot,y_dot,theta_dot,d = X
	#7, 8,     9,     10,     11, 12, 13, 14, 15,    16,    17,  18
	F1, F1_tp, F1_tm, gamma1, v1, Fn, Ft, F2, F2_tp, F2_tm, phi, omega = F
	idx = int(params[0])
	m, I, DistanceCentroidToCoM, r, dt, DynamicsConstraintEps,PositionConstraintEps,mu_ground,mu_finger,MaxInputForce,MaxRelVel = params[1:idx+1] # mass, inertia
	StateBound = np.array([np.array(params[idx+1:idx+7]),np.array(params[idx+7:idx+13])])
	r0 = DistanceCentroidToCoM # alias

	# the output is of the form H [X,U] <= h
	len_h = len(X)+len(F)
	# constraint 1: y + r0*cos(theta) - r == 0
	cons = np.zeros(len_h)
	cons[1] = 1
	cons[2] = -r0*np.sin(theta)
	h1 = r0*np.sin(theta)*theta + r0*np.cos(theta) - r 
	H = cons 
	h = -h1 + DynamicsConstraintEps
	H = np.vstack((H, -cons))
	h = np.vstack((h, h1 + DynamicsConstraintEps))

	# constraint 2: x - r0*sin(theta) +r*theta == 0
	cons = np.zeros(len_h)
	cons[0] = 1
	cons[2] = -r0*np.cos(theta) + r 
	h1 = r0*np.cos(theta)*theta - r0*np.sin(theta) 
	H = np.vstack((H,cons))
	h = np.vstack((h, -h1 + DynamicsConstraintEps))
	H = np.vstack((H, -cons))
	h = np.vstack((h, h1 + DynamicsConstraintEps))

	# constraint 3: d*sin(phi-theta) + r - w == 0
	cons = np.zeros(len_h)
	cons[6] = np.sin(phi-theta)
	cons[17] = d*np.cos(phi-theta)
	cons[2] = -d*np.cos(phi-theta)
	cons[18] = -1
	h1 = -np.sin(phi-theta)*d - d*np.cos(phi-theta)*phi + d*np.cos(phi-theta)*theta + omega
	H = np.vstack((H,cons))
	h = np.vstack((h, -h1 + DynamicsConstraintEps))
	H = np.vstack((H, -cons))
	h = np.vstack((h, h1 + DynamicsConstraintEps))

	# constraint 4: general constraints
	# -mu_g * Fn - Ft <= 0
	cons = np.zeros(len_h)
	cons[12] = -mu_ground
	cons[13] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# -mu_g * Fn + Ft <= 0
	cons = np.zeros(len_h)
	cons[12] = -mu_ground
	cons[13] = 1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# -d <= -PosEps
	cons = np.zeros(len_h)
	cons[6] = -1
	h2 = -PositionConstraintEps
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# d <= r - PosEps
	cons = np.zeros(len_h)
	cons[6] = 1
	h2 = r-PositionConstraintEps
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# -F2_tp <= 0
	cons = np.zeros(len_h)
	cons[15] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# F2_tp - mu_f * F2 <= 0
	cons = np.zeros(len_h)
	cons[15] = 1
	cons[14] = -mu_finger
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# -F2_tm <= 0
	cons = np.zeros(len_h)
	cons[16] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# F2_tm - mu_f * F2 <= 0
	cons = np.zeros(len_h)
	cons[16] = 1
	cons[14] = -mu_finger
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# -F1 <= 0
	cons = np.zeros(len_h)
	cons[7] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# F1 <= MaxInputForce
	cons = np.zeros(len_h)
	cons[7] = 1
	h2 = MaxInputForce
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# F1_tp <= 0
	cons = np.zeros(len_h)
	cons[8] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	cons = np.zeros(len_h)
	cons[8] = 1
	h2 = MaxInputForce
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# F1_tm <= 0
	cons = np.zeros(len_h)
	cons[9] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	cons = np.zeros(len_h)
	cons[9] = 1
	h2 = MaxInputForce
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# F2 <= 0
	cons = np.zeros(len_h)
	cons[14] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	cons = np.zeros(len_h)
	cons[14] = 1
	h2 = MaxInputForce
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# F2_tp <= 0
	cons = np.zeros(len_h)
	cons[15] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	cons = np.zeros(len_h)
	cons[15] = 1
	h2 = MaxInputForce
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# F2_tm <= 0
	cons = np.zeros(len_h)
	cons[16] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	cons = np.zeros(len_h)
	cons[16] = 1
	h2 = MaxInputForce
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# Fn <= 0
	cons = np.zeros(len_h)
	cons[12] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	cons = np.zeros(len_h)
	cons[12] = 1
	h2 = MaxInputForce
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# -gamma1 <= MaxRelVel
	cons = np.zeros(len_h)
	cons[10] = -1
	h2 = MaxRelVel
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# gamma1 <= MaxRelVel
	cons = np.zeros(len_h)
	cons[10] = 1
	h2 = MaxRelVel
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# -v1 <= MaxRelVel
	cons = np.zeros(len_h)
	cons[11] = -1
	h2 = MaxRelVel
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# v1 <= MaxRelVel
	cons = np.zeros(len_h)
	cons[11] = 1
	h2 = MaxRelVel
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# theta-phi <= 0
	cons = np.zeros(len_h)
	cons[2] = 1
	cons[17] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# phi-theta <= pi/2
	cons = np.zeros(len_h)
	cons[2] = -1
	cons[17] = 1
	h2 = np.pi/2
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# -phi <= -pi/3
	cons = np.zeros(len_h)
	cons[17] = -1
	h2 = -np.pi/3
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	# phi <= 5/6*pi
	cons = np.zeros(len_h)
	cons[17] = 1
	h2 = np.pi*5/6
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

	## contacts finger 1 sliding up, finger 2 sticking
	# -v1<=0 (actually should be strict inequality)
	cons = np.zeros(len_h)
	cons[11] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))
	# -gamma1<=0 (actually should be strict inequality)
	cons = np.zeros(len_h)
	cons[10] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))
	# v1-gamma1<=0
	cons = np.zeros(len_h)
	cons[11] = 1
	cons[10] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))
	# -(v1-gamma1)<=0
	cons = np.zeros(len_h)
	cons[11] = -1
	cons[10] = 1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))
	# -F1tp<=0 
	cons = np.zeros(len_h)
	cons[8] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))
	# F1tp<=0 
	cons = np.zeros(len_h)
	cons[8] = 1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))
	# -mu_f*F1 + F1tm <= 0
	cons = np.zeros(len_h)
	cons[7] = -mu_finger
	cons[9] = 1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))
	# mu_f*F1 - F1tm <= 0
	cons = np.zeros(len_h)
	cons[7] = mu_finger
	cons[9] = -1
	h2 = 0
	H = np.vstack((H, cons))
	h = np.vstack((h, h2))

	return H,h