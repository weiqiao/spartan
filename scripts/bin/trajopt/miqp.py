import numpy as np
from gurobipy import *
from GurobiModel import GurobiModel
from operator import le, ge, eq

from pwa_system_class import AffineSystem, Domain, PiecewiseAffineSystem

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

def build_mip_ch(S, T, Q, R, P, X_N, norm):
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
		obj += model.add_stage_cost(Q, R, x, u, norm)

	# terminal constraint
	for i in range(nm):
		model.add_linear_constraints(X_N.A.dot(z[i]), le, X_N.b * d[i])

	# terminal cost
	obj += model.add_terminal_cost(P, x_next, norm)
	model.setObjective(obj)

	return model 

if __name__ == '__main__':
	# problem parameters
	T = 20
	nx = 2
	nu = 1
	nm = 2

	# piece 1 of PWA system
	A1 = np.array([[1,0.01],[0.1,1]])
	B1 = np.array([[0.],[0.01]])
	c1 = np.array([0.,0.])
	affsys1 = AffineSystem(A1,B1,c1)
	dom1_A = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
	dom1_b = np.array([0.1,0.12,1,1,4,4])
	dom1 = Domain(dom1_A,dom1_b)

	# piece 2 of PWA system
	A2 = np.array([[1,0.01],[-9.9,1]])
	B2 = np.array([[0.],[0.01]])
	c2 = np.array([0.,1.])
	affsys2 = AffineSystem(A2,B2,c2)
	dom2_A = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
	dom2_b = np.array([0.12,-0.1,1,1,4,4])
	dom2 = Domain(dom2_A,dom2_b)

	# PWA system
	S = PiecewiseAffineSystem([affsys1,affsys2],[dom1,dom2])
	Q = np.eye(nx)
	R = np.eye(nu)
	P = np.eye(nx)
	dom3_A = np.array([[1,0],[-1,0],[0,1],[0,-1]])
	dom3_b = np.array([0.05, 0.05, 0.2, 0.2])
	X_N = Domain(dom3_A,dom3_b)

	# build model
	model = build_mip_ch(S, T, Q, R, P, X_N, 'two')

	# set initial state
	x0 = np.array([0.11, -0.2])
	for k in range(S.nx):
		model.getVarByName('x0[%d]'%k).LB = x0[k]
		model.getVarByName('x0[%d]'%k).UB = x0[k]
	model.update()

	# optimize
	model.optimize()

	# print optimal variables
	for v in model.getVars():
		print('%s %g' % (v.varName, v.x))
	# print optimal objective
	print('Obj: %g' % model.objVal)

	# plot trajectory
	traj = np.zeros((nx,T))
	for t in range(T):
		for k in range(nx):
			v = model.getVarByName('x'+str(t)+'[%d]'%k)
			traj[k,t] = v.x
	plt.plot(traj[0,:],traj[1,:])
	plt.show()