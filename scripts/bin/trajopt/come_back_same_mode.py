import pickle
import numpy as np
import scipy as sp
from gurobipy import Model,GRB,LinExpr
import time as time
from pypolycontain.lib.containment_encodings import subset_LP,subset_zonotopes
from pypolycontain.lib.polytope import polytope
from pypolycontain.lib.zonotope import zonotope

import pydrake.solvers.mathematicalprogram as mp
from pydrake.solvers.mosek import MosekSolver
import pydrake.symbolic as sym

import carrot_linearization_1 as calin1 
import carrot_linearization_2 as calin2
import carrot_linearization_3 as calin3
import poly_trajectory as polytraj 
from pwa_system import system, linear_cell


class GetbackController(mp.MathematicalProgram):
	def formulate_optimization_problem(self,cur_linear_cell,x,x_ref,G_inv):
		# cur_linear_cell: current linear cell
		# x: current state
		# f(x,u) + delta = Ax+Bu+c+delta is in x_ref + G*P
		n = len(x)
		m = int(B.shape[1])
		A = cur_linear_cell.A 
		B = cur_linear_cell.B
		c = cur_linear_cell.c 
		p = cur_linear_cell.p
		H = p.H
		h = p.h 
		num_constraints = H.shape[0]
		u = self.NewContinuousVariables(m,'u') 
		delta = self.NewContinuousVariables(n,'delta')
		#tmp = self.NewContinuousVariables(n,'tmp')
		tmp = G_inv.dot(A.dot(x) + B.dot(u) + c + delta - x_ref)
		for i in range(n):
			self.AddLinearConstraint(tmp[i] <= 1)
			self.AddLinearConstraint(tmp[i] >= -1)
		self.AddQuadraticCost((delta).dot(delta))
		# state-input space polytopic constraints
		for i in range(num_constraints):
			self.AddLinearConstraint(H[i,:n].dot(x) + H[i,n:].dot(u) <= h[i])

		return u, delta

	def get_solution(self,x):
		try:
			return self.GetSolution(x)
		except TypeError:
			return x

def compute_trajectory_given_modes(initial_x,T)
	file_name = "trajopt_example11_latest"+"_tube_output.p"
	file_input = pickle.load(open(file_name,"rb"))
	x = file_input['x']
	u = file_input['u']
	G = file_input['G']
	G_inv = file_input['G_inv']
	theta = file_input['theta']
	list_of_cells = file_input['list_of_cells']

	cur_x = initial_x

	l = len(u)
	n = len(x[0])
	x_stack = x[0]
	u_stack = u[0]
	G_stack = G[0]
	G_inv_stack = G_inv[0]
	theta_stack = theta[0]
	for i in range(1,l):
		np.vstack((x_stack,x[i]))
		np.vstack((u_stack,u[i]))
		np.vstack((G_stack,G[i]))
		np.vstack((G_inv_stack,G_inv[i]))
		np.vstack((theta_stack,theta[i]))
	# np.vstack((x_stack,x[l]))
	# np.vstack((G_stack,G[l]))
	# np.vstack((G_inv_stack,G_inv[l]))

	solver = MosekSolver()
	traj_x = []
	traj_x.append(initial_x)
	cur_x = initial_x
	for t in range(T):
		# compute distance from cur_x to the nearest polytope
		cur_x_stack = np.tile(cur_x,(l,1)) # equivalent to repmat
		px = G_inv.dot(cur_x_stack-x_stack)
		px_star = np.maximum(np.minimum(px,1),-1)
		d_signed = G.dot(px-px_star) 
		d_signed = d_signed.reshape((l,n))
		d = np.linalg.norm(d_signed,axis = 1)
		idx_min = np.argmin(d)
		d_min = d[idx_min]
		
		if d_min > 0:
			# outside nearest polytope, need solve QP
			idx_min -= 1 # the child of the node
			cur_linear_cell = list_of_cells[idx_min] # linear_cell(A,B,c,polytope(H,h))
			prog1 = GetbackController()
			cur_u_var, cur_delta_var = prog1.formulate_optimization_problem(cur_linear_cell,cur_x,x[idx_min],G_inv[idx_min])
			result = solver.Solve(prog1)
			assert result == mp.SolutionResult.kSolutionFound
			cur_u = prog1.get_solution(cur_u_var)
			print cur_u 


