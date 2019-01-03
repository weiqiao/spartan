from __future__ import absolute_import, division, print_function
from itertools import islice, chain
from collections import namedtuple
import time
import numpy as np
import pydrake.solvers.mathematicalprogram as mp
from pydrake.solvers.ipopt import IpoptSolver

class TrajectoryOptimization(mp.MathematicalProgram):
	def add_dynamics_constraints(self, mass, T, x_init, x_final, dt):
		F = self.NewContinuousVariables(T,'F')
		for t in range(T):
			self.AddLinearConstraint(F[t] <= 100)
			self.AddLinearConstraint(F[t] >= -100)
		x = self.NewContinuousVariables(2, "x_%d" % 0)
		x_over_time = x 
		for t in range(1,T+1):
			x = self.NewContinuousVariables(2, "x_%d" % t)
			x_over_time = np.vstack((x_over_time, x))
		n = 2
		for i in range(n):
			self.AddLinearConstraint(x_over_time[0,i] == x_init[i])
			self.AddLinearConstraint(x_over_time[-1,i] - x_final[i] <= 0.00001)
			self.AddLinearConstraint(x_over_time[-1,i] - x_final[i] >= -0.00001)
		for t in range(T):
			x_curr = x_over_time[t,:]
			x_next = x_over_time[t+1,:]
			self.AddLinearConstraint(x_next[0] - (x_curr[0] + x_curr[1]*dt) <= 0.00001)
			self.AddLinearConstraint(x_next[0] - (x_curr[0] + x_curr[1]*dt) >= -0.00001)
			self.AddLinearConstraint(x_next[1] - (x_curr[1] + F[t]/mass*dt) <= 0.00001)
			self.AddLinearConstraint(x_next[1] - (x_curr[1] + F[t]/mass*dt) >= -0.00001)
		return x_over_time, F

	def get_solution(self,x):
		try:
			return self.GetSolution(x)
		except TypeError:
			return x

if __name__=="__main__":
	prog = TrajectoryOptimization()
	mass = 1
	T = 20
	dt = 0.01
	x_init = np.array([0,0])
	x_final = np.array([1,0])
	x_over_time, F = prog.add_dynamics_constraints(mass, T, x_init, x_final, dt)
	solver = IpoptSolver()
	prog.SetSolverOption(mp.SolverType.kGurobi, "LogToConsole", 1)
	prog.SetSolverOption(mp.SolverType.kGurobi, "OutputFlag", 1)
	start_time = time.time()
	result = solver.Solve(prog)
	solve_time = time.time() - start_time
	assert result == mp.SolutionResult.kSolutionFound
	print(solve_time)
	print(prog.get_solution(x_over_time))