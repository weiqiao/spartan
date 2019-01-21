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


def tvlqr(A,B,Q,R,P0,T):
	# A, B, P, Q are 3d tensors, a tuple (At, Bt, Qt, Rt) at time t for t from 0 to T-1
	n = A.shape[0]
	m = B.shape[1]
	K = np.zeros((m,n,T))
	Pt = P0
	for t in range(T-1,-1,-1):
		At = A[:,:,t]
		Bt = B[:,:,t]
		Qt = Q[:,:,t]
		Rt = R[:,:,t]
		Kt = -np.linalg.inv(Rt+Bt.T.dot(Pt).dot(Bt)).dot(Bt.T).dot(Pt).dot(At)
		Ct = At + Bt.dot(Kt)
		Pt = Qt + (Kt.T).dot(Rt).dot(Kt) + (Ct.T).dot(Pt).dot(Ct)
		K[:,:,t] = Kt

	return K

def forward(A,B,K,x):
	# simulate linear system x_next = A*x+B*(K*x)
	y = A.dot(x) + B.dot(K).dot(x)
	return y

if __name__ == '__main__':
	n = 2
	m = 1
	dt = 0.01
	T = 1000
	A = np.array([[1,dt],[0,1]])
	B = np.array([[0],[dt]])
	Q = np.eye(n)
	R = np.eye(m)

	AT = np.zeros((n,n,T))
	BT = np.zeros((n,m,T))
	QT = np.zeros((n,n,T))
	RT = np.zeros((m,m,T))
	P0 = np.eye(n)
	for t in range(T):
		AT[:,:,t] = A
	for t in range(T):
		BT[:,:,t] = B
	for t in range(T):
		QT[:,:,t] = Q
	for t in range(T):
		RT[:,:,t] = R

	K = tvlqr(AT,BT,QT,RT,P0,T)

	x0 = np.array([[10],[20]])
	x = np.zeros((n,T+1))
	x[:,0] = x0[:,0]
	for t in range(T):
		x[:,t+1] = forward(AT[:,:,t],BT[:,:,t],K[:,:,t],x[:,t])
	plt.plot(x[0,:],x[1,:])
	plt.show()