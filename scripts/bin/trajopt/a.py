import unittest
import subprocess
import psutil
import sys
import os
import numpy as np
import time
import socket
import pickle
import scipy
from scipy.linalg import block_diag
from scipy import optimize




from numpy.linalg import matrix_rank


polytube_controller = pickle.load(open("trajopt_example15_latest"+"_tube_output.p","rb"))
polytube_controller_x = polytube_controller['x']
polytube_controller_u = polytube_controller['u']
polytube_controller_G = polytube_controller['G']

for i in range(0,65):
	#print(polytube_controller_G[i])
	#print(matrix_rank(polytube_controller_G[i]))
	#print(np.linalg.det(polytube_controller_G[i]))
	print("time=%d"%i)
	print(polytube_controller_x[i])
	print(polytube_controller_u[i])