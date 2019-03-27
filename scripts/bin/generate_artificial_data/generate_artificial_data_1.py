import unittest
import subprocess
import psutil
import sys
import os
import numpy as np
import time
import socket
import pickle


def generate_artificial_data_1(input_file, output_file):
    state_and_control = pickle.load(open(input_file,"rb"))
    pos_over_time = state_and_control["state"]
    params = state_and_control["params"]
    idx = int(params[0])
    T = int(params[idx+27])
    data = np.zeros((T+1,2))
    data[:,0] = pos_over_time[:,2] # theta
    data[:,1] = pos_over_time[:,0] # x
    data[T/4*3:,0] = pos_over_time[10,2] # for the last T/4 steps, the theta is the one at time step 10

    #output = {"data":data}
    pickle.dump( data, open(output_file,"wb"))



if __name__ == "__main__":
	file_name_1 = "trajopt_example11_latest.p"
	file_name_2 = "trajopt_example11_latest_artificial_disturbance.p"
	generate_artificial_data_1(file_name_1, file_name_2)