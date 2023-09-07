# code structure follows the style of Symplectic ODE-Net
# https://github.com/d-biswa/Symplectic-ODENet/blob/master/experiment-single-embed/data.py

import time
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from se3hamneuralode import to_pickle, from_pickle

# Pybullet drone environment

def get_dataset(test_split=0.5, save_dir=None, **kwargs):
    data = {}

    assert save_dir is not None
    path = '{}/jackal-SE3-pointclouds-dataset.pkl'.format(save_dir)

    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except Exception as e:
        print(e)
        raise NotImplementedError
    return data

def arrange_data(x, t, num_points=2):
    '''Arrange data to feed into neural ODE in small chunks'''
    assert 2 <= num_points <= len(t)
    x_stack = []
    for i in range(num_points):
        if i < num_points-1:
            x_stack.append(x[:, i:-num_points+i+1,:,:])
        else:
            x_stack.append(x[:, i:,:,:])
    x_stack = np.stack(x_stack, axis=1)
    x_stack = np.reshape(x_stack,
                (x.shape[0], num_points, -1, x.shape[3]))
    t_eval = t[0:num_points]
    return x_stack, t_eval
