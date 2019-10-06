import sys
import math

import numpy as np


try:
    import cord_park
except ImportError:
    raise


def input_cars_init_gidx(park):
    """
    Specify the initial indices of cars by the user

    return 1*car_dim np.array 'car_ini_gidx' where car_dim is the number of cars
    """
    g_dim = park.g_dim
    pk_dim = park.pk_num
    pk_g_idx = park.pk_g_idx

    car_dim = np.int(input('Please specify the num of cars: '))
    while car_dim > pk_dim:
        print('Too many cars!')
        car_dim = np.int(input('Please specify the num of cars: '))

    car_ini_gidx = -np.ones(car_dim, dtype = int)
    for idx in range(car_dim):
        print('Input as grid index ranging from 0 to',g_dim-1)
        spot_idx = np.int(input())
        while (spot_idx < 0) or (spot_idx >= g_dim) or (spot_idx in pk_g_idx)\
            or (spot_idx in car_ini_gidx):
            print('Invalid input!')
            print('Input as grid index ranging from 0 to',g_dim-1)
            spot_idx = np.int(input())
        car_ini_gidx[idx] = spot_idx

    car_ini_gidx.sort()

    return car_ini_gidx

def car_path_gidx(g_idx,u,park):
    """
    Find out the optimal grid path given the current calc_position

    Inputs:
    'g_idx': current vehicle position in terms of grid idx
    'u': optimal control obtained via value iterations
    'park': instance of class Park

    Output:
    'path': array of optimal path in terms of grid index
    """

    xy_dim = park.xy_dim

    g = g_idx
    path = np.array([g])
    while g not in park.pk_g_idx:
        xy_crd = cord_park.xycrd_frm_gidx(g,xy_dim)
        xy_crd = xy_crd + cord_park.xyu_frm_u(u[g])
        g = cord_park.gidx_frm_xycrd(xy_crd,xy_dim)
        path = np.append(path,g)

    return path

def car_path_gidx_len(path, length):
    """
    Contruct path with specific length

    Inputs:
    'path': array of optimal path in terms of grid index
    'length': desired length of the path
    Output:
    'path_modified': array of optimal path in terms of grid index
        with desired length
    """
    cur_length = path.shape[0]
    if cur_length >= length:
        path_modified = path
    else:
        path_modified = np.append(path, path[-1]*np.ones(length-path.shape[0]),0)

    return path_modified

def car_rollout_sim(paths):
    """
    Perform rollout simulation of the cars to check if collision occurs

    Input:
    'paths': length*car_dim array of car planned paths where length is the length and
        car_dim is the number of cars
    Output:
    'True' if collision is predicted to occur
    'False' if no collision is predicted
    """
    length = paths.shape[1]

    for i in range(length):
        positions = paths[:,i]
        pos, count = np.unique(positions, return_counts=True)
        duplication = pos[count > 1]
        break if duplication.size > 0

    if duplication.size == 0:
        return False
    else:
        return True