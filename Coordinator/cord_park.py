import numpy as np
import matplotlib.pyplot as plt

def input_xydim():
    """
    Specify the parking lot dimensions in x,y directions by the user

    return 1*2 np.array 'xy_dim'
    the first element of 'xy_dim' is the dimension in x_direction
    the second element of 'xy_dim' is the dimension in y_direction
    """
    print('Please specify the parking lot dimensions:')
    # Input needs to be integers
    x_dim = np.intp(input('In x-direction:'))
    y_dim = np.intp(input('In y-direction:'))
    xy_dim = np.array([x_dim,y_dim])
    return xy_dim


def gdim_frm_xydim(xy_dim):
    """
    Obtain the parking lot dimension for the grid index from xy_dim

    return 'g_dim'
    """
    g_dim = np.intp(xy_dim[0]*xy_dim[1])
    return g_dim

def input_pkgidx(g_dim):
    """
    Specify the parking spots index by the user

    return 1*pk_dim np.array 'pk_g_idx' where pk_dim is the number of spots
    """
    #print('Please specify the num of parking spots:')
    pk_dim = np.int(input('Please specify the num of parking spots:'))
    while pk_dim >= g_dim:
        print('Too many parking spots!')
        pk_dim = np.int(input('Please specify the num of parking spots:'))

    pk_g_idx = -np.ones(pk_dim, dtype = int)
    for idx in range(pk_dim):
        print('Input as grid index ranging from 0 to',g_dim-1)
        spot_idx = np.int(input())
        while (spot_idx < 0) or (spot_idx >= g_dim):
            print('Invalid input!')
            print('Input as grid index ranging from 0 to',g_dim-1)
            spot_idx = np.int(input())
        while spot_idx in pk_g_idx:
            print('Repeated input!')
            print('Input as grid index ranging from 0 to',g_dim-1)
            spot_idx = np.int(input())
            while (spot_idx < 0) or (spot_idx >= g_dim):
                print('Invalid input!')
                print('Input as grid index ranging from 0 to',g_dim-1)
                spot_idx = np.int(input())
        pk_g_idx[idx] = spot_idx

    pk_g_idx.sort()

    return pk_g_idx


def input_target_speed():
    """
    Specify the desired target speed in km/h

    return the desired target speed in km/h
    """
    target_speed = np.float32(input('Please specify the desired speed in km/h: '))/3.6

    return target_speed

def gidx_frm_xycrd(xy_crd, xy_dim):
    """
    Obtain the grid idex from grid xy_cord

    return 'g_idx', starting from 0. Example for a 3*2 grid
    1 | 3 | 5
    ---------
    0 | 2 | 4
    """
    g_idx = np.intp((xy_crd[0]-1)*xy_dim[1]+xy_crd[1]-1)
    return g_idx


def xycrd_frm_gidx(g_idx, xy_dim):
    """
    Obtain the xy_cord from grid_idx

    return 'xy_cord', starting from [1,1]. Example for a 3*2 grid
    [1,2] | [2,2] | [3,2]
    ---------------------
    [1,1] | [2,1] | [3,1]
    """
    x_crd = np.floor_divide(g_idx, xy_dim[1])+1
    y_crd = np.mod(g_idx, xy_dim[1])+1
    xy_crd = np.array([x_crd, y_crd])
    return xy_crd

def xyu_frm_u(u):
    """
    Compute the control signal in xy framework

    Input
    'u': control actions,
        including 0(still), 1(left), 2(right), 3(up), 4(down);
    Output
    'xyu': 2*1 array, control actions in xy frame
        [0,0](still), [-1,0](left), [1,0](right), [0,1](up), [0,-1](down)
    """
    return {0:np.array([0,0]),1:np.array([-1,0]),2:np.array([1,0]),\
            3:np.array([0,1]),4:np.array([0,-1])}[u]


class Park:
    """
    Define parking lot class

    store parking lot dimensions, parking spots
    """
    # Constructor
    def __init__(self, g_dim, xy_dim, pk_g_idx, unit_size):
        self.g_dim = g_dim              # Specify park grid index dimensions
        self.xy_dim = xy_dim            # Specify park xy cord dimensions
        self.pk_num = pk_g_idx.shape[0] # Specify parking spots numbers
        self.pk_g_idx = pk_g_idx        # Specify parking spots grid indeces
        self.unit_size = unit_size      # Specify per grid size 3*3 Parameters
        self.xy_size = self.xy_dim *\
         self.unit_size                 # Specify the overall size in xy directions
