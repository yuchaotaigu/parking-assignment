import sys
import math

import numpy as np
import matplotlib.pyplot as plt

try:
    import cord_park
except ImportError:
    raise


fake_infinity = np.int(0.8*sys.maxsize)


def p_single_car_u(park, u):
    """
    Define transition matrix for single car with specified control 'u'

    'park': instance park of class Park;
    'u': control actions,
        including 0(still), 1(left), 2(right), 3(up), 4(down);
        control not permitted by the park layout would result in 0.
    """
    g_dim = park.g_dim
    xy_dim = park.xy_dim
    pk_g_idx = park.pk_g_idx


    p_sgl_car_u = np.zeros((g_dim, g_dim))
    if u == 0:
        p_sgl_car_u = np.identity(g_dim)
    elif u == 1:
        for idx in range(g_dim):
            if idx in range(xy_dim[1]):
                p_sgl_car_u[idx, idx] = 1
            else:
                p_sgl_car_u[idx, idx-xy_dim[1]] = 1
    elif u == 2:
        for idx in range(g_dim):
            if idx in range(g_dim,g_dim-xy_dim[1]-1,-1):
                p_sgl_car_u[idx, idx] = 1
            else:
                p_sgl_car_u[idx, idx + xy_dim[1]] = 1
    elif u == 3:
        for idx in range(g_dim):
            if np.mod(idx+1, xy_dim[1]) == 0:
                p_sgl_car_u[idx, idx] = 1
            else:
                p_sgl_car_u[idx,idx+1] = 1
    elif u == 4:
        for idx in range(g_dim):
            if np.mod(idx, xy_dim[1]) == 0:
                p_sgl_car_u[idx, idx] = 1
            else:
                p_sgl_car_u[idx,idx-1] = 1

    else:
        print('Invalid input for control')

    # Rewrite the transition for parking spots
    for idx in pk_g_idx:
        p_sgl_car_u[idx, :] = np.zeros(g_dim, dtype = int)
        p_sgl_car_u[idx, idx] = 1

    return p_sgl_car_u


def p_single_car(park):
    """
    Define transition matrix for single car for all control 'u'

    Input: 'park': instance park of class Park;

    Return g_dim*g_dim*5 array, 5 is the number of control:
        0(still), 1(left), 2(right), 3(up), 4(down)
    """
    g_dim = park.g_dim

    u_dim = 5
    p_sgl_car = np.zeros((g_dim, g_dim, u_dim), dtype = int)
    for u_idx in range(u_dim):
        p_sgl_car[:,:,u_idx] = p_single_car_u(park, u_idx)

    return p_sgl_car


def g_single_car_u(park, u):
    """
    Define cost per stage at certain control 'u';
            every move costs 1;
            hiting wall costs inf;
            hiting each other costs inf;

    Input:  'park': instance park of class Park;
            'u': control 0(still), 1(left), 2(right), 3(up), 4(down)

    Return g_dim array, 5 is the number of control:
    """
    g_dim = park.g_dim
    pk_g_idx = park.pk_g_idx
    xy_dim = park.xy_dim

    g_sgl_c_u = np.zeros(g_dim)

    if u == 0:
        for idx in range(g_dim):
            if idx not in pk_g_idx:
                g_sgl_c_u[idx] = 1
    elif u == 1:
         for idx in range(g_dim):
             if idx not in pk_g_idx:
                if idx in range(xy_dim[1]):
                    g_sgl_c_u[idx] = np.Inf
                else:
                    g_sgl_c_u[idx] = 1
    elif u == 2:
        for idx in range(g_dim):
            if idx not in pk_g_idx:
                if idx in range(g_dim-xy_dim[1]):
                   g_sgl_c_u[idx] = 1
                else:
                   g_sgl_c_u[idx] = np.Inf
    elif u == 3:
        for idx in range(g_dim):
            if idx not in pk_g_idx:
                if np.mod(idx+1, xy_dim[1]) == 0:
                   g_sgl_c_u[idx] = np.Inf
                else:
                   g_sgl_c_u[idx] = 1
    elif u == 4:
        for idx in range(g_dim):
            if idx not in pk_g_idx:
                if np.mod(idx, xy_dim[1]) == 0:
                   g_sgl_c_u[idx] = np.Inf
                else:
                   g_sgl_c_u[idx] = 1
    else:
        print('Invalid input for control!')

    return g_sgl_c_u


def g_single_car(park):
    """
    Define cost per stage for all control;
            every move costs 1;
            hiting wall costs inf;
            hiting each other costs inf;

    Input:  'park': instance park of class Park;
            'u': control 0(still), 1(left), 2(right), 3(up), 4(down)

    Return g_dim*5 array, 5 is the number of control:
    """
    g_dim = park.g_dim
    #pk_g_idx = park.pk_g_idx
    #xy_dim = park.xy_dim
    u_dim = 5

    g_sgl_car = np.zeros((g_dim,u_dim))
    for u_idx in range(u_dim):
        g_sgl_car[:,u_idx] = g_single_car_u(park, u_idx)

    return g_sgl_car

def g_coupled(rollout_result):
    """
    Compute the coupled cost of vehicle paths based on the rollout result

    Input:
    'rollout_result': True if collision is prediced, False otherwise
    Output:
    np.int(0) for no collision and np.Inf for collision
    """
    if not rollout_result:
        return np.int(0)
    else:
        return np.Inf


def value_iteration(p, g, j_bar):
    """
    value iteration algorithm of undiscounted shortest path problem
        s_dim is dimension of states, u_dim is dimension of control,
        vi_it is the maximum number of iterations

    Input:  p: transition matrix, dimension is s_dim*s_dim*u_dim
            g: cost matrix, dimension is s_dim*u_dim
            j_bar: final stage cost
    Output: u_vi[:,idx0_vi]: optimal controls
            j_vi[:,idx0_vi]: optimal costs
            vi_real_it: number of iterations performed

    """
    u_dim = p.shape[2]
    g_dim = p.shape[0]
    s_dim = g_dim

    vi_it = 1000
    vi_real_it = 0

    u_vi = np.zeros((s_dim,vi_it))
    j_vi = np.zeros((s_dim,vi_it+1))
    j_vi[:,vi_it] = j_bar
    #j_vi_min = np.zeros((vi_it,1))
    #if u_dim == g_car.shape[1]:
    for idx0_vi in np.arange(vi_it-1,-1,-1):
        if (idx0_vi <= vi_it-2) and (np.prod(j_vi[:,idx0_vi+1] == j_vi[:,idx0_vi+2])):
            j_vi[:,idx0_vi] = j_vi[:,idx0_vi+1]
            u_vi[:,idx0_vi] = u_vi[:,idx0_vi+1]
            break
        else:
            for idx1_vi in np.arange(s_dim):
                r_vi = np.zeros((u_dim,1))
                for idx2_vi in np.arange(u_dim):
                    r_vi[idx2_vi,:] = g[idx1_vi,idx2_vi] + \
                    p[idx1_vi,:,idx2_vi].dot(j_vi[:,idx0_vi+1])
                    u_vi[idx1_vi,idx0_vi] = r_vi.argmin()
                    j_vi[idx1_vi,idx0_vi] = r_vi.min()

    if idx0_vi == 0:
        vi_real_it = vi_it
    else:
        vi_real_it = 1000-idx0_vi+2
    return u_vi[:,idx0_vi], j_vi[:,idx0_vi], vi_real_it








def plot_sgl_car_cost(park, j_g):
    """
    Plotting the optimal cost functions on the xy cordinate system
    """
    xy_dim = park.xy_dim
    g_dim = park.g_dim

    for i in range(j_g.shape[0]):
        if j_g[i] >= fake_infinity:
            j_g[i] = np.nan

    j_xy = np.zeros((xy_dim[1],xy_dim[0]))
    if g_dim == j_g.shape[0]:
        for g_idx in np.arange(g_dim):
            [x_crd, y_crd] = cord_park.xycrd_frm_gidx(g_idx,xy_dim)
            j_xy[y_crd-1,x_crd-1] = j_g[g_idx]
    else:
        print('Invalid input!')

    plt.pcolor(j_xy)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.show()

def plot_sgl_car_path(park,path):
    """
    Ploting the path assigned to one car
    """
    xy_dim = park.xy_dim
    g_dim = park.g_dim

    p_xy = np.zeros((xy_dim[1],xy_dim[0]))

    for idx in range(g_dim):
        if idx in path:
            [x_crd, y_crd] = cord_park.xycrd_frm_gidx(idx,xy_dim)
            p_xy[y_crd-1,x_crd-1] = 1

    plt.pcolor(p_xy)
    plt.colorbar()
    plt.show()
