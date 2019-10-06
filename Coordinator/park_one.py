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
    def __init__(self, g_dim, xy_dim, pk_g_idx):
        self.g_dim = g_dim              # Specify park grid index dimensions
        self.xy_dim = xy_dim            # Specify park xy cord dimensions
        self.pk_num = pk_g_idx.shape[0] # Specify parking spots numbers
        self.pk_g_idx = pk_g_idx        # Specify parking spots grid indices
        self.unit_size = 3.0            # Specify per grid size 3*3 Parameters
        self.xy_size = self.xy_dim *\
         self.unit_size                 # Specify the overall size in xy directions

def plot_park_grid_world(park):
    """

    """
    xy_dim = park.xy_dim
    unit_size = park.unit_size
    xy_size = park.unit_size

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Major ticks every 20, minor ticks every 5
    x_grids = np.arange(0, xy_dim[0]+1) * unit_size
    y_grids = np.arange(0, xy_dim[1]+1) * unit_size

    ax.set_xticks(x_grids)
    ax.set_yticks(y_grids)

    #And a corresponding grid
    ax.grid(which='both')

    #plt.show()





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

def value_iteration(p_car, g_car):
    """
    value iteration algorithm of undiscounted shortest path problem
        s_dim is dimension of states, u_dim is dimension of control,
        vi_it is the maximum number of iterations

    Input:  p_car: transition matrix, dimension is s_dim*s_dim*u_dim
            g_car: cost matrix, dimension is s_dim*u_dim
    Output: u_vi[:,idx0_vi]: optimal controls
            j_vi[:,idx0_vi]: optimal costs
            vi_real_it: number of iterations performed

    """
    u_dim = p_car.shape[2]
    g_dim = p_car.shape[0]
    s_dim = g_dim

    vi_it = 1000
    vi_real_it = 0

    u_vi = np.zeros((s_dim,vi_it))
    j_vi = np.zeros((s_dim,vi_it+1))
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
                    r_vi[idx2_vi,:] = g_car[idx1_vi,idx2_vi] + \
                    p_car[idx1_vi,:,idx2_vi].dot(j_vi[:,idx0_vi+1])
                    u_vi[idx1_vi,idx0_vi] = r_vi.argmin()
                    j_vi[idx1_vi,idx0_vi] = r_vi.min()

    if idx0_vi == 0:
        vi_real_it = vi_it
    else:
        vi_real_it = 1000-idx0_vi+2
    return u_vi[:,idx0_vi], j_vi[:,idx0_vi], vi_real_it

def grid_access(g_idx,u,park):
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
        xy_crd = xycrd_frm_gidx(g,xy_dim)
        xy_crd = xy_crd + xyu_frm_u(u[g])
        g = gidx_frm_xycrd(xy_crd,xy_dim)
        path = np.append(path,g)

    return path



def xycrd_for_plan(path, park, max_len = 7):
    """
    Return the x cords and y cords of the access-granted grid within the length of max_len

    Inputs:
    'path': array of optimal path in terms of grid index
    'park': instance of class Park
    'max_len': maximum length of the point cords

    Output:
    'x_cords': a list of x cords
    'y_cords': a list of y cords
    """
    xy_dim = park.xy_dim
    g_dim = park.g_dim

    unit_size = park.unit_size

    pth_dim = path.shape[0]
    x_cords = []
    y_cords = []

    g_pth_idx = []

    if pth_dim <= max_len:
        for idx in range(pth_dim):
            xy_cords = xycrd_frm_gidx(path[idx],xy_dim)
            x_cords.append(unit_size*xy_cords[0]-unit_size/2)
            y_cords.append(unit_size*xy_cords[1]-unit_size/2)

    else:
        xy_cords = xycrd_frm_gidx(path[0],xy_dim)
        x_cords.append(unit_size*xy_cords[0]-unit_size/2)
        y_cords.append(unit_size*xy_cords[1]-unit_size/2)
        width = (pth_dim-2) // (max_len-2)
        for idx in [i * width for i in range(1,max_len-1)]:
            xy_cords = xycrd_frm_gidx(path[idx],xy_dim)
            x_cords.append(unit_size*xy_cords[0]-unit_size/2)
            y_cords.append(unit_size*xy_cords[1]-unit_size/2)

        xy_cords = xycrd_frm_gidx(path[-1],xy_dim)
        x_cords.append(unit_size*xy_cords[0]-unit_size/2)
        y_cords.append(unit_size*xy_cords[1]-unit_size/2)

    return x_cords, y_cords


def plot_sgl_car_cost(park, j_g):
    """
    Plotting the optimal cost functions on the xy cordinate system
    """
    xy_dim = park.xy_dim
    g_dim = park.g_dim

    j_xy = np.zeros((xy_dim[1],xy_dim[0]))
    if g_dim == j_g.shape[0]:
        for g_idx in np.arange(g_dim):
            [x_crd, y_crd] = xycrd_frm_gidx(g_idx,xy_dim)
            j_xy[y_crd-1,x_crd-1] = j_g[g_idx]
    else:
        print('Invalid input!')

    plt.pcolor(j_xy)
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
            [x_crd, y_crd] = xycrd_frm_gidx(idx,xy_dim)
            p_xy[y_crd-1,x_crd-1] = 1

    plt.pcolor(p_xy)
    plt.colorbar()
    plt.show()






def input_car_init_gidx(park):
    """
    Specify the initial position as its grid idx of a car

    'park': instance park of class Park;
    """
    g_dim = park.g_dim

    ini_g_idx = -1
    print('Please specify the initial spot of a car ranging from 0 to',g_dim-1,':')
    ini_g_idx = np.int(input())
    while (ini_g_idx < 0) or (ini_g_idx > g_dim-1):
        print('Invalid input!')
        print('Please specify the initial spot of a car ranging from 0 to',g_dim-1,':')
        ini_g_idx = np.int(input())

    c_ini_g_idx = ini_g_idx

    return c_ini_g_idx



class Car:
    """
    Define car class

    store parking lot dimensions
    """
    # Constructor
    def __init__(self):
        self.ini_g_idx = -np.int(1)                 # Set car initial grid index to -1
        self.xy_cord = -np.ones(2, dtype = int)     # Set car initial xy cord to [-1 -1]
        self.g_idx = -np.int(1)                     # Sst car current grid index to -1

    def input_init_gidx(self, park):
        """
        Specify the initial position as its grid idx of a car

        'park': instance park of class Park;
        """
        g_dim = park.g_dim
        xy_dim = park.xy_dim

        ini_g_idx = -1
        print('Please specify the initial spot of a car ranging from 0 to',g_dim-1,':')
        ini_g_idx = np.int(input())
        while (ini_g_idx < 0) or (ini_g_idx > g_dim-1):
            print('Invalid input!')
            print('Please specify the initial spot of a car ranging from 0 to',g_dim-1,':')
            ini_g_idx = np.int(input())

        self.ini_g_idx = ini_g_idx
        self.g_idx = self.ini_g_idx
        self.xy_cord = xycrd_frm_gidx(self.g_idx, xy_dim)

    def update_current_gidx(self, park, u):
        """
        Update the current position as its grid idx and xy cord given control u

        'park': instance park of class Park;
        'u': control 0(still), 1(left), 2(right), 3(up), 4(down)
        """
        xy_dim = park.xy_dim
        p_sgl_car_u = p_single_car_u(park,u)
        self.g_idx = np.where(p_sgl_car_u[self.g_idx,:])[0][0]
        self.xy_cord = xycrd_frm_gidx(self.g_idx, xy_dim)



"""

xy_dim = input_xydim()
g_dim = gdim_frm_xydim(xy_dim)

#print(g_dim)

# test input_pkgidx
pk_g_idx = input_pkgidx(g_dim)
#print(pk_g_idx[0],pk_g_idx[1])

# test class Park
#print(xycrd_frm_gidx(g_dim-1,xy_dim))
#print(gidx_frm_xycrd(xycrd_frm_gidx(g_dim-1,xy_dim),xy_dim))

park = Park(g_dim, xy_dim, pk_g_idx)
#print(park.xy_dim, park.g_dim, park.pk_g_idx)



#u = np.intp(input('control:'))



# test input_car_initial_gidx
#c_g_idx = input_car_init_gidx(park)
#print(c_g_idx)

car = Car()
car.input_init_gidx(park)
print(car.ini_g_idx == c_g_idx)
print(car.ini_g_idx,car.xy_cord,car.g_idx)
car.update_current_gidx(park,np.int(input('Input control ')))

print(car.ini_g_idx,car.xy_cord,car.g_idx)


# test p_single_car
#p_u = p_single_car_u(park, u)
#print(p_u)
p = p_single_car(park)
#print(p[:,:,u]== p_u)

# test g_single_car
#g_u = g_single_car_u(park, u)
#print(g_u)
g = g_single_car(park)
#print(g[:,u]== g_u)
#print(p[:,:,1])

#print(g[:,1])
u_vi, j_vi, id= value_iteration(p,g)
print(xy_dim)
print(j_vi)
print(u_vi)
print(id)
plot_sgl_car_cost(park, j_vi)

g_idx = np.int(input('initial position'))
path = grid_access(g_idx,u_vi,park)
print(path)

plot_sgl_car_path(park,path)

x_cords, y_cords = xycrd_for_plan(path, park, max_len = 7)
print(x_cords)
print(y_cords)
"""
