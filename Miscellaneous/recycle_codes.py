def u_multiple_dim(n_car):
    """
    Compute the dimension of control space from the number of cars n_car
        refer to https://en.wikipedia.org/wiki/Pentatope_number

    Input: 'n_car' the number of cars

    Output: effective dimensions of the control space

    """
    f = math.factorial
    n = n_car+1
    r = 4
    return np.int(n*(n+1)*(n+2)*(n+3) / f(r))

def p_car(p_sgl_car, n_car):
    """
    Construct transition matrix of multiple vehicles for all combination of controls,
        including the redundant control pairs, e.g., (1,2) and (2,1) are regarded different

    Input:
        'p_sgl_car': g_dim*g_dim*5 array, 5 is the number of control:
            0(still), 1(left), 2(right), 3(up), 4(down)
        'n_car': number of cars
    Output: g_dim**n_car, g_dim**n_car, u_dim**n_car array
    """

    u_dim = 5
    g_dim = p_sgl_car.shape[0]


    p_car = np.zeros((g_dim**n_car, g_dim**n_car, u_dim**n_car), dtype = int)
    p_cal = p_sgl_car
    if n_car == 1:
        p_car = p_cal
    else:
        for car_idx in range(2, n_car+1):
            p_old_cal = p_cal
            u_tem_dim = p_old_cal.shape[2] * 5
            p_cal = np.zeros((g_dim**car_idx, g_dim**car_idx, u_tem_dim), dtype = int)
            for u_sgl_idx in range(u_dim):
                for old_idx in range(p_old_cal.shape[2]):
                    p_cal[:,:,u_sgl_idx*old_idx+old_idx] = np.kron(p_sgl_car[:,:,u_sgl_idx], p_old_cal[:,:,old_idx])

    p_car = p_cal

    return p_car



class Primitives:
    """
    Define motion premitives for path assembling
    """

    def __init__(self, radius, ds):
        self.cx = {0:{1:[], 2:[], 3:[], 4:[]}}
        self.cx[1] = {1:[], 2:[], 3:[], 4:[]}
        self.cx[2] = {1:[], 2:[], 3:[], 4:[]}
        self.cx[3] = {1:[], 2:[], 3:[], 4:[]}
        self.cx[4] = {1:[], 2:[], 3:[], 4:[]}
        self.cx[-1] = {1:[], 2:[], 3:[], 4:[]}

        self.cy = {0:{1:[], 2:[], 3:[], 4:[]}}
        self.cy[1] = {1:[], 2:[], 3:[], 4:[]}
        self.cy[2] = {1:[], 2:[], 3:[], 4:[]}
        self.cy[3] = {1:[], 2:[], 3:[], 4:[]}
        self.cy[4] = {1:[], 2:[], 3:[], 4:[]}
        self.cy[-1] = {1:[], 2:[], 3:[], 4:[]}


        self.cyaw = {0:{1:[], 2:[], 3:[], 4:[]}}
        self.cyaw[1] = {1:[], 2:[], 3:[], 4:[]}
        self.cyaw[2] = {1:[], 2:[], 3:[], 4:[]}
        self.cyaw[3] = {1:[], 2:[], 3:[], 4:[]}
        self.cyaw[4] = {1:[], 2:[], 3:[], 4:[]}
        self.cyaw[-1] = {1:[], 2:[], 3:[], 4:[]}

        self.ck = {0:{1:[], 2:[], 3:[], 4:[]}}
        self.ck[1] = {1:[], 2:[], 3:[], 4:[]}
        self.ck[2] = {1:[], 2:[], 3:[], 4:[]}
        self.ck[3] = {1:[], 2:[], 3:[], 4:[]}
        self.ck[4] = {1:[], 2:[], 3:[], 4:[]}
        self.ck[-1] = {1:[], 2:[], 3:[], 4:[]}

        nd = int(radius/ds)
        d_theta = np.pi/4/nd
        semi_pi = np.pi/2
        k_r = 1/radius


        for i in range(-1,5):
            if i == -1:
                for j in range(1,5):
                    if j == 1:
                        self.cx[i][j] = [-idx*ds for idx in range(nd)]
                        self.cy[i][j] = [0.0 for idx in range(nd)]
                        self.cyaw[i][j] = [np.pi for idx in range(nd)]
                        self.ck[i][j] = [0.0 for idx in range(nd)]
                    elif j == 2:
                        self.cx[i][j] = [idx*ds for idx in range(nd)]
                        self.cy[i][j] = [0.0 for idx in range(nd)]
                        self.cyaw[i][j] = [0.0 for idx in range(nd)]
                        self.ck[i][j] = [0.0 for idx in range(nd)]
                    elif j == 3:
                        self.cx[i][j] = [0.0 for idx in range(nd)]
                        self.cy[i][j] = [idx*ds for idx in range(nd)]
                        self.cyaw[i][j] = [np.pi/2 for idx in range(nd)]
                        self.ck[i][j] = [0.0 for idx in range(nd)]
                    elif j == 4:
                        self.cx[i][j] = [0.0 for idx in range(nd)]
                        self.cy[i][j] = [radius-idx*ds for idx in range(nd)]
                        self.cyaw[i][j] = [-np.pi/2 for idx in range(nd)]
                        self.ck[i][j] = [0.0 for idx in range(nd)]
            elif i == 0:
                for j in range(1,5):
                    if j == 1:
                        self.cx[i][j] = [radius-idx*ds for idx in range(nd)]
                        self.cy[i][j] = [0.0 for idx in range(nd)]
                        self.cyaw[i][j] = [np.pi for idx in range(nd)]
                        self.ck[i][j] = [0.0 for idx in range(nd)]
                    elif j == 2:
                        self.cx[i][j] = [-radius+idx*ds for idx in range(nd)]
                        self.cy[i][j] = [0.0 for idx in range(nd)]
                        self.cyaw[i][j] = [0.0 for idx in range(nd)]
                        self.ck[i][j] = [0.0 for idx in range(nd)]
                    elif j == 3:
                        self.cx[i][j] = [0.0 for idx in range(nd)]
                        self.cy[i][j] = [-radius+idx*ds for idx in range(nd)]
                        self.cyaw[i][j] = [np.pi/2 for idx in range(nd)]
                        self.ck[i][j] = [0.0 for idx in range(nd)]
                    elif j == 4:
                        self.cx[i][j] = [0.0 for idx in range(nd)]
                        self.cy[i][j] = [radius-idx*ds for idx in range(nd)]
                        self.cyaw[i][j] = [-np.pi/2 for idx in range(nd)]
                        self.ck[i][j] = [0.0 for idx in range(nd)]
            elif i == 1:
                for j in range(5):
                    if j == 1:
                        self.cx[i][j] = [radius-idx*ds for idx in range(2*nd)]
                        self.cy[i][j] = [0.0 for idx in range(2*nd)]
                        self.cyaw[i][j] = [np.pi for idx in range(2*nd)]
                        self.ck[i][j] = [0.0 for idx in range(2*nd)]
                    elif j == 3:
                        self.cx[i][j] = [radius-radius*math.sin(idx*d_theta) for idx in range(1, 2*nd)]
                        self.cy[i][j] = [radius-radius*math.cos(idx*d_theta) for idx in range(1, 2*nd)]
                        self.cyaw[i][j] = [np.pi-idx*d_theta for idx in range(1,2*nd)]
                        self.ck[i][j] = [-k_r for idx in range(1,2*nd)]
                    elif j == 4:
                        self.cx[i][j] = [radius+radius*math.cos(np.pi/2+idx*d_theta) for idx in range(1, 2*nd)]
                        self.cy[i][j] = [radius*math.sin(np.pi/2+idx*d_theta)-radius for idx in range(1, 2*nd)]
                        self.cyaw[i][j] = [-np.pi+idx*d_theta for idx in range(1,2*nd)]
                        self.ck[i][j] = [k_r for idx in range(1,2*nd)]
            elif i == 2:
                for j in range(1,5):
                    if j == 2:
                        self.cx[i][j] = [-radius+idx*ds for idx in range(2*nd)]
                        self.cy[i][j] = [0.0 for idx in range(2*nd)]
                        self.cyaw[i][j] = [0.0 for idx in range(2*nd)]
                        self.ck[i][j] = [0.0 for idx in range(2*nd)]
                    elif j == 3:
                        self.cx[i][j] = [-radius+radius*math.sin(idx*d_theta) for idx in range(1,2*nd)]
                        self.cy[i][j] = [radius-radius*math.cos(idx*d_theta) for idx in range(1,2*nd)]
                        self.cyaw[i][j] = [idx*d_theta for idx in range(1,2*nd)]
                        self.ck[i][j] = [k_r for idx in range(1,2*nd)]
                    elif j == 4:
                        self.cx[i][j] = [-radius+radius*math.sin(idx*d_theta) for idx in range(1, 2*nd)]
                        self.cy[i][j] = [-radius+radius*math.cos(idx*d_theta) for idx in range(1, 2*nd)]
                        self.cyaw[i][j] = [-idx*d_theta for idx in range(1,2*nd)]
                        self.ck[i][j] = [-k_r for idx in range(1,2*nd)]
            elif i == 3:
                for j in range(1,5):
                    if j == 1:
                        self.cx[i][j] = [-radius+radius*math.cos(idx*d_theta) for idx in range(1,2*nd)]
                        self.cy[i][j] = [-radius+radius*math.sin(idx*d_theta) for idx in range(1,2*nd)]
                        self.cyaw[i][j] = [np.pi/2+idx*d_theta for idx in range(1,2*nd)]
                        self.ck[i][j] = [k_r for idx in range(1,2*nd)]
                    elif j == 2:
                        self.cx[i][j] = [radius+radius*math.cos(np.pi-idx*d_theta) for idx in range(1,2*nd)]
                        self.cy[i][j] = [-radius+radius*math.sin(idx*d_theta) for idx in range(1,2*nd)]
                        self.cyaw[i][j] = [np.pi/2-idx*d_theta for idx in range(1,2*nd)]
                        self.ck[i][j] = [-k_r for idx in range(1,2*nd)]
                    elif j == 3:
                        self.cx[i][j] = [0.0 for idx in range(2*nd)]
                        self.cy[i][j] = [-radius+idx*ds for idx in range(2*nd)]
                        self.cyaw[i][j] = [np.pi/2 for idx in range(2*nd)]
                        self.ck[i][j] = [0.0 for idx in range(2*nd)]
            elif i == 4:
                for j in range(1,5):
                    if j == 1:
                        self.cx[i][j] = [-radius+radius*math.cos(-idx*d_theta) for idx in range(1,2*nd)]
                        self.cy[i][j] = [radius+radius*math.sin(-idx*d_theta) for idx in range(1,2*nd)]
                        self.cyaw[i][j] = [-np.pi/2-idx*d_theta for idx in range(2*nd)]
                        self.ck[i][j] = [-k_r for idx in range(1,2*nd)]
                    elif j == 2:
                        self.cx[i][j] = [radius-radius*math.cos(idx*d_theta) for idx in range(1,2*nd)]
                        self.cy[i][j] = [radius-radius*math.sin(idx*d_theta) for idx in range(1,2*nd)]
                        self.cyaw[i][j] = [-np.pi/2+idx*d_theta for idx in range(2*nd)]
                        self.ck[i][j] = [k_r for idx in range(1,2*nd)]
                    elif j == 4:
                        self.cx[i][j] = [0.0 for idx in range(1,2*nd)]
                        self.cy[i][j] = [radius-idx*ds for idx in range(2*nd)]
                        self.cyaw[i][j] = [-np.pi/2 for idx in range(nd)]
                        self.ck[i][j] = [0.0 for idx in range(nd)]




def calc_lattice_course(primi, x_crds, y_crds, u_path):
    """

    """
    path_dim = len(x_crds)


    rx = []
    ry = []
    ryaw = []
    rk = []

    print(path_dim)
    print(len(u_path))
    u_path.astype(int)
    for i in range(path_dim):
        print(i)

        rx.extend([ix+x_crds[i] for ix in primi.cx[u_path[i]][u_path[i+1]]])
        ry.extend([iy+y_crds[i] for iy in primi.cy[u_path[i]][u_path[i+1]]])
        ryaw.extend([iyaw for iyaw in primi.cyaw[u_path[i]][u_path[i+1]]])
        rk.extend([ik for ik in primi.ck[u_path[i]][u_path[i+1]]])

    return rx, ry, ryaw, rk



def main():
    import matplotlib.pyplot as plt

    primi = Primitives(4.5, 0.1)
    x_crds = np.array([1.0, 5.5, 10, 14.5, 19.0, 19.0, 19.0, 19.0, 19.0])
    y_crds = np.array([1.0, 1.0, 1.0, 1.0, 5.5, 10, 14.5, 19, 23.5])
    u_path = np.array([-1, 2, 2, 2, 2, 3, 3, 3, 3, 0])

    rx, ry, ryaw, rk = calc_lattice_course(primi, x_crds, y_crds, u_path)
    """
    plt.subplots(1)

    plt.plot(rx, ry, "-r", label="spline")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()

    plt.show()
    """

if __name__ == '__main__':
    main()
