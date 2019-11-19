import math
import sys

import numpy as np


try:
    import cubic_spline_planner
except ImportError:
    raise

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
            elif i == 1:
                for j in range(5):
                    if j == 0:
                        self.cx[i][j] = [radius-idx*ds for idx in range(nd)]
                        self.cy[i][j] = [0.0 for idx in range(nd)]
                        self.cyaw[i][j] = [np.pi for idx in range(nd)]
                        self.ck[i][j] = [0.0 for idx in range(nd)]
                    elif j == 1:
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
                for j in range(5):
                    if j == 0:
                        self.cx[i][j] = [-radius+idx*ds for idx in range(nd)]
                        self.cy[i][j] = [0.0 for idx in range(nd)]
                        self.cyaw[i][j] = [0.0 for idx in range(nd)]
                        self.ck[i][j] = [0.0 for idx in range(nd)]
                    elif j == 2:
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
                for j in range(5):
                    if j == 0:
                        self.cx[i][j] = [0.0 for idx in range(nd)]
                        self.cy[i][j] = [-radius+idx*ds for idx in range(nd)]
                        self.cyaw[i][j] = [np.pi/2 for idx in range(nd)]
                        self.ck[i][j] = [0.0 for idx in range(nd)]
                    elif j == 1:
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
                for j in range(5):
                    if j == 0:
                        self.cx[i][j] = [0.0 for idx in range(nd)]
                        self.cy[i][j] = [radius-idx*ds for idx in range(nd)]
                        self.cyaw[i][j] = [-np.pi/2 for idx in range(nd)]
                        self.ck[i][j] = [0.0 for idx in range(nd)]
                    elif j == 1:
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
        rx.extend([ix+x_crds[i] for ix in primi.cx[u_path[i]][u_path[i+1]]])
        ry.extend([iy+y_crds[i] for iy in primi.cy[u_path[i]][u_path[i+1]]])
        ryaw.extend([iyaw for iyaw in primi.cyaw[u_path[i]][u_path[i+1]]])
        rk.extend([ik for ik in primi.ck[u_path[i]][u_path[i+1]]])

    return rx, ry, ryaw, rk



def main():
    import matplotlib.pyplot as plt

    primi = Primitives(4.5, 0.1)
    x_crds = np.array([-8, 1, 10, 19, 28.0, 28, 28, 28, 28])
    y_crds = np.array([1.0, 1.0, 1.0, 1.0, 1, 10, 19, 28, 37])
    u_path = np.array([-1, 2, 2, 2, 2, 3, 3, 3, 3, 0])

    rx, ry, ryaw, rk = calc_lattice_course(primi, x_crds, y_crds, u_path)

    plt.subplots(1)

    plt.plot(rx, ry, "-r", label="spline")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
