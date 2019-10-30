import math
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import cubic_spline_planner, lqr_speed_steer_control
except ImportError:
    raise


lqr_Q = np.eye(5)
lqr_R = np.eye(2)
dt = 0.1  # time tick[s]
L = 0.5  # Wheel base of the vehicle [m]
max_steer = np.deg2rad(45.0)  # maximum steering angle[rad]
show_animation = True

#target_speed = 10.8/3.6

class Planner:

    def __init__(self, cx, cy, cyaw, ck, s, sp):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ck = ck
        self.s = s
        self.sp = sp


def sgl_car_state_ini(x_cords, y_cords):
    """

    """
    ax = x_cords
    ay = y_cords
    x = ax[0]
    y = ay[0]
    goal = [ax[-1], ay[-1]]

    vehicle_ini_state = lqr_speed_steer_control.State(x, y, yaw=0, v=0)

    return vehicle_ini_state

def cars_state_ini(cord_cars_xy_cords):
    """

    """
    car_num = len(cord_cars_xy_cords)
    cars_states = {}
    for i_c in range(car_num):
        cars_states[i_c] = sgl_car_state_ini(cord_cars_xy_cords[i_c][0],\
            cord_cars_xy_cords[i_c][1])

    return cars_states

def cars_planners_setup(cord_cars_xy_cords):
    """

    """
    car_num = len(cord_cars_xy_cords)
    cars_plannars = {}

    for i_c in range(car_num):
        cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
            cord_cars_xy_cords[i_c][0], cord_cars_xy_cords[i_c][1], ds=0.1)

        target_speed = 3  # simulation parameter km/h -> m/s

        sp = lqr_speed_steer_control.calc_speed_profile(cyaw, target_speed)
        cars_plannars[i_c] = Planner(cx, cy, cyaw, ck, s, sp)

    return cars_plannars


def vehicle_simulation(cord_cars_xy_cords, park):
    """

    """
    T = 500.0  # max simulation time
    goal_dis = 0.3
    stop_speed = 0.05
    time = 0.0
    t = [0.0]

    xy_dim = park.xy_dim
    unit_size = park.unit_size
    xy_size = park.xy_size

    # Major ticks every 20, minor ticks every 5
    x_grids = np.arange(0, xy_dim[0]+1) * unit_size
    y_grids = np.arange(0, xy_dim[1]+1) * unit_size
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    car_num = len(cord_cars_xy_cords)

    cars_states = cars_state_ini(cord_cars_xy_cords)
    cars_plannars = cars_planners_setup(cord_cars_xy_cords)

    cars_x, cars_y, cars_yaw, cars_v = {}, {}, {}, {}

    cars_e, cars_eth, cars_tind = np.zeros(car_num), np.zeros(car_num), np.zeros(car_num, dtype= int)

    for i_c in range(car_num):
        cars_x[i_c] = [cars_states[i_c].x]
        cars_y[i_c] = [cars_states[i_c].y]
        cars_yaw[i_c]= [cars_states[i_c].yaw]
        cars_v[i_c] = [cars_states[i_c].v]




    while T >= time:

        for i_c in range(car_num):
            dl, cars_tind[i_c], cars_e[i_c], cars_eth[i_c], ai =\
                lqr_speed_steer_control.lqr_speed_steering_control(
                cars_states[i_c], cars_plannars[i_c].cx, cars_plannars[i_c].cy,\
                cars_plannars[i_c].cyaw, cars_plannars[i_c].ck, cars_e[i_c], cars_eth[i_c],\
                cars_plannars[i_c].sp, lqr_Q, lqr_R)

            cars_states[i_c] = lqr_speed_steer_control.update(cars_states[i_c], \
                ai, dl)

            if abs(cars_states[i_c].v) <= stop_speed:
                cars_tind[i_c] += 1

            cars_x[i_c].append(cars_states[i_c].x)
            cars_y[i_c].append(cars_states[i_c].y)
            cars_yaw[i_c].append(cars_states[i_c].yaw)
            cars_v[i_c].append(cars_states[i_c].v)

        time = time + dt
        t.append(time)

        plt.cla()
        plt.xlim(0, xy_size[0])
        plt.ylim(0, xy_size[1])
        ax.set_xticks(x_grids)
        ax.set_yticks(y_grids)

        for i_c in range(car_num):
            plt.plot(cars_plannars[i_c].cx, cars_plannars[i_c].cy, "-r", label="course")
            plt.plot(cars_x[i_c], cars_y[i_c], "ob", label="trajectory")
            plt.plot(cars_plannars[i_c].cx[cars_tind[i_c]],\
            cars_plannars[i_c].cy[cars_tind[i_c]], "xg", label="target")

        plt.grid(which='both')
        plt.gca().set_aspect('equal', adjustable='box')
        #plt.grid(True)
        #plt.title("speed[km/h]:" + str(round(state.v * 3.6, 2))+ ",target index:" + str(target_ind))
        plt.pause(0.0001)
