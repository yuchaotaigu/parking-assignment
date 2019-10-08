import math
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np
import scipy.linalg as la



sys.path.append("Vehicle/")
sys.path.append("Coordinator/")
sys.path.append("Miscellaneous/")
try:
    import cord_park, cord_car, cord_plan, park_one
except ImportError:
    raise

xy_dim = cord_park.input_xydim()
g_dim = cord_park.gdim_frm_xydim(xy_dim)


pk_g_idx = cord_park.input_pkgidx(g_dim)
print(pk_g_idx)

#v_target = cord_park.input_target_speed()
v_target = 10

park = cord_park.Park(g_dim, xy_dim, pk_g_idx,v_target)

p = cord_plan.p_single_car(park)

g = cord_plan.g_single_car(park)
"""
j_bar = np.zeros(p.shape[0])
j_bar_1 = np.zeros(p.shape[0])
j_bar_1[pk_g_idx[0]] = np.int(0.8*sys.maxsize)

u_vi, j_vi, id= cord_plan.value_iteration(p,g,j_bar_1)
"""
shortest_p_c = cord_plan.shortest_path_control(p,g,park)
car_gidx = cord_car.input_cars_init_gidx(park)

#b_w = cord_plan.cord_bipartite_weights(car_gidx, shortest_p_c, park)
"""
print(b_w)
start = time.time()
b_w = cord_plan.cord_bipartite_weights(car_gidx, shortest_p_c, park)
j_tilde = cord_plan.cord_bp_rollout_cost(b_w, car_gidx, shortest_p_c, park)
print(j_tilde)

end = time.time()
print(end - start)
"""
start = time.time()
j_delta, j, u_star = cord_plan.cord_multiagent_rollout(car_gidx, shortest_p_c, park)
print(j_delta)
print(j)
print(u_star)
end = time.time()
print(end - start)
#check_result = np.prod(shortest_p_c[pk_g_idx[0]][1]==j_vi)
#print(shortest_p_c[pk_g_idx[0]][1]==j_vi)
"""
start = time.time()
u_vi, j_vi, id= cord_plan.value_iteration(p,g,j_bar)
end = time.time()
print(end - start)
"""
#u_vi_1, j_vi_1, id_1= cord_plan.value_iteration(p,g,j_bar_1)
"""
cord_plan.plot_sgl_car_cost(park, j_vi)

cord_plan.plot_sgl_car_cost(park, j_vi_1)

path = cord_car.car_path_gidx(2, u_vi, park)
print(path)

path_mod = cord_car.car_path_gidx_len(path, path.size + 1)
print(path_mod)

paths = np.zeros((3, path_mod.size))
for i in range(3):
    paths[i,:] = path_mod.T

print(paths)
print(cord_car.car_rollout_sim(paths))

"""

#print(False in (j_vi == j_vi_al))


#u_vi_al, j_vi_al, id_al= park_one.value_iteration(p,g)
#print(xy_dim)
#print(j_vi)
#print(u_vi)
#print(id)



#car_ini_gidx = cord_car.input_cars_init_gidx(park)

#print(car_ini_gidx)



"""
try:
    import cubic_spline_planner, lqr_speed_steer_control, park_one
except ImportError:
    raise


lqr_Q = np.eye(5)
lqr_R = np.eye(2)
dt = 0.1  # time tick[s]
L = 0.5  # Wheel base of the vehicle [m]
max_steer = np.deg2rad(45.0)  # maximum steering angle[rad]
show_animation = True

def do_simulation_ini(park, cx, cy, cyaw, ck, speed_profile, goal, x, y, yaw, v=0):
    T = 500.0  # max simulation time
    goal_dis = 0.3
    stop_speed = 0.05

    #state = State(x=-0.0, y=-0.0, yaw=0.0, v=10.0/3.6)
    state = lqr_speed_steer_control.State(x, y, yaw, v)

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]

    xy_dim = park.xy_dim
    unit_size = park.unit_size
    xy_size = park.xy_size


    # Major ticks every 20, minor ticks every 5
    x_grids = np.arange(0, xy_dim[0]+1) * unit_size
    y_grids = np.arange(0, xy_dim[1]+1) * unit_size
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    e, e_th = 0.0, 0.0

    while T >= time:
        dl, target_ind, e, e_th, ai = lqr_speed_steer_control.lqr_speed_steering_control(
            state, cx, cy, cyaw, ck, e, e_th, speed_profile, lqr_Q, lqr_R)

        state = lqr_speed_steer_control.update(state, ai, dl)

        if abs(state.v) <= stop_speed:
            target_ind += 1

        time = time + dt

        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        if math.sqrt(dx ** 2 + dy ** 2) <= goal_dis:
            print("Goal")
            break

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        if target_ind % 1 == 0 and show_animation:
            plt.cla()
            #plt.xlim(0, round(xy_size[0]))
            #plt.ylim(0, round(xy_size[1]))
            plt.xlim(0, xy_size[0])
            plt.ylim(0, xy_size[1])
            ax.set_xticks(x_grids)
            ax.set_yticks(y_grids)
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            #plt.axis("equal")
            plt.grid(which='both')
            plt.gca().set_aspect('equal', adjustable='box')
            #plt.grid(True)
            plt.title("speed[km/h]:" + str(round(state.v * 3.6, 2))
                      + ",target index:" + str(target_ind))
            plt.pause(0.0001)

    return t, x, y, yaw, v


def sgl_vehicle_sim(park, x_cords, y_cords):
    print("LQR steering control tracking start!!")
    #ax = [0.0, 6.0, 12.5, 10.0, 17.5, 20.0, 25.0]
    #ay = [0.0, -3.0, -5.0, 6.5, 3.0, 0.0, 0.0]
    ax = x_cords
    ay = y_cords
    ini = [ax[0], ay[0]]
    goal = [ax[-1], ay[-1]]

    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=0.1)
    target_speed = 10.0 / 3.6  # simulation parameter km/h -> m/s

    sp = lqr_speed_steer_control.calc_speed_profile(cyaw, target_speed)

    t, x, y, yaw, v = do_simulation_ini(
        park, cx, cy, cyaw, ck, sp, goal, ini[0], ini[1], np.pi/4, 0.0)

    if show_animation:  # pragma: no cover
        plt.close()
        plt.subplots(1)
        plt.plot(ax, ay, "xb", label="waypoints")
        plt.plot(cx, cy, "-r", label="target course")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots(1)
        plt.plot(s, [np.rad2deg(iyaw) for iyaw in cyaw], "-r", label="yaw")
        plt.grid(True)
        plt.legend()
        plt.xlabel("line length[m]")
        plt.ylabel("yaw angle[deg]")

        plt.subplots(1)
        plt.plot(s, ck, "-r", label="curvature")
        plt.grid(True)
        plt.legend()
        plt.xlabel("line length[m]")
        plt.ylabel("curvature [1/m]")

        plt.show()



xy_dim = park_one.input_xydim()
g_dim = park_one.gdim_frm_xydim(xy_dim)


pk_g_idx = park_one.input_pkgidx(g_dim)


park = park_one.Park(g_dim, xy_dim, pk_g_idx)

p = park_one.p_single_car(park)

g = park_one.g_single_car(park)


#print(g[:,1])
u_vi, j_vi, id= park_one.value_iteration(p,g)
print(xy_dim)
print(j_vi)
print(u_vi)
print(id)
park_one.plot_sgl_car_cost(park, j_vi)

g_idx = np.int(input('initial position'))
path = park_one.grid_access(g_idx,u_vi,park)
print(path)

park_one.plot_sgl_car_path(park,path)

x_cords, y_cords = park_one.xycrd_for_plan(path, park, max_len = 7)
print(x_cords)
print(y_cords)


sgl_vehicle_sim(park, x_cords, y_cords)

#park_one.plot_park_grid_world(park)
"""
