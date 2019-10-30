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
    import cord_park, cord_car, cord_plan, vehicle_sim
except ImportError:
    raise

xy_dim = cord_park.input_xydim()
g_dim = cord_park.gdim_frm_xydim(xy_dim)


pk_g_idx = cord_park.input_pkgidx(g_dim)
print(pk_g_idx)

#v_target = cord_park.input_target_speed()
v_target = 10.8
#v_target = 10
park = cord_park.Park(g_dim, xy_dim, pk_g_idx,v_target/3.6*3)
#park = cord_park.Park(g_dim, xy_dim, pk_g_idx,v_target)

p = cord_plan.p_single_car(park)

g = cord_plan.g_single_car(park)
shortest_p_c = cord_plan.shortest_path_control(p,g,park)
car_gidx = cord_car.input_cars_init_gidx(park)
b_w = cord_plan.cord_bipartite_weights(car_gidx, shortest_p_c, park)
_, p_assign_num = cord_plan.bp_assign_dense(b_w)
#bip = cord_plan.cord_bp_rollout_cost(b_w, car_gidx, shortest_p_c, park)
print(p_assign_num)
#j_tilde, j, u_star = cord_plan.cord_multiagent_rollout(car_gidx, shortest_p_c, p,g, park)

#print(j_tilde,j)
paths, pk_assign_gidx = cord_plan.cord_pseudo_paths(car_gidx, shortest_p_c, park)
cars_xy_cords = cord_car.cars_xycords_from_path(paths, park)

print(cars_xy_cords[0][0],cars_xy_cords[0][1])

cars_states = vehicle_sim.cars_state_ini(cars_xy_cords)
cars_plannars = vehicle_sim.cars_planners_setup(cars_xy_cords)

vehicle_sim.vehicle_simulation(cars_xy_cords, park)
