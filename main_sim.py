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
shortest_p_c = cord_plan.shortest_path_control(p,g,park)
car_gidx = cord_car.input_cars_init_gidx(park)
b_w = cord_plan.cord_bipartite_weights(car_gidx, shortest_p_c, park)
print(b_w)
j_tilde = cord_plan.cord_bp_rollout_cost(b_w, car_gidx, shortest_p_c, park)
print(j_tilde)
print(p[0,:,3])
print('results')
bip, j, u_star = cord_plan.cord_multiagent_rollout_1(car_gidx, shortest_p_c, p,g, park)
print(bip)
print(j)
print(u_star)
