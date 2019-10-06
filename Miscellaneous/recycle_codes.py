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
