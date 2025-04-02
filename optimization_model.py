import numpy as np


# ODE definition
def model(Y, x, params, inp):
    '''

    :param Y:
    :param x:
    :param params:
    :param inp:
    :return:
    '''

    y, a = Y
    tau, w, tau_a, b, y0, a0 = params
    max_idx = len(inp)-1
    idx = min(int(np.floor(x)), max_idx) # max_idx because otherwise odeint will solve outside of the bounds of x

    dy_dt = ((-y + inp[idx] * w) -a * b) / tau
    da_dt = (- a + y) / tau_a # spike frequency adaptation

    return [dy_dt, da_dt]
