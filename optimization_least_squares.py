import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import least_squares
from optimization_model import model as model
from experimental_data import * # Calcium traces Weber 2023 (Figure 3) data are imported from experimental_data.py



def comp_residual(params, data, x, inp):
    '''

    :param params:
    :param data:
    :param x:
    :param inp:
    :param data_var:
    :return:
    '''
    solution = odeint(model, y0=params[-2:], t=x, args=(params, inp))[:,0]
    residual = data - solution

    return residual

def optimize(data, x, inp, low, high, i, bounds):
    '''

    :param data:
    :param data_var:
    :param x:
    :param inp:
    :param low:
    :param high:
    :param i:
    :return:
    '''

    # randomly draw initial parameter values
    tau = np.random.uniform(low[0], high[0], 1)[0]
    w = np.random.uniform(low[1], high[1], 1)[0]
    tau_a = np.random.uniform(low[2], high[2], 1)[0]
    b = np.random.uniform(low[3], high[3], 1)[0]
    y0 = data[0]
    a0 = 0
    params = [tau, w, tau_a, b, y0, a0]

    results = least_squares(comp_residual, x0=params, bounds=bounds, kwargs={'data': data, 'x': x, 'inp': inp})

    # save perfoprmance, optimal parameter values and initial parameter values
    model_performances[i, 0] = results.cost
    model_performances[i, 1] = results.x[0] # optimal tau
    model_performances[i, 2] = results.x[1] # optimal w
    model_performances[i, 3] = results.x[2] # optimal tau_a
    model_performances[i, 4] = results.x[3] # optimal b
    model_performances[i, 5] = results.x[4] # optimal y0
    model_performances[i, 6] = results.x[5] # optimal a0
    model_performances[i, 7] = tau # initial
    model_performances[i, 8] = w
    model_performances[i, 9] = tau_a
    model_performances[i, 10] = b
    model_performances[i, 11] = y0
    model_performances[i, 12] = a0

    # plt.figure()
    # plt.plot(x_salt_g1, salt_mean_g1, label='data')
    # plt.axvspan(np.where(inp==1)[0][0], np.where(inp==1)[0][-1], color='grey', alpha=0.3)
    # plt.plot(x_salt_g1, odeint(model, y0=results.x[-2:], t=x_salt_g1, args=(results.x, inp_g1))[:, 0], label='solution')
    # plt.legend()
    #
    # plt.show()


# tau and tau_a can not be negative
tau_min = 0.5
tau_max = np.inf
b_min = 0.0
b_max = np.inf

bounds = ([tau_min, -np.inf, tau_min, b_min, -np.inf, -np.inf],  # lower bounds
              [tau_max, np.inf, tau_max, b_max, np.inf, np.inf])  # upper bounds

# select initial values for params
low = [tau_min, 0, tau_min, 0] # tau, w, tau_a, b
high = [5, 10, 10, 10] # tau, w, tau_a, b


# optimize model using least squares n times with randomly drawn initial values
np.seed = 999
n = 10000 # number optimization attempts
model_performances = np.zeros((n, 13)) # performance, optimal tau, optimal w, optimal tau_a, optimal b, optimal y0, optimal a0, initial tau, initial w, initial tau_a, initial b, initial y0, initial a0
data = salt_mean_c1
x = x_salt_c1
inp = inp_c1


for i, elem in enumerate(range(n)):
    optimize(data=data,x=x, inp=inp, low=low, high=high, i=i, bounds=bounds)

# find the solution that minimizes the cost function
best_model = np.argmin(model_performances[:,0]) # find the model with the smallest sum of residuals

plt.figure()
plt.plot(x_salt_c1,salt_mean_c1,label='data')
plt.axvspan(np.where(inp==1)[0][0], np.where(inp==1)[0][-1], color='grey', alpha=0.3)
plt.plot(x_salt_c1,odeint(model, y0=model_performances[best_model, -2:], t=x_salt_c1, args=(model_performances[best_model, 1:7], inp_c1))[:,0],label='solution')
plt.legend()
plt.show()


# save data
model_performances = pd.DataFrame(model_performances, columns=['performance', 'optimal tau', 'optimal w', 'optimal tau_a', 'optimal b', 'optimal y0', 'optimal a0', 'initial tau', 'initial w', 'initial tau_a', 'initial b', 'initial y0', 'initial a0'])
model_performances.to_csv('/Users/anna/Documents/Python/Data/Thum_Collab/lsq_PPLc1_pun.csv')

pd.set_option("display.max_columns", None)

print(model_performances.iloc[best_model])
















