import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import least_squares
from optimization_model import model as model
from experimental_data import * # Calcium traces Weber 2023 (Figure 3) data are imported from experimental_data.py

# after the PPL DAN parameters were fit using the salt response data, their weight is fit to the sugar response data (other parameters stay fixed)
# after the PAM DAN parameters were fit using the sugar response data, their weight is fit to the salt response data (other parameters stay fixed)



def comp_residual(params, fixed_params, data, x, inp):
    '''

    :param params:
    :param data:
    :param x:
    :param inp:
    :param data_var:
    :return:
    '''

    # params:  w, y0, a0
    # fixed_params:  tau, tau_a, b
    params = [fixed_params[0], params[0], fixed_params[1], fixed_params[2], params[1], params[2]]

    solution = odeint(model, y0=params[-2:], t=x, args=(params, inp))[:,0]
    residual = data - solution

    return residual

def optimize(data, fixed_params, x, inp, low, high, i):
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
    w = np.random.uniform(low[0], high[0], 1)[0]
    y0 = data[0]
    a0 = 0

    params = [w, y0, a0]

    results = least_squares(comp_residual, x0=params, kwargs={'data': data, 'x': x, 'fixed_params' : fixed_params, 'inp': inp})

    # save perfoprmance, optimal parameter values and initial parameter values
    model_performances[i, 0] = results.cost
    model_performances[i, 1] = fixed_params[0]  # fixed tau
    model_performances[i, 2] = results.x[0] # optimal w
    model_performances[i, 3] = fixed_params[1] # fixed tau_a
    model_performances[i, 4] = fixed_params[2] # fixed b
    model_performances[i, 5] = results.x[1]  # optimal y0
    model_performances[i, 6] = results.x[2]  # optimal a0
    model_performances[i, 7] = w  # initial w
    model_performances[i, 8] = y0
    model_performances[i, 9] = a0



# optimize model using least squares n times with randomly drawn initial values
np.seed = 999
n = 10000 # number optimization attempts
model_performances = np.zeros((n, 10)) # performance, optimal tau, optimal w, optimal tau_a, optimal b, optimal y0, optimal a0, initial tau, initial w, initial tau_a, initial b, initial y0, initial a0
data = salt_mean_PAM
x = x_salt_PAM
inp = inp_PAM
fixed_params = [0.637312, 3.243857, 0.193716] # tau, tau_a, b from the first optimization with salt (DL1) or fructose (pPAM)

# select initial values for params
low = [-20] # w
high = [20] # w

for i, elem in enumerate(range(n)):
    optimize(data=data, fixed_params=fixed_params, x=x, inp=inp, low=low, high=high, i=i)


# find the solution that minimizes the cost function
best_model = np.argmin(model_performances[:,0]) # find the model with the smallest sum of residuals


plt.figure()
plt.plot(x_salt_PAM,salt_mean_PAM,label='data')
plt.axvspan(np.where(inp==1)[0][0], np.where(inp==1)[0][-1], color='grey', alpha=0.3)
plt.plot(x_salt_PAM,odeint(model, y0=model_performances[best_model,5:7], t=x_salt_PAM, args=(model_performances[best_model, 1:7], inp_PAM))[:,0],label='solution')
plt.legend()
plt.show()


# save data
model_performances = pd.DataFrame(model_performances, columns=['performance', 'fixed tau', 'optimal w', 'fixed tau_a', 'fixed b', 'optimal y0', 'optimal a0', 'initial w', 'initial y0', 'initial a0', ])
model_performances.to_csv('/Users/anna/Documents/Python/Data/Thum_Collab/lsq_PAM_pun.csv')

pd.set_option("display.max_columns", None)

print(model_performances.iloc[best_model])
















