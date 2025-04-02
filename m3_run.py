from ANNarchy import *
from m2_model import build_model
from m2_parameters import params
import matplotlib.pyplot as plt
import stimulus as stimulus


# this is needed for multiprocessing on apple silicone chips
import platform
if platform.system() == "Darwin":
    import multiprocessing as mp
    mp.set_start_method('fork')

# ANNarchy setup
clear()
setup(dt=params['dt'])
setup(method='rk4')


# creates and compiles parent model
net, pops, syns, mons, KC_input = build_model(params)

def simulation(idx, net, params, save_data=True):
    '''

        simulates a single model instance

        :param idx: network instance (assigned by ANNarchy.parallel_run())
        :param net: ANNarchy network object
        :param params: simulation params
        :param save_data: bool

        :return: None if save_data=True, ANNarchy monitor objects if save_data=False

        '''

    # this ensures that the random number generators work properly to create different KC activiation between the N instances when calling parallel_run()
    KC_input = stimulus.odor_stim(params)
    net.get(pops['KC']).rates = KC_input

    # randomly draw DAN baseline rate for each instance within a call of parallel_run()
    bl = Uniform(params['baseline'][0],params['baseline'][1]).get_value()
    net.get(pops['DAN_PAM']).bl = bl
    net.get(pops['DAN_c1']).bl = bl
    net.get(pops['DAN_d1']).bl = bl
    net.get(pops['DAN_g1']).bl = bl
    net.get(pops['DAN_f1']).bl = bl

    # randomly draw learning rate for each instance within a call of parallel_run()
    lr = Uniform(params['lr'][0],params['lr'][0]).get_value()
    net.get(syns['KC_MBONn']).lr = lr
    net.get(syns['KC_MBONp']).lr = lr

    # simulate
    net.simulate(params['sim_duration'])

    # save data
    if save_data == True:

        data = dict()
        monitors = dict()

        save_path = params['save_path']

        for mon in mons.keys():

            if '_r' in mon:
                monitors[mon] = (net.get(mons[mon]).get('r'))
            if '_w' in mon:
                monitors[mon] = net.get(mons[mon]).get('w')

        data.update({'sim_params': params, 'monitors': monitors})
        filename = 'sim' + str(idx)
        np.savez(os.path.join(save_path, filename), data=data)

        return

    if save_data == False:

        rates = dict()
        weights = dict()

        rates['KC'] = net.get(mons['KC_r']).get('r')
        rates['MBONp'] = net.get(mons['MBONp_r']).get('r')
        rates['MBONn'] = net.get(mons['MBONn_r']).get('r')
        rates['readout'] = net.get(mons['readout_r']).get('r')
        rates['c1'] = net.get(mons['c1_r']).get('r')
        rates['d1'] = net.get(mons['d1_r']).get('r')
        rates['f1'] = net.get(mons['f1_r']).get('r')
        rates['g1'] = net.get(mons['g1_r']).get('r')
        rates['PAM'] = net.get(mons['PAM_r']).get('r')

        weights['KC_MBONp'] = net.get(mons['KC_MBONp_w']).get('w')
        weights['KC_MBONn'] = net.get(mons['KC_MBONn_w']).get('w')

        return rates, weights



# run simulation of N = params['N'] network instances
results = parallel_run(method=simulation,number=params['N'], params=[params for i in range(params['N'])], save_data=[True for i in range(params['N'])],sequential=False, same_seed=False)




# readout = results[0][0]['readout']
# KC_rate = results[0][0]['KC']
# MBONp_rate = results[0][0]['MBONp']
# MBONn_rate = results[0][0]['MBONn']
# DAN_c1_rate = results[0][0]['c1']
# DAN_d1_rate = results[0][0]['d1']
# DAN_f1_rate = results[0][0]['f1']
# DAN_g1_rate = results[0][0]['g1']
# DAN_PAM_rate = results[0][0]['PAM']
# KC_MBONp_w = results[0][1]['KC_MBONp']
# KC_MBONn_w = results[0][1]['KC_MBONn']



