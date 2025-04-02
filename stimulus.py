from ANNarchy import *


def odor_stim(params):
    '''
    Creates an odor input stimulus for the network simulation. It can directly be used as input into the first layer.

    '''

    # baseline
    base_rate = np.zeros(params['n_KC'])

    # select KCs activated by amylacetate
    pc_active_KC_AM = DiscreteUniform(params['pc_active_KC'][0], params['pc_active_KC'][1]).get_value()
    n_active_KC_AM = int((params['n_KC'] / 100) * pc_active_KC_AM)
    active_KC_AM = DiscreteUniform(0, params['n_KC']-1).get_values(n_active_KC_AM)
    activation_AM = Uniform(params['activation'][0], params['activation'][1]).get_values(n_active_KC_AM)

    # select KCs activated by benzaldehyde
    pc_active_KC_BA = DiscreteUniform(params['pc_active_KC'][0], params['pc_active_KC'][1]).get_value()
    n_active_KC_BA = int((params['n_KC'] / 100) * pc_active_KC_BA)
    active_KC_BA = DiscreteUniform(0, params['n_KC']-1).get_values(n_active_KC_BA-1)
    activation_BA = Uniform(params['activation'][0], params['activation'][1]).get_values(n_active_KC_BA-1)

    start_offset_AM = DiscreteUniform(params['start_offset'][0], params['start_offset'][1]).get_values(n_active_KC_AM)
    stop_offset_AM = DiscreteUniform(params['start_offset'][0], params['start_offset'][1]).get_values(n_active_KC_AM)

    start_offset_BA = DiscreteUniform(params['start_offset'][0], params['start_offset'][1]).get_values(n_active_KC_BA)
    stop_offset_BA = DiscreteUniform(params['start_offset'][0], params['start_offset'][1]).get_values(n_active_KC_BA)

    # create TimedArray
    KC_input = np.ones((int(params['sim_duration'] / params['dt']), params['n_KC'])) * base_rate

    for i, elem in enumerate(params['odors']):
        start = params['odor_start'][i]
        stop = params['odor_stop'][i]

        if params['SFA'] == True:
            if elem == 'odor_AM':
                for idx, KC in enumerate(active_KC_AM):
                    if params['start_offset'] != (0,0):
                        start_offset = start_offset_AM[idx]
                        stop_offset = stop_offset_AM[idx]
                        activation_length = int((stop + stop_offset) / params['dt']) - int((start + start_offset) / params['dt'])
                        activation = np.ones(activation_length) * activation_AM[idx]
                        sfa = 0.06 * (1 - np.exp(-9 * np.arange(0, activation_length) / activation_length / 2))
                        KC_input[int((start + start_offset) / params['dt']):int((stop + stop_offset) / params['dt']),
                        KC] = activation - sfa
                    elif params['start_offset'] == (0,0):
                        activation_length = int(stop / params['dt']) - int(start / params['dt'])
                        activation = np.ones(activation_length) * activation_AM[idx]
                        sfa = 0.06 * (1 - np.exp(-9 * np.arange(0, activation_length) / activation_length / 2))
                        KC_input[int(start / params['dt']):int(stop / params['dt']),KC] = activation - sfa

            elif elem == 'odor_BA':
                for idx, KC in enumerate(active_KC_BA):
                    if params['start_offset'] != (0, 0):
                        start_offset = start_offset_BA[idx]
                        stop_offset = stop_offset_BA[idx]
                        activation_length = int((stop + stop_offset) / params['dt']) - int((start + start_offset) / params['dt'])
                        activation = np.ones(activation_length) * activation_BA[idx]
                        sfa = 0.06 * (1 - np.exp(-9 * np.arange(0, activation_length) / activation_length / 2))
                        KC_input[int((start + start_offset) / params['dt']):int((stop + stop_offset) / params['dt']),
                        KC] = activation - sfa
                    elif params['start_offset'] == (0, 0):
                        activation_length = int(stop / params['dt']) - int(start / params['dt'])
                        activation = np.ones(activation_length) * activation_BA[idx]
                        sfa = 0.06 * (1 - np.exp(-9 * np.arange(0, activation_length) / activation_length / 2))
                        KC_input[int(start / params['dt']):int(stop / params['dt']), KC] = activation - sfa

        elif params['SFA'] ==False:
            if elem == 'odor_AM':
                for idx, KC in enumerate(active_KC_AM):
                    if params['start_offset'] != (0, 0):
                        start_offset = start_offset_AM[idx]
                        stop_offset = stop_offset_AM[idx]
                        activation_length = int((stop + stop_offset) / params['dt']) - int(
                            (start + start_offset) / params['dt'])
                        activation = np.ones(activation_length) * activation_AM[idx]
                        KC_input[int((start + start_offset) / params['dt']):int((stop + stop_offset) / params['dt']),
                        KC] = activation
                    elif params['start_offset'] == (0, 0):
                        activation_length = int(stop / params['dt']) - int(start / params['dt'])
                        activation = np.ones(activation_length) * activation_AM[idx]
                        KC_input[int(start / params['dt']):int(stop / params['dt']), KC] = activation

            elif elem == 'odor_BA':
                for idx, KC in enumerate(active_KC_BA):
                    if params['start_offset'] != (0, 0):
                        start_offset = start_offset_BA[idx]
                        stop_offset = stop_offset_BA[idx]
                        activation_length = int((stop + stop_offset) / params['dt']) - int(
                            (start + start_offset) / params['dt'])
                        activation = np.ones(activation_length) * activation_BA[idx]
                        KC_input[int((start + start_offset) / params['dt']):int((stop + stop_offset) / params['dt']),
                        KC] = activation
                    elif params['start_offset'] == (0, 0):
                        activation_length = int(stop / params['dt']) - int(start / params['dt'])
                        activation = np.ones(activation_length) * activation_BA[idx]
                        KC_input[int(start / params['dt']):int(stop / params['dt']), KC] = activation

    return KC_input


def reinforcement_fig1():

    reinforcement_stim = dict()

    reinforcement_stim['c1'] = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
    reinforcement_stim['d1'] = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0])
    reinforcement_stim['g1'] = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
    reinforcement_stim['f1'] = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0])
    reinforcement_stim['PAM'] = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0])

    return reinforcement_stim


def reinforcement(params):

    reinforcement_stim = dict()
    baseline = np.zeros(int(params['sim_duration'] / params['dt']))

    reinforcement_stim['c1'] = baseline.copy()
    reinforcement_stim['d1'] = baseline.copy()
    reinforcement_stim['f1'] = baseline.copy()
    reinforcement_stim['g1'] = baseline.copy()
    reinforcement_stim['PAM'] = baseline.copy()

    for start, stop in zip(params['r_start'], params['r_stop']):
        # Convert start and stop times to indices
        start_idx = int(start / params['dt'])
        stop_idx = int(stop / params['dt'])
        # Assign the activation level to the specified slice
        reinforcement_stim['c1'][start_idx:stop_idx] = params['r_activation']
        reinforcement_stim['d1'][start_idx:stop_idx] = params['r_activation']
        reinforcement_stim['f1'][start_idx:stop_idx] = params['r_activation']
        reinforcement_stim['g1'][start_idx:stop_idx] = params['r_activation']
        reinforcement_stim['PAM'][start_idx:stop_idx] = params['r_activation']

    return reinforcement_stim


def optogenetic_activation(params):

    optogenetic_stim = dict()
    baseline = np.zeros(int(params['sim_duration'] / params['dt']))

    optogenetic_stim['f1'] = baseline.copy()
    optogenetic_stim['g1'] = baseline.copy()

    for start, stop in zip(params['opt_start'], params['opt_stop']):
        # Convert start and stop times to indices
        start_idx = int(start / params['dt'])
        stop_idx = int(stop / params['dt'])
        # Assign the activation level to the specified slice
        optogenetic_stim['f1'][start_idx:stop_idx] = params['input_ChR_f1']
        optogenetic_stim['g1'][start_idx:stop_idx] = params['input_ChR_g1']

    return optogenetic_stim



def optogenetic_inhibition(params):

    opt_inh = dict()
    baseline = np.zeros(int(params['sim_duration'] / params['dt']))

    opt_inh['f1'] = baseline.copy()
    opt_inh['g1'] = baseline.copy()

    for start, stop in zip(params['opt_inh_start'], params['opt_inh_stop']):
        # Convert start and stop times to indices
        start_idx = int(start / params['dt'])
        stop_idx = int(stop / params['dt'])
        # Assign the activation level to the specified slice
        opt_inh['f1'][start_idx:stop_idx] = params['input_opt_inh_f1']
        opt_inh['g1'][start_idx:stop_idx] = params['input_opt_inh_g1']

    return opt_inh








