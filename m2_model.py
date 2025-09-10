from ANNarchy import *
import stimulus as stimulus
import sys
# numpy is imported with ANNArchy


def build_model(params):

    '''

      Creates the model architecture (neuron templates, neuron populations, synapse models, projections, monitors) and compiles it

      :return: net (ANNarchy network object), network objects(pops, syns, mons),  KC_input

    '''


    # DAN activity baseline (same for all DANs in the network)
    baseline = 0.0
    learning_rate = 0.0

    # initialize dict for neuron populations and synapses
    pops = dict()
    syns = dict()

    # KC input
    KC_input = stimulus.odor_stim(params)
    pops['KC'] = TimedArray(rates=KC_input) # create input population with defined rate

    #DAN input and optogenetic activation/inactivation
    #external reward input
    reinf_input = stimulus.reinforcement(params)
    DAN_c1_input = reinf_input['c1'].reshape(-1, 1)
    DAN_d1_input = reinf_input['d1'].reshape(-1, 1)
    DAN_f1_input = reinf_input['f1'].reshape(-1, 1)
    DAN_g1_input = reinf_input['g1'].reshape(-1, 1)
    DAN_PAM_input = reinf_input['PAM'].reshape(-1, 1)
    pops['input_c1'] = TimedArray(rates=DAN_c1_input)  # create input population with defined rate
    pops['input_d1'] = TimedArray(rates=DAN_d1_input)
    pops['input_f1'] = TimedArray(rates=DAN_f1_input)
    pops['input_g1'] = TimedArray(rates=DAN_g1_input)
    pops['input_PAM'] = TimedArray(rates=DAN_PAM_input)

    # Input optogenetic activation (ChR2XXL)
    optogenetic_input = stimulus.optogenetic_activation(params)
    DAN_f1_opt_inp = optogenetic_input['f1'].reshape(-1, 1)
    DAN_g1_opt_inp = optogenetic_input['g1'].reshape(-1, 1)
    pops['opt_f1'] = TimedArray(rates=DAN_f1_opt_inp)
    pops['opt_g1'] = TimedArray(rates=DAN_g1_opt_inp)

    # inpt optogenetic inactivation
    opt_inh = stimulus.optogenetic_inhibition(params)
    DAN_f1_opt_silence = opt_inh['f1'].reshape(-1, 1)
    DAN_g1_opt_silence = opt_inh['g1'].reshape(-1, 1)
    pops['silence_f1'] = TimedArray(rates=DAN_f1_opt_silence)
    pops['silence_g1'] = TimedArray(rates=DAN_g1_opt_silence)


    # neuron templates
    # MBON
    #### IMPORTANT: mp (membran potential is the calcium signal here --> therfor there is no transfer function). Annarchy looks for keyword mp and r i equation during model compilation
    MBON_temp = Neuron(
        parameters="""
            tau = """+str(params['tau_MBON'])+"""
        """,
        equations="""
            dmp/dt = (-mp + sum(exc)) / tau
            r = mp : min = 0.0
            DAN_PAM_r = sum(copy)
            DAN_DL1_r = sum(copy)   
        """)

    # readout neuron (score is sum(MBONn - MBONp)
    readout_temp = Neuron(
        equations="""
            r = sum(exc) + sum(inh)
        """
    )

    # DAN
    DAN_PAM_temp = Neuron(
        parameters="""
            b = """+str(params['b_PAM'])+"""
            tau = """+str(params['tau_DAN_PAM'])+"""
            tau_a = """+str(params['tau_a_DAN_PAM'])+"""  
            bl = """ + str(baseline) + """
           
        """,
        equations="""
            dmp/dt = (-mp + sum(exc) -a * b) / tau
            da/dt = (-a + mp) / tau_a
            r =  mp + (bl*4)
        """)

    DAN_f1_temp = Neuron(
        parameters="""
            b = """ + str(params['b_f1']) + """
            tau = """ + str(params['tau_DAN_f1']) + """
            tau_a = """ + str(params['tau_a_DAN_f1']) + """  
            bl = """ + str(baseline) + """
            af = """ + str(params['ablation_f1']) + """
        """,
        equations="""
            dmp/dt = (-mp + sum(exc) + sum(inh) -a * b) / tau 
            da/dt = (-a + mp) / tau_a
            r = (mp + bl) * af
        """)

    DAN_g1_temp = Neuron(
        parameters="""
            b = """ + str(params['b_g1']) + """
            tau = """ + str(params['tau_DAN_g1']) + """
            tau_a = """ + str(params['tau_a_DAN_g1']) + """  
            bl = """ + str(baseline) + """
            af = """ + str(params['ablation_g1']) + """
        """,
        equations="""
            dmp/dt = (-mp + sum(exc) + sum(inh) -a * b) / tau  
            da/dt = (-a + mp) / tau_a
            r = (mp + bl) * af
        """)

    DAN_d1_temp = Neuron(
        parameters="""
            b = """ + str(params['b_d1']) + """
            tau = """ + str(params['tau_DAN_d1']) + """
            tau_a = """ + str(params['tau_a_DAN_d1']) + """  
            bl = """ + str(baseline) + """
            af = """ + str(params['ablation_d1']) + """
        """,
        equations="""
            dmp/dt = (-mp + sum(exc) + sum(inh) -a * b) / tau
            da/dt = (-a + mp) / tau_a
            r = (mp + bl) * af
        """)

    DAN_c1_temp = Neuron(
        parameters="""
            b = """ + str(params['b_c1']) + """
            tau = """ + str(params['tau_DAN_c1']) + """
            tau_a = """ + str(params['tau_a_DAN_c1']) + """  
            bl = """ + str(baseline) + """
            af = """ + str(params['ablation_c1']) + """
        """,
        equations="""
            dmp/dt = (-mp + sum(exc) + sum(inh) -a * b) / tau
            da/dt = (-a + mp) / tau_a 
            r = (mp + bl) * af
        """)

    # Neuron populations
    pops['readout'] = Population(geometry=1, neuron=readout_temp, name='readout')

    pops['MBONp'] = Population(geometry=params['n_MBONp'], neuron=MBON_temp, name='MBONp')
    pops['MBONn'] = Population(geometry=params['n_MBONn'], neuron=MBON_temp, name='MBONn')

    pops['DAN_PAM'] = Population(geometry=params['n_DAN_PAM'], neuron=DAN_PAM_temp, name='DAN_PAM')
    pops['DAN_f1'] = Population(geometry=params['n_DAN_f1'], neuron=DAN_f1_temp, name='DAN_f1')
    pops['DAN_g1'] = Population(geometry=params['n_DAN_g1'], neuron=DAN_g1_temp, name='DAN_g1')
    pops['DAN_d1'] = Population(geometry=params['n_DAN_d1'], neuron=DAN_d1_temp, name='DAN_d1')
    pops['DAN_c1'] = Population(geometry=params['n_DAN_c1'], neuron=DAN_c1_temp, name='DAN_c1')

    if params['reinforcement'] == 'salt':

        syns['r_input_DAN_PAM'] = Projection(pre=pops['input_PAM'], post=pops['DAN_PAM'], target='exc').connect_one_to_one(params['w_pun_DANPAM'])  # deliver external reward signal to DANs
        syns['r_input_DAN_f1'] = Projection(pre=pops['input_f1'], post=pops['DAN_f1'], target='exc').connect_one_to_one(params['w_pun_DANf1'])
        syns['r_input_DAN_g1'] = Projection(pre=pops['input_g1'], post=pops['DAN_g1'], target='exc').connect_one_to_one(params['w_pun_DANg1'])
        syns['r_input_DAN_d1'] = Projection(pre=pops['input_d1'], post=pops['DAN_d1'], target='exc').connect_one_to_one(params['w_pun_DANd1'])
        syns['r_input_DAN_c1'] = Projection(pre=pops['input_c1'], post=pops['DAN_c1'], target='exc').connect_one_to_one(params['w_pun_DANc1'])

    if params['reinforcement'] == 'sugar':

        syns['r_input_DAN_PAM'] = Projection(pre=pops['input_PAM'], post=pops['DAN_PAM'], target='exc').connect_one_to_one(params['w_rew_DANPAM'])  # deliver external reward signal to DANs
        syns['r_input_DAN_f1'] = Projection(pre=pops['input_f1'], post=pops['DAN_f1'], target='exc').connect_one_to_one(params['w_rew_DANf1'])
        syns['r_input_DAN_g1'] = Projection(pre=pops['input_g1'], post=pops['DAN_g1'], target='exc').connect_one_to_one(params['w_rew_DANg1'])
        syns['r_input_DAN_d1'] = Projection(pre=pops['input_d1'], post=pops['DAN_d1'], target='exc').connect_one_to_one(params['w_rew_DANd1'])
        syns['r_input_DAN_c1'] = Projection(pre=pops['input_c1'], post=pops['DAN_c1'], target='exc').connect_one_to_one(params['w_rew_DANc1'])

    # Optogenetic (ChR2XXL) activation of DANs
    syns['opt_inp_f1'] = Projection(pre=pops['opt_f1'], post=pops['DAN_f1'], target='exc').connect_one_to_one(1.0)
    syns['opt_inp_g1'] = Projection(pre=pops['opt_g1'], post=pops['DAN_g1'], target='exc').connect_one_to_one(1.0)

    # optogenetic silencing of DANs (no baseline)
    syns['opt_silencing_f1'] = Projection(pre=pops['silence_f1'], post=pops['DAN_f1'], target='inh').connect_one_to_one(-1.0)
    syns['opt_silencing_g1'] = Projection(pre=pops['silence_g1'], post=pops['DAN_g1'], target='inh').connect_one_to_one(-1.0)

    # synapse models
    # default synapse is: psp = w * pre.r
    KC_MBONn_plast = Synapse(
        parameters="""
            lr = """ + str(learning_rate) + """
            max_weight = """ + str(params['w_KC_MBON_n']) + """
        """,
        equations="""
            psp = w * pre.r
            dw/dt = pre.r * post.DAN_PAM_r * lr : max=max_weight
        """
    )

    KC_MBONp_plast = Synapse(
        parameters="""
                lr = """ + str(learning_rate) + """
                max_weight = """ + str(params['w_KC_MBON_p']) + """
            """,
        equations="""
                psp = w * pre.r
                dw/dt = pre.r * post.DAN_DL1_r * lr : max=max_weight
            """
    )

    syns['MBONp_readout'] = Projection(pre=pops['MBONp'], post=pops['readout'], target='inh').connect_all_to_all(-1.0)
    syns['MBONn_readout'] = Projection(pre=pops['MBONn'], post=pops['readout'], target='exc').connect_all_to_all(1.0)

    syns['KC_MBONp'] = Projection(pre=pops['KC'], post=pops['MBONp'], target="exc", synapse=KC_MBONp_plast)
    syns['KC_MBONp'].connect_all_to_all(params['w_KC_MBON_p'])

    syns['KC_MBONn'] = Projection(pre=pops['KC'], post=pops['MBONn'], target="exc", synapse=KC_MBONn_plast)
    syns['KC_MBONn'].connect_all_to_all(params['w_KC_MBON_n'])

    syns['provide_DAN_PAM_r'] = Projection(pre=pops['DAN_PAM'], post=pops['MBONn'], target="copy").connect_all_to_all(1.0) # this synaspse only informs MBONn about the DANp_rate at dt-1

    # each DL1 DAN innervates one appetitive compartment
    syns['provide_DAN_c1_r'] = Projection(pre=pops['DAN_c1'], post=pops['MBONp'][0], target="copy").connect_one_to_one(1.0)
    syns['provide_DAN_d1_r'] = Projection(pre=pops['DAN_d1'], post=pops['MBONp'][1], target="copy").connect_one_to_one(1.0)
    syns['provide_DAN_f1_r'] = Projection(pre=pops['DAN_f1'], post=pops['MBONp'][2], target="copy").connect_one_to_one(1.0)
    syns['provide_DAN_g1_r'] = Projection(pre=pops['DAN_g1'], post=pops['MBONp'][3], target="copy").connect_one_to_one(1.0)

    # initialize variable monitors
    mons = dict()
    # rate monitors
    for i in params['rate_monitors'].items():
        mon = i[0]
        target = i[1]
        mons[mon] = Monitor(pops[target],'r')
    # weight monitors
    for i in params['weight_monitors'].items():
        mon = i[0]
        target = i[1]
        mons[mon] = Monitor(syns[target],'w')


    net = Network(everything=True)
    net.compile()

    return net, pops, syns, mons, KC_input












