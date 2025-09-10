

params = dict(

    N = 200,  # number of model instances to simulate
    dt = 1000.0, # in ms
    sim_duration = 360000, # in ms
    save_path = ,
    rate_monitors = {'KC_r':'KC', 'MBONp_r':'MBONp', 'MBONn_r':'MBONn','readout_r':'readout','c1_r':'DAN_c1', 'd1_r':'DAN_d1', 'f1_r':'DAN_f1', 'g1_r':'DAN_g1', 'PAM_r':'DAN_PAM'},
    weight_monitors = {'KC_MBONp_w' : 'KC_MBONp', 'KC_MBONn_w' : 'KC_MBONn'},

    # number of neurons
    n_KC=100,
    n_MBONp=4,
    n_MBONn=1, # collects the synaptic depression of all PAM DANs

    n_DAN_f1=1,
    n_DAN_g1=1,
    n_DAN_d1=1,
    n_DAN_c1=1,
    n_DAN_PAM=1,

    # Neuron parameters
    tau_DAN_PAM = 0.637312 * 1000,
    b_PAM = 0.193716,
    tau_a_DAN_PAM = 3.243857 * 1000,

    tau_DAN_c1 =8.952809 * 1000,
    b_c1= 13.75433,
    tau_a_DAN_c1 = 0.865808 * 1000,

    tau_DAN_d1 = 11.550999 * 1000,
    b_d1 = 10.237083,
    tau_a_DAN_d1 = 1.24174 * 1000,

    tau_DAN_g1 = 0.667098 * 1000,
    b_g1= 0.273706,
    tau_a_DAN_g1=32.700818 * 1000,

    tau_DAN_f1=0.579765 * 1000,
    b_f1=0.006707,
    tau_a_DAN_f1=61.637608 * 1000,

    tau_MBON=0.5 * 1000,

    # synapses
    w_rew_DANc1=0.54274,
    w_rew_DANd1=0.093441,
    w_rew_DANf1=-0.023435,
    w_rew_DANg1=-0.011659,
    w_rew_DANPAM=0.379457,
    w_pun_DANc1=0.826388,
    w_pun_DANd1=1.002981,
    w_pun_DANf1=0.012125,
    w_pun_DANg1=0.061857,
    w_pun_DANPAM=-0.175809,

    w_KC_MBON_p = 0.7, # weight (initial)
    w_KC_MBON_n = 0.7*4, # weight (initial)
    lr=(-0.0001,-0.0003),# negative value for depression or 0

    # KC activation
    odors = ['odor_AM'],
    odor_start = [30000], # in ms
    odor_stop = [330000], # in ms
    start_offset = (0,5000), # (upper,lower)
    stop_offset = (-5000,5000), # (upper,lower)
    activation = (0.1,0.15),# KC odor activation  (upper,lower)
    pc_active_KC = (5,10),# percentage of odor activated KC adult (5% - 10%) (upper,lower) - Turner 2008
    SFA = True,

    # reinforcement stimulus
    reinforcement = 'salt',
    r_start = [30000], # in ms
    r_stop = [330000], # in ms
    r_activation =1.0, # off=0, on=1
    baseline = (0.01,0.03), # (upper,lower)

    # optogenetic activation
    opt_start = [30000], # in ms,
    opt_stop = [330000], # in ms,
    input_ChR_f1 = 0.0,# 0.5 or 0
    input_ChR_g1 = 0.0,# 0.5 or 0

    # optogenetic silencing
    opt_inh_start = [30000],
    opt_inh_stop = [330000],
    input_opt_inh_f1=0.0, # 0.5 or 0
    input_opt_inh_g1=0.0, # 0.5 or 0

    # neuron in network
    ablation_c1 = 1.0, # 1.0 = neuron in network, 0.0 = neuron ablated
    ablation_d1 = 1.0, # 1.0 = neuron in network, 0.0 = neuron ablated
    ablation_f1 = 1.0, # 1.0 = neuron in network, 0.0 = neuron ablated
    ablation_g1 = 1.0  # 1.0 = neuron in network, 0.0 = neuron ablated

)
