""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
August 2023
Imperial College London
Department of Civil Engineering

1. Relationships for the LIF model between MN properties
Derived from Table 4 in Caillet, A. H., Phillips, A. T., Farina, D., & Modenese, L. (2022). Mathematical relationships between spinal motoneuron properties. Elife, 11, e76489.

2. LIF model with the MN-specific parameters
Derived from the Github code provided with Caillet, A. H., Phillips, A. T., Farina, D., & Modenese, L. (2022). Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling. PLOS Computational Biology, 18(9), e1010556.

"""

import numpy as np

# Relationships between MN properties
def R_S_func(S, kR):
    return kR / S**2.43

def C_S_func(S, Cm_rec): 
    return Cm_rec * S  # [F]

def tau_R_C_func(R,C):
    return R * C  # [s]

# LIF model
def RC_solve_func(I, time_list, Size, step_size, ARP, ARP_rand=10): 
    '''
    This function solves the ODE of integrator RC circuit using a hybrid analytical 
    numerical method of Fourier transforms and convolution. 
    This method is computationally cheaper than a numerical Runge-Kutta method
    For simplicity without loss of generality, V varies in [0,27mV] rather than
    in [-85;-58mV]. The precision is given by step_size. Inputs and outputs are in
    [s]. Two different values of specific capacitance are given for recruitment and
    derecruitment phases. 
    MN Size defines all other relevant MN parameters R, C and tau

    Parameters
    ----------
    I : function of time, [A]
        Time-course of the current input to the system
    time_list : array, [s]
        Array of time in seconds
    Size : float, [m2]
        The size of the MN. This parameter controls the whole model and the 
        value of the other parameters
    Cm_rec : float, [F/m2]
        Specific capacitance of the RC circuit for the recruitment phase:
        t<t_plateau_end
    ARP : MN-specific ARP value [s], float

    Returns
    -------
    tim_list : numpy array, [s]
        List of time instants of time step 'step_size' at which the values of 
        the membrane potential V are calculated
    V : numpy array, [V]
        List of calculated membrane potentials at each time instants V(tim_list)
    firing_times_arr : numpy array, [s]
        List of the time instants at which an action potential is fired
    parameters: numpy array
        List of important parameters related to the size of the MN input to 
        the func

    '''
    # MN PARAMETERS
    Vth = 27 * 10**-3  # [V] 
    kR = 1.68 * 10**-10  # gain between input resistance R and MN size S in the elife Table 4
    Cm_rec = 1.3 * 10**-2  # Constant value of specific capacitance
    ARP_ini = ARP  # [s] 
    R = R_S_func(Size, kR)  # [Ohm] 
    C = C_S_func(Size, Cm_rec)  #[F]
    tau = tau_R_C_func(R, C)  # [s]
    parameters = [Vth, ARP_ini, R, C, tau]

    # MEMBRANE VOLTAGE INITIALIZATION
    V = np.zeros(len(time_list))
    firing_times_arr = np.array([])
    t_fire = -7 * tau  # initializing

    #SOLVING
    for i in range(len(time_list)):  # solving at each time instant
        # INITIAL TIME
        if i == 0:
            Vnt = R * step_size / tau * I(time_list[0])  # Initial condition on V
        else:
            nt = time_list[i]  # time instant in seconds
            Vnt = np.exp(-step_size / tau) * Vnt + R * step_size / tau * I(nt)  # LIF equation of MN membrane potential

            #FIRING TIME
            if Vnt > Vth:  # If the threshold is reached at time nt
                Vnt = 0  # The potential is reset to rest state
                V[i] = 0  # V(nt)=0
                firing_times_arr = np.append(firing_times_arr, nt)  # Storing the firing time nt
                t_fire = nt  # Reset tfire with the latest found value nt
                ARP = np.random.normal(ARP_ini, ARP_ini / ARP_rand)  # Randomly vary the ARP value following a Gaussian probabilistic curve of sigma=ARP/10, to model the fulctuations in firing rate saturation

            #REFRACTORY PERIOD    
            elif nt > t_fire and nt < t_fire + ARP:  # If during the refractory period
                V[i] = 0  # The potential remains at resting state
                Vnt = 0
            # MEMBRANE CHARGING    
            else:
                V[i] = Vnt  # In any other cases, calculate V(nt) and store
    return V, firing_times_arr, parameters
