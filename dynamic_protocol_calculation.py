import myokit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def set_dynamic_protocol(periods, level, start, duration, n_stimulus):
    """ 
    create dynamic stimulation Myokit protocol
    
    :param 
    periods -- an ordered list of stimulation periods in ms
    level -- Myokit protocol parameter
    start -- Myokit protocol parameter
    duration -- Myokit protocol parameter
    n_stimulus -- the number of consecutive calculated stimuli
    without changing the stimulation period
    
    :return:
    myokit._protocol.Protocol
    """
    
    n_periods = len(periods)
    protocol = myokit.Protocol()
    for i, period in enumerate(periods):
        protocol.schedule(level, start, duration,
                          period=period, multiplier=n_stimulus)
        if i+1 < n_periods:
            start += period * (n_stimulus - 1) + periods[i+1]

    return protocol


def run_dynamic_0D_protocol(model, parameters, n_stimulus, duration,
                            state=None, log=["membrane.V"],
                            maxstep=None, level=1, start=0):
    """
    run the dynamic stimulation protocol using the CVODES solver
    
    The dynamic stimulation protocol assumes a gradual decrease 
    in the stimulation period in small steps. 
    This protocol implementation takes into account changes 
    in model parameters as the period changes.
    
    :param
    model -- myokit._model_api.Model
    parameters -- Pandas table (DataFrame) with absolute values of model parameters, 
    where 
    - index -- periods,
    - columns -- parameter names in Myokit model
    n_stimulus -- the number of consecutive calculated stimuli
    without changing the period
    duration -- Myokit protocol parameter
    state -- the model's initial state in a format acceptable to Myokit 
    (e.g., a list ordered in the order in which the variables 
    in the model file are mentioned)
    log -- variables for which the result should be written
    maxstep -- maximum step size for CVODE
    level -- Myokit protocol parameter
    start -- Myokit protocol parameter
    
    return:
    list containing myokit datalog simulations for different stimulation periods, 
    ordered according to the parameter table indexes 
    """

    periods, n_periods = parameters.index, len(parameters.index)
    protocol = set_dynamic_protocol(periods, level, start,
                                    duration, n_stimulus)

    simulation = myokit.Simulation(model, protocol)
    simulation.set_max_step_size(dtmax=maxstep)
    state = model.state() if state is None else state
    simulation.set_state(state)

    datalog = []
    for i, period in enumerate(periods):
        for key, value in parameters.loc[period].items():
            simulation.set_constant(key, value)
        if i + 1 < n_periods:
            simulation_time = period * (n_stimulus - 1) + periods[i+1]
        else:
            simulation_time = period * n_stimulus

        d_ = simulation.run(simulation_time, log_interval=1.0, log=log)
        datalog.append(d_)

    return datalog


def apd(signal, level):
    """
    calculate APD to level
    
    param: signal -- 1D array, level -- float
  
    return: float
    """

    if np.all(signal <= level):
        return 0.0
    start = np.where(signal > level)[0][0]

    if np.all(signal[start:] >= level):
        return np.inf
    finish = np.where(signal[start:] < level)[0][0]

    return finish


def get_restitution_curve(datalog, periods, n_stimulus):
    """
    get APD restition curve 
    
    param:
    datalog -- list containing myokit datalog simulations for different stimulation periods, 
    ordered according to the 'periods'
    periods -- an ordered list of stimulation periods in ms
    n_stimulus -- the number of consecutive calculated stimuli
    without changing the stimulation period
    
    return:
    (periods, 
    mean values of APD for each period, 
    STD values of APD for each period)
    
    """
    n_periods = len(periods)
    key = 'membrane.V'

    apds = np.empty((n_stimulus, n_periods))

    for period_index, period in enumerate(periods):
        record = np.array(datalog[period_index][key])
        n_tail = period*n_stimulus - len(record)
        extended_record = np.append(record, [np.nan]*n_tail)
        aps = np.split(extended_record, n_stimulus)

        for ap_index, ap in enumerate(aps):
            min_ = np.nanmin(ap)
            max_ = np.nanmax(ap)
            level90 = min_ + (max_ - min_) * (1 - 0.9)

            apds[ap_index, period_index] = apd(ap, level90)

    return periods, apds.mean(axis=0), apds.std(axis=0)


if __name__ == '__main__':
    # calculation parameters
    is_datalog_saver = False
    model_ids = [1, 2, 3, 4]
    model_types = ['C', 'L']
    max_step = None    # 0.005
    n_stimulus = 40
    log = ['membrane.V']

    parameter_filename = './model_parameters/model{}{}_parameters.csv'
    model_filename = 'updated_Gattoni2016_model_1C.mmt'
    steadystate_filename = './model_parameters/steady_states.csv'

    # figure parameters
    line_styles = {'C': '--', 'L': '-'}
    colors = {'C': 'b', 'L': 'r'}
    ylim = (0, 300)
    xticks = [90, 250, 500, 1000]
    yticks = [50, 100, 200, 250]
    xlabel = 'BCL (ms)'
    ylabel = 'APD90 (ms)'
    
    # dynamic protocol simulation for models 1C-4C, 1L-4L
    model, protocol, _ = myokit.load(model_filename)
    duration = protocol.events()[0].duration()
    steady_states = pd.read_csv(steadystate_filename)

    datalog = {}
    for model_id in model_ids:
        for model_type in model_types:
            label = f'{model_id}{model_type}'
            parameters = pd.read_csv(
                parameter_filename.format(model_id, model_type), index_col=0
                )
            datalog[label] = run_dynamic_0D_protocol(
                model, parameters,
                n_stimulus, duration,
                state=steady_states[label], maxstep=max_step, log=log,
                )
    if is_datalog_saver:
        np.savez('datalog.npz', datalog=datalog)

    # calculating and drawing APD restitution curves for models
    sns.set(font_scale=2.25, style='whitegrid')

    nrow = 1
    ncol = len(model_ids)
    fig = plt.figure(figsize=(ncol*6, nrow*6))
    periods = parameters.index

    for ax_index, model_id in enumerate(model_ids, 1):
        ax = plt.subplot(nrow, ncol, ax_index)
        for model_type in model_types:
            label = f'{model_id}{model_type}'
            periods, mean, std = get_restitution_curve(datalog[label],
                                                       periods, n_stimulus
                                                       )
            plt.errorbar(periods, mean, std, label=model_type,
                         ls=line_styles[model_type], c=colors[model_type],
                         marker='o', lw=2, ms=7, capsize=5
                         )
        plt.ylim(*ylim)
        plt.xticks(xticks)
        plt.yticks(yticks)
        if ax_index == 1:
            plt.ylabel(ylabel)
        if ax_index == ncol:
            plt.legend()
        plt.xlabel(xlabel)
        plt.title(f'Model {model_id}')

    fig.savefig('rc.png', format='png', facecolor='#fff', bbox_inches='tight')
