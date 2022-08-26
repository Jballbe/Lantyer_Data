#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 16:16:12 2022

@author: julienballbe
"""

import numpy as np
import pandas as pd
import tqdm

from plotnine import ggplot, geom_line, aes, geom_abline, geom_point, geom_text, labels,geom_histogram,ggtitle

import scipy
from scipy.stats import linregress
from scipy import optimize
import random


from matplotlib.colors import LogNorm
import warnings
import pandas
import os
from lmfit.models import LinearModel, StepModel, ExpressionModel, Model,ExponentialModel,ConstantModel,GaussianModel
from lmfit import Parameters, Minimizer,fit_report
from plotnine.scales import scale_y_continuous,ylim,xlim,scale_color_manual
from plotnine.labels import xlab
from plotnine.coords import coord_cartesian
from sklearn.metrics import mean_squared_error
import seaborn as sns
from scipy.interpolate import splrep, splev
from scipy.misc import derivative
import ast
from scipy import signal
#%%


#%%5ms
def compute_features(file_suffix,per_time=False,first_x_ms=0,per_nth_spike=False,first_nth_spike=0):
    FI_params_col=['Cell_id',
           "Fit",
           "best_single_QNRMSE",
           "Best_single_amplitude",
           "Best_single_center",
           "Best_single_sigma",
           "Best_compo_QNRMSE",
           "Best_heaviside_step",
           "Best_sigmoid_amplitude",
           "Best_sigmoid_center",
           "Best_sigmoid_sigma"]
    FI_params_units=['--',
           '--',
           '--',
           'Hz',
           'pA',
           'Hz/pA',
           '--',
           'pA',
           'Hz',
           'pA',
           'Hz/pA']
    first_FI_params_line=pd.Series(FI_params_col,index=FI_params_col)
    second_FI_params_line=pd.Series(FI_params_units,index=FI_params_col)
    first_two_FI_params_lines=pd.DataFrame([first_FI_params_line,second_FI_params_line])
    
    FI_features_col=["Cell_id",
                     "Fit",
                     "Gain",
                     "Threshold",
                     "Saturation"]
    
    FI_features_units=["--",
                       "--",
                       "Hz/pA",
                       "pA",
                       "Hz"]
    
    first_FI_feature_line=pd.Series(FI_features_col,index=FI_features_col)
    second_FI_feature_line=pd.Series(FI_features_units,index=FI_features_col)
    first_two_FI_feature_lines=pd.DataFrame([first_FI_feature_line,second_FI_feature_line])
    
    Adapt_col=['Cell_id','Starting_frequency_A','Adapt_cst_B','Steady_state_frequency_C']
    Adapt_units=['--',
            'Hz_rel',
            'Spike_index',
            'Hz_rel']
    first_Adapt_line=pd.Series(Adapt_col,index=Adapt_col)
    second_Adapt_line=pd.Series(Adapt_units,index=Adapt_col)
    first_two_Adapt_lines=pd.DataFrame([first_Adapt_line,second_Adapt_line])
    
    
    
            
    cell_id_list=pd.read_csv(filepath_or_buffer="/Users/julienballbe/My_Work/Lantyer_Data/Cell_id_list.csv")
    cell_id_list=cell_id_list.iloc[:,1]
    
    
    
    
    

    for cell_id in tqdm.tqdm(cell_id_list):
        
        heaviside_fit_results=heaviside_fit_sigmoid_Lantyer(cell_id,per_time=per_time,first_x_ms=first_x_ms,per_nth_spike=per_nth_spike,first_nth_spike=first_nth_spike)
        new_params_line=pd.Series([str(cell_id),*heaviside_fit_results],
                       index=FI_params_col)
        my_dataframe=pd.DataFrame(columns=FI_params_col)

        my_line=pd.Series([str(cell_id),*heaviside_fit_results],index=FI_params_col)
        my_dataframe=my_dataframe.append(my_line,ignore_index=True)
        
        
        
        first_two_FI_params_lines=first_two_FI_params_lines.append(new_params_line,ignore_index=True)
        #
       
        first_two_FI_feature_lines=pd.concat([first_two_FI_feature_lines,compute_f_I_params_Lantyer(my_dataframe,per_time=per_time,first_x_ms=first_x_ms,per_nth_spike=per_nth_spike,first_nth_spike=first_nth_spike)])
        
        #
        
        new_adapt_line=pd.Series([str(cell_id),*fit_exponential_decay_Lantyer(cell_id,per_time=per_time,first_x_ms=first_x_ms,per_spike_nb=per_nth_spike,first_nth_spikes=first_nth_spike)],
                                 index=Adapt_col)
        
        first_two_Adapt_lines=first_two_Adapt_lines.append(new_adapt_line,ignore_index=True)
        
    
        
    
    
   
    first_two_FI_params_lines=first_two_FI_params_lines.iloc[1:,:]
    first_two_FI_feature_lines=first_two_FI_feature_lines.iloc[1:,:]
    first_two_Adapt_lines=first_two_Adapt_lines.iloc[1:,:]
    
    first_two_FI_params_lines.to_csv(path_or_buf=str('/Users/julienballbe/My_Work/Lantyer_Data/Feature_computations/FI_features/FI_params_'+str(file_suffix)+'.csv'),na_rep='nan',index=False)
    first_two_FI_feature_lines.to_csv(path_or_buf=str('/Users/julienballbe/My_Work/Lantyer_Data/Feature_computations/FI_features/FI_features_'+str(file_suffix)+'.csv'),na_rep='nan',index=False)  
    first_two_Adapt_lines.to_csv(path_or_buf=str('/Users/julienballbe/My_Work/Lantyer_Data/Feature_computations/Adaptation_features/Adapt_features_'+str(file_suffix)+'.csv'),na_rep='nan',index=False)  
    
    
   
    
    print("Done")

#%%


def import_spike_time_table(cell_id):
    stim_spike_file= pd.read_csv(filepath_or_buffer=str('/Users/julienballbe/My_Work/Lantyer_Data/Stim_spike_tables/'+str(cell_id)+'_stim_spike_table.csv'))
    stim_spike_file=stim_spike_file.loc[:,'Trace_id':]
    stim_spike_table=stim_spike_table=pd.DataFrame(columns=['Trace_id',
                                           "Sweep_id",
                                           'Stim_amp_pA',
                                           'Stim_start_s',
                                           'Stim_end_s', 
                                           'Spike_times_s',
                                           'Spike_thresh_time_s',
                                           'Spike_thresh_pot_mV',
                                           'Spike_peak_time_s',
                                           'Spike_peak_pot_mV',
                                           'Spike_upstroke_time_s',
                                           'Spike_upstroke_pot_mV',
                                           'Spike_downstroke_time_s',
                                           'Spike_downstroke_pot_mV',
                                           'Trough_time_s',
                                           'Trough_pot_mV'])
    
    for line in range(stim_spike_file.shape[0]):
        new_line=pd.Series([str(stim_spike_file.loc[line,"Trace_id"]),
                      int(stim_spike_file.loc[line,"Sweep_id"]),
                      stim_spike_file.loc[line,"Stim_amp_pA"],
                      stim_spike_file.loc[line,"Stim_start_s"],
                      stim_spike_file.loc[line,"Stim_end_s"],
                      np.array(pd.eval(stim_spike_file.loc[line,"Spike_times_s"])),
                      np.array(pd.eval(stim_spike_file.loc[line,"Spike_thresh_time_s"])),
                      np.array(pd.eval(stim_spike_file.loc[line,"Spike_thresh_pot_mV"])),
                      np.array(pd.eval(stim_spike_file.loc[line,"Spike_peak_time_s"])),
                      np.array(pd.eval(stim_spike_file.loc[line,"Spike_peak_pot_mV"])),
                      np.array(pd.eval(stim_spike_file.loc[line,"Spike_upstroke_time_s"])),
                      np.array(pd.eval(stim_spike_file.loc[line,"Spike_upstroke_pot_mV"])),
                      np.array(pd.eval(stim_spike_file.loc[line,"Spike_downstroke_time_s"])),
                      np.array(pd.eval(stim_spike_file.loc[line,"Spike_downstroke_pot_mV"])),
                      np.array(pd.eval(stim_spike_file.loc[line,"Trough_time_s"])),
                      np.array(pd.eval(stim_spike_file.loc[line,"Trough_pot_mV"]))],
                            index=['Trace_id',
                            "Sweep_id",
                            'Stim_amp_pA',
                            'Stim_start_s',
                            'Stim_end_s', 
                            'Spike_times_s',
                            'Spike_thresh_time_s',
                            'Spike_thresh_pot_mV',
                            'Spike_peak_time_s',
                            'Spike_peak_pot_mV',
                            'Spike_upstroke_time_s',
                            'Spike_upstroke_pot_mV',
                            'Spike_downstroke_time_s',
                            'Spike_downstroke_pot_mV',
                            'Trough_time_s',
                            'Trough_pot_mV'])
        stim_spike_table = stim_spike_table.append(new_line, ignore_index=True)
    
    return stim_spike_table
#to get spike time : x=pd.eval(mytest.iloc[7,6])
def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))

#%% Download files



def filter_trace(table,cutoff_freq,do_plot=False):
    '''
    Apply a low pass filter on the stimulus trace 

    Parameters
    ----------
    table : DataFrame
        Stimulus trace table, 2 columns
        First column = 'Time_s'
        Second column = 'Stimulus_amplitude_pA'
    cutoff_freq : float
        Cut off frequency to use in the low pass filter in Hz.
    do_plot : Bool, optional
        The default is False.

    Returns
    -------
    table : DataFrame
        Stimulus trace table, 3 columns
        First column = 'Time_s'
        Second column = 'Stimulus_amplitude_pA'.
        Third column = "Filtered_Stimulus_trace_pA" --> filtered stimulus trace

    '''
    
    sampling_frequency=table.shape[0]/table.iloc[-1,0] #Hz
    
    b, a = scipy.signal.butter(3,Wn=cutoff_freq, fs=sampling_frequency)

    table['Filtered_Stimulus_trace_pA']=scipy.signal.filtfilt(b, a, table.iloc[:,1])

    if do_plot==True:
        myplot=ggplot(table,aes(x=table.iloc[:,0],y=table.iloc[:,1]))+geom_line(color='blue')+geom_line(table,aes(x=table.iloc[:,0],y=table.iloc[:,2]),color='red')+coord_cartesian(xlim=(0.04,0.06))
        myplot
                                               
    return table                

def get_autocorr(table,time_shift,do_plot=False):
    '''
    Compute autocorrelation at each time point for a stimulus trace table

    Parameters
    ----------
    table : DataFrame
        Stimulus trace table, 3 columns
        First column = 'Time_s'
        Second column = 'Stimulus_amplitude_pA'.
        Third column = "Filtered_Stimulus_trace_pA" 
    time_shift : float
        Time shift in s .
    do_plot : Bool, optional
        The default is False.

    Returns
    -------
    table : DataFrame
        Stimulus trace table, 6 columns
        First column = 'Time_s'
        Second column = 'Stimulus_amplitude_pA'.
        Third column = "Filtered_Stimulus_trace_pA" 
        4th column = "Filtered_Stimulus_trace_derivative_pA/ms"
        5th column = "Shifted_trace" --> Filtered stiumulus trace derivatove shifted by 'time_shift'
        6th column = 'Autocorrelation --> Autocorrelation between 4th and 5th column
    '''

    shift=int(time_shift/(table.iloc[1,0]-table.iloc[0,0]))

    table["Shifted_trace"]=table['Filtered_Stimulus_trace_derivative_pA/ms'].shift(-shift)

    table['Autocorrelation']=table['Filtered_Stimulus_trace_derivative_pA/ms']*table["Shifted_trace"]

    
    if do_plot==True:

        myplot=ggplot(table,aes(x=table.loc[:,'Time_s'],y=table.loc[:,'Filtered_Stimulus_trace_derivative_pA/ms']))+geom_line(color='blue')+geom_line(table,aes(x=table.loc[:,'Time_s'],y=table.loc[:,'Autocorrelation']),color='red')
        myplot+=geom_abline(aes(intercept=-10000,slope=0))
        myplot+=xlab(str("Time_s; Time_shift="+str(time_shift)))
        myplot

    return table
    

def get_stim_onset(stim_table):
    '''
    Compute from stimulus trace table the stimulus onset, and its duration by using autocorrelation

    Parameters
    ----------
    stim_table : DataFrame
        Stimulus trace table, 2 columns
        First column = 'Time_s'
        Second column = 'Stimulus_amplitude_pA'

    Returns
    -------
    best_autocorr : float
        lowest autocorrelation coefficient encontered
    best_stim_start : float
        Stimulus onset time point estimated from lowest autocorrelation coef (in s)
    best_time_autocorr : float
        Stimulus duration estimated from time shift used for best autocorrelation coefficient (in s)

    '''
    stim_table=filter_trace(stim_table,2000)
    
    stim_table=get_smooth_deriv_stim(stim_table)

    limit=stim_table.shape[0]-int(0.05/(stim_table.iloc[1,0]-stim_table.iloc[0,0])) #remove last 50ms of signal (potential step)

    stim_table.loc[limit:,"Filtered_Stimulus_trace_derivative_pA/ms"]=np.nan

    best_autocorr=0
    best_time_autocorr=0
    best_stim_start=0
    for i in np.arange(0.497,0.503,0.001):

        stim_table=get_autocorr(stim_table,i,do_plot=False)
        if np.nanmin(stim_table.loc[:,'Autocorrelation'])<best_autocorr:

            best_autocorr=np.nanmin(stim_table.loc[:,'Autocorrelation'])
            best_time_autocorr=i
            best_stim_start=stim_table[stim_table['Autocorrelation']==np.nanmin(stim_table['Autocorrelation'])].iloc[0,0]
    return best_autocorr,best_stim_start,best_time_autocorr
    
    
    
def find_Stim_onset(file,cell_id,Train,max_sweep):
    stim_onset_table=pd.DataFrame(columns=['Sweep_id','Stim_start','Stim_end', 'Stim_duration'])

    for current_sweep in range (1,max_sweep+1): #Loop to determine stimulus start and stimulus end over a train of stimulus 
        current_id=str('Trace_'+str(cell_id)+'_'+str(Train)+'_'+str(current_sweep))
        first_current_stim_trace= pd.DataFrame(file[str(current_id+'_1')],columns=['Time_s','Stimulus_amplitude_pA'])
        #first_current_stim_trace.loc[:,'Stimulus_amplitude_pA']*=1e12
        
        
        autocorr,stim_start,stim_duration=get_stim_onset(first_current_stim_trace)
        stim_end=stim_start+stim_duration     
        new_line=pd.Series([current_sweep,stim_start,stim_end,stim_duration],index=['Sweep_id','Stim_start','Stim_end', 'Stim_duration'])
        stim_onset_table=stim_onset_table.append(new_line,ignore_index=True)
    
    stim_start=stim_onset_table['Stim_start'].mode()[0]
    stim_duration=stim_onset_table['Stim_duration'].mode()[0]
    
    stim_end=stim_start+stim_duration
    return stim_start,stim_end,stim_duration


def has_fixed_dt(t): #from AllenSDK
    """Check that all time intervals are identical."""
    dt = np.diff(t)
    return np.allclose(dt, np.ones_like(dt) * dt[0])


def calculate_dadt(a_Stim, t, filter=None): 
    """Low-pass filters (if requested) and differentiates voltage by time.

    Parameters
    ----------
    a : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default None)

    Returns
    -------
    dvdt : numpy array of time-derivative of voltage (V/s = mV/ms)
    """

    if has_fixed_dt(t) and filter:
        delta_t = t[1] - t[0]
        sample_freq = 1. / delta_t
        filt_coeff = (filter * 1e3) / (sample_freq / 2.) # filter kHz -> Hz, then get fraction of Nyquist frequency
        if filt_coeff < 0 or filt_coeff >= 1:
            raise ValueError("bessel coeff ({:f}) is outside of valid range [0,1); cannot filter sampling frequency {:.1f} kHz with cutoff frequency {:.1f} kHz.".format(filt_coeff, sample_freq / 1e3, filter))
        b, a = signal.bessel(4, filt_coeff, "low")
        a_filt = signal.filtfilt(b, a, a_Stim, axis=0)
        da = np.diff(a_filt)
    else:
        da = np.diff(a_Stim)

    dt = np.diff(t)
    dadt = 1e-3 * da / dt # in pA/s = pA/ms

    # Remove nan values (in case any dt values == 0)
    dadt = dadt[~np.isnan(dadt)]

    return dadt

def get_smooth_deriv_stim(stim_trace_table,do_plot=False):
    '''
    

    Parameters
    ----------
    membrane_potential_table : DataFrame
        First column represent time in s
        Second column represents stimulus trace in pA
        
    do_plot : Boolean, optional
        If true, print the plot

    Returns
    -------
    derivative_table : DataFrame
        First column represents time in s
        Second column represents membrane potential derivative in pA/ms.

    '''
    
    
    derivative_trace=pd.Series(calculate_dadt(a_Stim=stim_trace_table.iloc[:,2],
                                    t=stim_trace_table.iloc[:,0],
                                    filter=None))

    derivative_table=pd.DataFrame(np.column_stack((stim_trace_table.iloc[1:,0],derivative_trace)),columns=['Time_s',"Filtered_Stimulus_trace_derivative_pA/ms"])
    
    if do_plot:
        myplot=ggplot(derivative_table,aes(x=derivative_table.iloc[:,0],y=derivative_table.loc[:,'Filtered_Stimulus_trace_derivative_pA/ms']))+geom_line(color='red')
        
#        myplot+=geom_abline(intercept=20,slope=0)
        myplot
        # myplot=ggplot(stim_trace_table,aes(x=stim_trace_table.iloc[:,0],y=stim_trace_table.iloc[:,1]))+geom_line(color='blue')
        print(myplot)
    stim_trace_table=pd.merge(stim_trace_table,derivative_table,on='Time_s',how='outer')
    return stim_trace_table

    
#%% Extract F/I curve
    
def extract_stim_freq_Lantyer(cell_id,per_time=False,first_x_ms=0,per_nth_spike=False,first_nth_spike=0):
    '''
    Function to extract for each specified specimen_id and the corresponding stimulus the frequency of the response
    Frequency is defined as the number of spikes divided by the time between the stimulus start and the time of the specified index
    Parameters
    ----------
    specimen_id : int

    
    Returns
    -------
    f_I_table : DataFrame
        DataFrame with a column "specimen_id"(factor),the sweep number (int),the stimulus amplitude in pA(float),and the computed frequency of the response (float).
    '''
    
    f_I_table = pd.DataFrame(columns=['Cell_id', 'Sweep_number', 'Stim_amp_pA', 'Frequency_Hz'])
    cell_table=import_spike_time_table(str(cell_id))
    
    number_of_index = cell_table.shape[0]
    
    
    for current_index in range(number_of_index):
        
        stim_start_time=cell_table.loc[current_index,'Stim_start_s']
        current_spike_time_array=cell_table.loc[current_index,"Spike_times_s"]
        
        if len(current_spike_time_array) <1:
            freq = 0

        else :
            if per_nth_spike==True:
                reshaped_spike_times=current_spike_time_array[:first_nth_spike]
    
                t_last_spike = reshaped_spike_times[-1]
                freq=len(reshaped_spike_times)/((t_last_spike - stim_start_time))

            elif per_time==True:
                end_time=stim_start_time+(first_x_ms*1e-3)

                reshaped_spike_times=current_spike_time_array[current_spike_time_array <= end_time ]
                

                if len(reshaped_spike_times) !=0:
                    
                    freq=len(reshaped_spike_times)/(first_x_ms*1e-3)
                else:
                    freq=0
        new_line = pd.Series([str(cell_id),
                              int(cell_table.loc[current_index,"Sweep_id"]),
                              cell_table.loc[current_index,'Stim_amp_pA'],
                              freq],
                             index=['Cell_id',
                                    'Sweep_number',
                                    'Stim_amp_pA',
                                    'Frequency_Hz'])
        f_I_table = f_I_table.append(new_line, ignore_index=True)
    
    f_I_table = f_I_table.sort_values(by=["Cell_id", 'Stim_amp_pA'])
    f_I_table['Cell_id'] = pd.Categorical(f_I_table['Cell_id'])
    f_I_table['Sweep_number']=np.int64(f_I_table['Sweep_number'])
    f_I_table['Frequency_Hz']=np.float64(f_I_table['Frequency_Hz'])
    
    return f_I_table
#%%

def fit_specimen_fi_slope(stim_amps, avg_rates):
    """
    Fit the rate and stimulus amplitude to a line and return the slope of the fit.

    Parameters
    ----------
    stim_amps: array of sweeps amplitude in mA
    avg_rates: array of sweeps avergae firing rate in Hz
    Returns
    -------
    m: f-I curve slope for the specimen
    c:f-I curve intercept for the specimen

    """

    x = stim_amps
    y = avg_rates
    
    
    A = np.vstack([x, np.ones_like(x)]).T
   
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    return m, c
#%%
def normalized_root_mean_squared_error(true, pred,pred_extended):
    #Normalization by the interquartile range
    squared_error = np.square((true - pred))
    sum_squared_error = np.sum(squared_error)
    rmse = np.sqrt(sum_squared_error / true.size)
    Q1=np.percentile(pred_extended,25)
    Q3=np.percentile(pred_extended,75)
    #print('Q1=',Q1,", Q3=",Q3)
    nrmse_loss = rmse/(Q3-Q1)
    return nrmse_loss
#%%
def single_sigmoid_to_minimize(params,x_data,data):
    single_sigmoid_amplitude=params['single_sigmoid_amplitude']
    single_sigmoid_center=params['single_sigmoid_center']
    single_sigmoid_sigma=params['single_sigmoid_sigma']

    model=single_sigmoid_amplitude*(1-(1/(1+np.exp((x_data-single_sigmoid_center)/single_sigmoid_sigma))))

    return model-data

def sigmoid_function(x,amplitude,center,sigma):
     y=amplitude*(1-(1/(1+np.exp((x-center)/sigma))))
     return y   
 
def sigmoid_heaviside_to_minimize(params,x_data,data):
    sigmoid_amplitude=params['sigmoid_amplitude']
    sigmoid_center=params["sigmoid_center"]
    sigmoid_sigma=params["sigmoid_sigma"]
    heaviside_step=params["heaviside_step"]
   # heaviside_amplitude=params["heaviside_amplitude"]
    
   

    heaviside=Heaviside_function(x_data,heaviside_step)
    
    model=(sigmoid_amplitude*(1-(1/(1+np.exp((x_data-sigmoid_center)/sigmoid_sigma)))))*heaviside
    return model
def Heaviside_function(x, mid):
    """Heaviside step function."""
    
    if mid<=min(x):
        o=np.ones(x.size)
        return o
    elif mid>=max(x):
        o=np.zeros(x.size)
        return o
    else:
        o = np.zeros(x.size)
        
        imid = max(np.where(x < mid)[0])
        
        o[imid:] = 1
        return o
    
def sigmoid_heaviside(x,sigmoid_amplitude,sigmoid_center,sigmoid_sigma,heaviside_step):
    heaviside=Heaviside_function(x,heaviside_step)
    y=(sigmoid_amplitude*(1-(1/(1+np.exp((x-sigmoid_center)/sigmoid_sigma)))))*heaviside
    return y
 
def heaviside_fit_sigmoid_Lantyer (cell_id,per_time=False,first_x_ms=0,per_nth_spike=False,first_nth_spike=0,do_plot=False):
    try:

         
         # extract f_I table for the specimen and use only the "coarse" annotated sweeps
         f_I_table=extract_stim_freq_Lantyer(cell_id,per_time=per_time,first_x_ms=first_x_ms,per_nth_spike=per_nth_spike,first_nth_spike=first_nth_spike)
         
         
         x_data=f_I_table.loc[:,'Stim_amp_pA']
         y_data=f_I_table.loc[:,"Frequency_Hz"]
         
         #get initial estimate of parameters for single sigmoid fit
         without_zero_index=next(x for x, val in enumerate(y_data) if val >0 )

         
         median_firing_rate_index=next(x for x, val in enumerate(y_data) if val >= np.median(y_data.iloc[without_zero_index:]))
         #Get the stimulus amplitude correspondingto the median non-zero firing rate
         x0=x_data.iloc[median_firing_rate_index]
         #Get the slope from the linear fit of the firing rate
         slope,intercept=fit_specimen_fi_slope(x_data,y_data)
         
         first_non_zero_x=x_data.iloc[without_zero_index]
         new_x_data=pd.Series(np.arange(min(x_data),max(x_data),1))
         first_non_zero_extended_x_index=next(x for x, val in enumerate(new_x_data) if val >=first_non_zero_x )
         best_single_amplitude=np.nan
         best_single_center=np.nan
         best_single_sigma=np.nan
         best_compo_QNRMSE=None
         best_sigmoid_amplitude=np.nan
         best_sigmoid_center=np.nan
         best_sigmoid_sigma=np.nan
         
         best_heaviside_step=np.nan
         best_sigmoid_amplitude=np.nan
         best_sigmoid_center=np.nan
         best_sigmoid_sigma=np.nan

         best_single_QNRMSE=None
         fit='Rejected'
         typeII_tested=False
         
         ##First, try to fit a single sigmoid
         params_single_sigmoid=Parameters()
         params_single_sigmoid.add('single_sigmoid_amplitude',value=max(y_data),min=0)
         params_single_sigmoid.add('single_sigmoid_center',value=x0)
         params_single_sigmoid.add('single_sigmoid_sigma',value=500,min=0.1)
         params_single_sigmoid['single_sigmoid_amplitude'].set(brute_step=20)
         params_single_sigmoid['single_sigmoid_center'].set(brute_step=21)
         params_single_sigmoid['single_sigmoid_sigma'].set(brute_step=31)

         single_sigmoid_fitter=Minimizer(single_sigmoid_to_minimize,params_single_sigmoid,fcn_args=(x_data,y_data))

         single_result_brute=single_sigmoid_fitter.minimize(method='brute',Ns=10,keep=10)

         

         #plot_results_brute(single_result_brute,best_vals=True,varlabels=None)
         for current_single_result in single_result_brute.candidates:

             current_single_sigmoid_amplitude=current_single_result.params["single_sigmoid_amplitude"].value
             current_single_sigmoid_center=current_single_result.params["single_sigmoid_center"].value
             current_single_sigmoid_sigma=current_single_result.params["single_sigmoid_sigma"].value

             single_sigmoid_mod=StepModel(form='logistic',prefix='single_sigmoid_')
             single_sigmoid_mod_params=single_sigmoid_mod.make_params()
             single_sigmoid_mod_params['single_sigmoid_amplitude'].set(value=current_single_sigmoid_amplitude)
             single_sigmoid_mod_params['single_sigmoid_center'].set(value=current_single_sigmoid_center)
             single_sigmoid_mod_params['single_sigmoid_sigma'].set(value=current_single_sigmoid_sigma)
             
             single_sigmoid_out=single_sigmoid_mod.fit(y_data,single_sigmoid_mod_params,x=x_data)
             current_best_single_sigmoid_amplitude=single_sigmoid_out.best_values['single_sigmoid_amplitude']
             current_best_single_sigmoid_center=single_sigmoid_out.best_values['single_sigmoid_center']
             current_best_single_sigmoid_sigma=single_sigmoid_out.best_values['single_sigmoid_sigma']
             true=y_data.iloc[without_zero_index:]
             pred=pd.Series(sigmoid_function(x_data.iloc[without_zero_index:],current_best_single_sigmoid_amplitude,
                                                                                    current_best_single_sigmoid_center,
                                                                                    current_best_single_sigmoid_sigma))
             pred_extended=pd.Series(sigmoid_function(new_x_data.loc[first_non_zero_extended_x_index:],current_best_single_sigmoid_amplitude,
                                                                                    current_best_single_sigmoid_center,
                                                                                    current_best_single_sigmoid_sigma))

             if best_single_QNRMSE==None or best_single_QNRMSE>normalized_root_mean_squared_error(true,pred,pred_extended):
                 
                 best_single_amplitude=current_best_single_sigmoid_amplitude
                 best_single_center=current_best_single_sigmoid_center
                 best_single_sigma=current_best_single_sigmoid_sigma
                 true=y_data.iloc[without_zero_index:]
                 pred=pd.Series(sigmoid_function(x_data.iloc[without_zero_index:],current_best_single_sigmoid_amplitude,
                                                                                        current_best_single_sigmoid_center,
                                                                                        current_best_single_sigmoid_sigma))
                 pred_extended=pd.Series(sigmoid_function(new_x_data.loc[first_non_zero_extended_x_index:],current_best_single_sigmoid_amplitude,
                                                                                        current_best_single_sigmoid_center,
                                                                                        current_best_single_sigmoid_sigma))

                 best_single_QNRMSE=normalized_root_mean_squared_error(true,pred,pred_extended)

        
         single_sigmoid_y_data=pd.Series(sigmoid_function(new_x_data,best_single_amplitude,
                                                                                best_single_center,
                                                                                best_single_sigma))



         if best_single_QNRMSE<0.5:
             fit='TypeI'
             

         ##Define condition to test double sigmoid fit
         if best_single_QNRMSE<1e-3 or best_single_QNRMSE>0.5 or best_single_sigma<1:

             params=Parameters()
             params.add('sigmoid_amplitude',value=max(y_data),min=0)
             params.add('sigmoid_center',value=x0)
             params.add("sigmoid_sigma",value=104,min=40)
            
             params.add('heaviside_step',value=first_non_zero_x)

             
             params['sigmoid_amplitude'].set(brute_step=80)
             params["sigmoid_center"].set(brute_step=30)
             params["sigmoid_sigma"].set(brute_step=20)
            
             params['heaviside_step'].set(brute_step=5)


             fitter=Minimizer(sigmoid_heaviside_to_minimize,params,fcn_args=(x_data,y_data))

             result_brute=fitter.minimize(method='brute',Ns=20,keep=20)
             

    
    
             for current_results in result_brute.candidates:
                  current_sigmoid_amplitude=current_results.params['sigmoid_amplitude'].value
                  current_sigmoid_center=current_results.params['sigmoid_center'].value
                  current_sigmoid_sigma=current_results.params['sigmoid_sigma'].value
                
                  current_heaviside_step=current_results.params['heaviside_step'].value
                 
                  composite_model=Model(Heaviside_function)*Model(sigmoid_function)
                  pars=composite_model.make_params(amplitude=current_sigmoid_amplitude,
                                                   center=current_sigmoid_center,
                                                   sigma=current_sigmoid_sigma,
                                                   mid=current_heaviside_step)
                 
    
                  compo_out=composite_model.fit(y_data,pars,x=x_data)
    
                  # Get parameters best estimations
                  
                  true=y_data.iloc[without_zero_index:]

                  pred=pd.Series(sigmoid_heaviside(x_data.iloc[without_zero_index:],compo_out.best_values['amplitude'],compo_out.best_values['center'],compo_out.best_values["sigma"],compo_out.best_values["mid"]))

                  pred_extended=pd.Series(sigmoid_heaviside(new_x_data.loc[first_non_zero_extended_x_index:],compo_out.best_values['amplitude'],compo_out.best_values['center'],compo_out.best_values["sigma"],compo_out.best_values["mid"]))


                  if best_compo_QNRMSE==None or best_compo_QNRMSE>normalized_root_mean_squared_error(true,pred,pred_extended):

                      true=y_data.iloc[without_zero_index:]
                      pred=pd.Series(sigmoid_heaviside(x_data.iloc[without_zero_index:],compo_out.best_values['amplitude'],compo_out.best_values['center'],compo_out.best_values["sigma"],compo_out.best_values["mid"]))
                      pred_extended=pd.Series(sigmoid_heaviside(new_x_data.loc[first_non_zero_extended_x_index:],compo_out.best_values['amplitude'],compo_out.best_values['center'],compo_out.best_values["sigma"],compo_out.best_values["mid"]))
                     
                      best_compo_QNRMSE=normalized_root_mean_squared_error(true,pred,pred_extended)
                      
                      best_sigmoid_amplitude=compo_out.best_values["amplitude"]
                      best_sigmoid_center=compo_out.best_values["center"]
                      best_sigmoid_sigma=compo_out.best_values["sigma"]
                      best_heaviside_step=compo_out.best_values['mid']
                      
                      

             computed_y_data=pd.Series(pd.Series(sigmoid_heaviside(new_x_data,best_sigmoid_amplitude,best_sigmoid_center,best_sigmoid_sigma,best_heaviside_step)))
            
             model_table=pd.DataFrame(np.column_stack((new_x_data,computed_y_data)),columns=["Stim_amp_pA","Frequency_Hz"])

             typeII_tested=True

             if 2*best_compo_QNRMSE<=best_single_QNRMSE and best_compo_QNRMSE<0.1 and best_compo_QNRMSE>1e-4:
                 fit= 'TypeII'
             elif best_single_QNRMSE<0.5 and best_single_sigma>1:
                 fit='TypeI'
             else:
                 fit='Rejected'
                 
        

         if best_compo_QNRMSE==None:
             best_compo_QNRMSE=np.nan

         

         if do_plot == True:
              single_sigmoid_table=pd.DataFrame(np.column_stack((new_x_data,single_sigmoid_y_data)),columns=["Stim_amp_pA","Frequency_Hz"])
              my_plot=ggplot(f_I_table,aes(x=f_I_table["Stim_amp_pA"],y=f_I_table["Frequency_Hz"]))+geom_point()
              
              if fit=='TypeII':
                   compo_line='solid'
                   single_line='dashed'
                   my_plot+=geom_line(model_table,aes(x=model_table["Stim_amp_pA"],y=model_table['Frequency_Hz']),color='red',linetype=compo_line)
              elif fit=='TypeI':
                   compo_line='dashed'
                   single_line='solid'

                   if typeII_tested==True:

                       my_plot+=geom_line(model_table,aes(x=model_table["Stim_amp_pA"],y=model_table['Frequency_Hz']),color='red',linetype=compo_line)

              else:
                  single_line="dashed"
              
              my_plot+=geom_line(single_sigmoid_table,aes(x=single_sigmoid_table["Stim_amp_pA"],y=single_sigmoid_table['Frequency_Hz']),color='blue',linetype=single_line)

              my_plot+=ggtitle(str('F/I curve fit, Cell:'+str(cell_id)))
              print(my_plot)
              

         
        
         return fit,best_single_QNRMSE,best_single_amplitude,best_single_center,best_single_sigma,best_compo_QNRMSE,best_heaviside_step,best_sigmoid_amplitude,best_sigmoid_center,best_sigmoid_sigma
         
    except(StopIteration):
         print("Stop Iteration")
         return("Failed",np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
         
    except (ValueError):
         print("stopped_valueError")
         
         return("Failed",np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
        
    except (RuntimeError):
         print("Can't fit sigmoid, least-square optimization failed")
        
         return('Failed',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
    except (TypeError):
         print("Stop Type Error")
         return('Failed',np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)

        


def compute_f_I_params_Lantyer(fit_table,per_time=False,first_x_ms=0,per_nth_spike=False,first_nth_spike=0,do_plot=False):
    mycol=["Cell_id","Fit","Gain","Threshold","Saturation"]
    f_I_params_table=pd.DataFrame(columns=mycol)

    for current_cell_id in fit_table.loc[:,"Cell_id"]:
        current_fit_value=fit_table[fit_table['Cell_id']==str(current_cell_id)].Fit.values[0]
        
        current_f_I_table=extract_stim_freq_Lantyer(current_cell_id,per_time=per_time,first_x_ms=first_x_ms,per_nth_spike=per_nth_spike,first_nth_spike=first_nth_spike)
        

        x_data=current_f_I_table.loc[:,'Stim_amp_pA']
        new_x_data=pd.Series(np.arange(min(x_data),max(x_data),0.1))
        if current_fit_value == 'TypeI':
            
            single_amplitude_value=fit_table[fit_table['Cell_id']==str(current_cell_id)].Best_single_amplitude.values[0]
            single_center_value=fit_table[fit_table['Cell_id']==str(current_cell_id)].Best_single_center.values[0]
            single_sigma_value=fit_table[fit_table['Cell_id']==str(current_cell_id)].Best_single_sigma.values[0]
            single_sigmoid_y_data=pd.Series(sigmoid_function(new_x_data,single_amplitude_value,single_center_value,single_sigma_value))
                                                                                   
            
            twentyfive_index=next(x for x, val in enumerate(single_sigmoid_y_data) if val >(0.25*max(single_sigmoid_y_data)))
            seventyfive_index=next(x for x, val in enumerate(single_sigmoid_y_data) if val >(0.75*max(single_sigmoid_y_data)))
            #fit linear line to linear sigmoid portion
            Gain,Intercept=fit_specimen_fi_slope(new_x_data.iloc[twentyfive_index:seventyfive_index],sigmoid_function(new_x_data.iloc[twentyfive_index:seventyfive_index],single_amplitude_value,single_center_value,single_sigma_value))
            Threshold=(0-Intercept)/Gain
            extended_f_I_table=pd.DataFrame(np.column_stack((new_x_data,single_sigmoid_y_data)),columns=["Stim_amp_pA","Frequency_Hz"])
            my_derivative=np.array(derivative(sigmoid_function,new_x_data,dx=1e-3,args=(single_amplitude_value,single_center_value,single_sigma_value)))
            end_slope=np.mean(my_derivative[-10:])
            Saturation=np.nan

            if end_slope <=0.005:
                Saturation=np.mean(single_sigmoid_y_data[-10:])
                
            
                
                
                
        elif current_fit_value == 'TypeII':
            
            best_heaviside_step=fit_table[fit_table['Cell_id']==str(current_cell_id)].Best_heaviside_step.values[0]
            best_sigmoid_amplitude=fit_table[fit_table['Cell_id']==str(current_cell_id)].Best_sigmoid_amplitude.values[0]
            best_sigmoid_center=fit_table[fit_table['Cell_id']==str(current_cell_id)].Best_sigmoid_center.values[0]
            best_sigmoid_sigma=fit_table[fit_table['Cell_id']==str(current_cell_id)].Best_sigmoid_sigma.values[0]
            
            
            new_y_data=pd.Series(sigmoid_heaviside(new_x_data,best_sigmoid_amplitude,best_sigmoid_center,best_sigmoid_sigma,best_heaviside_step))
            extended_f_I_table=pd.DataFrame(np.column_stack((new_x_data,new_y_data)),columns=["Stim_amp_pA","Frequency_Hz"])
           
            second_derivative=new_y_data.diff().diff()
            linear_portion_start_index=next(x for x, val in enumerate(second_derivative) if val <0)
            linear_portion_start_index+=1
            linear_portion_y_data=new_y_data.iloc[linear_portion_start_index:]
            linear_portion_x_data=new_x_data.iloc[linear_portion_start_index:]
            
            twentyfive_index=next(x for x, val in enumerate(linear_portion_y_data) if val >((min(linear_portion_y_data)+0.25*(max(linear_portion_y_data)-min(linear_portion_y_data)))))
            seventyfive_index=next(x for x, val in enumerate(linear_portion_y_data) if val >((min(linear_portion_y_data)+0.75*(max(linear_portion_y_data)-min(linear_portion_y_data)))))
            
            Gain,Intercept=fit_specimen_fi_slope(linear_portion_x_data.iloc[twentyfive_index:seventyfive_index],linear_portion_y_data.iloc[twentyfive_index:seventyfive_index])
            first_derivative=new_y_data.diff()
            Threshold=best_heaviside_step
            Saturation=np.nan
            if np.mean(first_derivative[-10:]) <=0.005:
                Saturation=np.mean(first_derivative[-10:])
            
            
        else:
            Gain,Threshold,Saturation=np.nan,np.nan,np.nan
            
        if do_plot==True and current_fit_value=='TypeI' or do_plot==True and current_fit_value=='TypeII':
            
            myplot=ggplot(current_f_I_table,aes(x=current_f_I_table["Stim_amp_pA"],y=current_f_I_table["Frequency_Hz"]))+geom_point()
            myplot+=geom_line(extended_f_I_table,aes(x=extended_f_I_table["Stim_amp_pA"],y=extended_f_I_table["Frequency_Hz"]),color='blue')
            myplot+=geom_abline(aes(intercept=Intercept,slope=Gain))
            Threshold_table=pd.DataFrame({'Stim_amp_pA':[Threshold],'Frequency_Hz':[0]})
            myplot+=geom_point(Threshold_table,aes(x=Threshold_table["Stim_amp_pA"],y=Threshold_table["Frequency_Hz"]),color='green')
            if Saturation!=np.nan:
                myplot+=geom_abline(aes(intercept=Saturation,slope=0))
            myplot+=ggtitle(str('F/I curve fit, Cell:'+str(current_cell_id)))
            print(myplot)
        new_line=pd.Series([str(current_cell_id),current_fit_value,Gain,Threshold,Saturation],
                       index=mycol)
        f_I_params_table=f_I_params_table.append(new_line,ignore_index=True)
        
        
    return(f_I_params_table)
        
     
    
#%%

def extract_inst_freq_table_Lantyer(cell_id,per_time=False,first_x_ms=0,per_spike_nb=False,first_nth_spikes=0):
    '''
    Compute the instananous frequency in each interspike interval per sweep for a cell

    Parameters
    ----------
    specimen_id : int
        specimencell id.
    species_sweep_stim_table : DataFrame
        Coming from create_species_sweeps_stim_table function.

    Returns
    -------
    inst_freq_table: DataFrame
        Table containing for a given cell for each sweep the stimulus amplitude and the instantanous frequency per interspike interval.

    '''
    
    
   
    cell_table=import_spike_time_table(str(cell_id))
    
    number_of_index = cell_table.shape[0]
    maximum_nb_interval =0
    
    for current_index in range(number_of_index):
        stim_start_time=cell_table.loc[current_index,'Stim_start_s']
        current_spike_time_array=cell_table.loc[current_index,"Spike_times_s"]
        
        if per_time==True:
            
            end_time=stim_start_time+(first_x_ms*1e-3)
            
            spike_times=current_spike_time_array[current_spike_time_array <= end_time ]
            if len(spike_times)>maximum_nb_interval:
                maximum_nb_interval=len(spike_times)
        elif per_spike_nb==True:
            spike_times=current_spike_time_array[:first_nth_spikes]
            if len(spike_times)>maximum_nb_interval:
                maximum_nb_interval=len(spike_times)
        else:
            spike_times=current_spike_time_array
            if len(spike_times)>maximum_nb_interval:
                maximum_nb_interval=len(spike_times)
       
    mycolumns=["Cell_id","Sweep","Stim_amplitude_pA"]+["Interval_"+str(i) for i in range(1,(maximum_nb_interval))]
    inst_freq_table=pd.DataFrame(index=np.arange(number_of_index),columns=mycolumns)
    for col in range(inst_freq_table.shape[1]):
        inst_freq_table.iloc[:,col]=np.nan

    for line in range(number_of_index):
        stim_start_time=cell_table.loc[line,'Stim_start_s']
        current_spike_time_array=cell_table.loc[line,"Spike_times_s"]
        
        stim_amplitude=cell_table.loc[line,'Stim_amp_pA']
        if per_time==True:
            
            spike_times=current_spike_time_array[current_spike_time_array <= end_time ]
        elif per_spike_nb==True:
            spike_times=current_spike_time_array[:first_nth_spikes]
            
        else:
            spike_times=current_spike_time_array
            
        
        inst_freq_table.iloc[line,0]=str(cell_id)
        inst_freq_table.iloc[line,1]=int(cell_table.loc[line,"Sweep_id"])   
        inst_freq_table.iloc[line,2]=stim_amplitude
        # Put a minimum number of spikes to compute adaptation
        if len(spike_times) >2:
            for current_spike_time_index in range(1,len(spike_times)):
                current_inst_frequency=1/(spike_times[current_spike_time_index]-spike_times[current_spike_time_index-1])
                
                inst_freq_table.iloc[line,(current_spike_time_index+2)]=current_inst_frequency
        
            inst_freq_table.iloc[line,3:]/=inst_freq_table.iloc[line,3]
    inst_freq_table = inst_freq_table.sort_values(by=["Cell_id", 'Stim_amplitude_pA'])
    inst_freq_table['Cell_id']=pd.Categorical(inst_freq_table['Cell_id'])

    interval_freq_table=pd.DataFrame(columns=['Interval','Inst_frequency','Stimulus_amplitude'])
    isnull_table=inst_freq_table.isnull()
    for col in range(3,(inst_freq_table.shape[1])):
        for line in range(inst_freq_table.shape[0]):
            if isnull_table.iloc[line,col] == False:
                new_line=pd.Series([int(col-2),inst_freq_table.iloc[line,col],np.float64(inst_freq_table.iloc[line,2])],
                                   index=['Interval','Inst_frequency','Stimulus_amplitude'])
                interval_freq_table=interval_freq_table.append(new_line,ignore_index=True)
   
    specimen=pd.Series(np.array([inst_freq_table.iloc[0,0]]*interval_freq_table.shape[0]))
    interval_freq_table=pd.concat([specimen,interval_freq_table],axis=1)
    interval_freq_table.columns=["Cell_id",'Interval','Inst_frequency','Stimulus_amplitude']
    interval_freq_table['Cell_id']=pd.Categorical(interval_freq_table['Cell_id'])

    return interval_freq_table

def expo_to_minimize(params,x_data,data):
    A=params['amplitude']
    B=params['decay']
    C=params['steady_state']
    model=A*np.exp(-(x_data-1)/B)+C
    return model-data

def my_exponential_decay(x,A,B,C):
    '''
    Parameters
    ----------
    x : Array
        interspike interval index array.
    A: flt
        initial instantanous frequency .
    B : flt
        Adaptation index constant.
    C : flt
        intantaneous frequency limit.

    Returns
    -------
    y : array
        Modelled instantanous frequency.

    '''
    y=A*np.exp(-(x-1)/B)+C
    
    return y

def fit_exponential_decay_Lantyer(cell_id,per_time=False,first_x_ms=0,per_spike_nb=False,first_nth_spikes=0,do_plot=False):
    '''
    Parameters
    ----------
    interval_freq_table : DataFrame
        Comming from table_to_fit function.

    Returns
    -------
    my_plot : ggplot
        
    starting_freq : flt
        estimated initial instantanous frequency.
    adapt_cst : flt
        Adaptation index constant.
    limit_freq : flt
        intantaneous frequency limit.
    pcov_overall : 2-D array
        The estimated covariance of popt

    '''
    
    try:
        
        interval_freq_table=extract_inst_freq_table_Lantyer(cell_id,per_time=per_time,first_x_ms=first_x_ms,per_spike_nb=per_spike_nb,first_nth_spikes=first_nth_spikes)
        if interval_freq_table.shape[0]==0:
            print('Not enough spike to compute adaptation')
            my_plot=np.nan
            A=np.nan
            B=np.nan
            C=np.nan

            return A,B,C
        x_data=interval_freq_table.loc[:,"Interval"]

        median_table=interval_freq_table.groupby(by=["Interval"],dropna=True).median()
        median_table["Count_weights"]=pd.DataFrame(interval_freq_table.groupby(by=["Interval"],dropna=True).count()).iloc[:,-1]
        median_table["Interval"]=np.arange(1,(median_table.shape[0]+1))
        median_table["Interval"]=np.float64(median_table["Interval"])  
        

        try:
            lower_index=next(x for x, val in enumerate(median_table["Inst_frequency"]) if val<((min(median_table["Inst_frequency"])+median_table["Inst_frequency"][1])/2))
        except(StopIteration):
            lower_index=np.inf
        try:
        
            higher_index= next(x for x, val in enumerate(median_table["Inst_frequency"]) if val>((max(median_table["Inst_frequency"])+median_table["Inst_frequency"][1])/2))
        except(StopIteration):
            higher_index=np.inf
        #med_index=next(x for x, val in enumerate(median_table["inst_frequency"]) if val<0.5*max(median_table['inst_frequency']))

        if lower_index<higher_index:
            med_index=lower_index
            initial_amplitude=1
        else:
            med_index=higher_index
            initial_amplitude=-1
        initial_decay_value=np.mean(median_table["Interval"][med_index:med_index+1])

        

        decay_step=15
        while max(x_data)%decay_step==0:
            decay_step+=1
        params=Parameters()
        params.add('amplitude',value=initial_amplitude)
        params.add('decay',value=initial_decay_value,min=0.001)
        params.add('steady_state',value=0.1,min=0,expr="1-amplitude")
        
        params['amplitude'].set(brute_step=5)
        params['decay'].set(brute_step=decay_step)
        params['steady_state'].set(brute_step=0.1,min=0)
        
        
           
        #data=my_exponential_decay(x=median_table["interval"], A=0.5, B=2, C=0.1)
        fitter=Minimizer(expo_to_minimize,params,fcn_args=(median_table["Interval"],median_table["Inst_frequency"]))
        result_brute=fitter.minimize(method='brute',Ns=40,keep=40)


        
        
        
        
        best_chi=None
        
        for current_result in result_brute.candidates:
            current_amplitude=current_result.params['amplitude'].value
            current_decay=current_result.params['decay'].value
            current_steady_state=current_result.params['steady_state'].value

            
            exponential_model=ExponentialModel(prefix="expo_")
            expo_params=exponential_model.make_params()
            expo_params["expo_amplitude"].set(value=current_amplitude)
            expo_params["expo_decay"].set(value=current_decay,min=0.1)
            
            constant_model=ConstantModel(prefix="const_")
            expo_params.update(constant_model.make_params())
            expo_params["const_c"].set(value=current_steady_state,min=0,expr='1-expo_amplitude')
            
            full_expo_model=exponential_model+constant_model
            full_expo_out=full_expo_model.fit(median_table["Inst_frequency"],expo_params,x=median_table["Interval"],weights=median_table["Count_weights"])
            
            
            if best_chi==None:
                best_chi=full_expo_out.chisqr
                best_A=full_expo_out.best_values["expo_amplitude"]
                best_B=full_expo_out.best_values["expo_decay"]
                best_C=full_expo_out.best_values["const_c"]
            elif best_chi>full_expo_out.chisqr:
                best_chi=full_expo_out.chisqr
                best_A=full_expo_out.best_values["expo_amplitude"]
                best_B=full_expo_out.best_values["expo_decay"]
                best_C=full_expo_out.best_values["const_c"]
        
        A_norm=best_A/(best_A+best_C)
        C_norm=best_C/(best_A+best_C)
        interval_range=np.arange(1,max(median_table["Interval"])+1,1)

        simulation=my_exponential_decay(interval_range,best_A,best_B,best_C)
        norm_simulation=my_exponential_decay(interval_range,A_norm,best_B,C_norm)
        sim_table=pd.DataFrame(np.column_stack((interval_range,simulation)),columns=["Interval","Normalized_freq"])
        norm_sim_table=pd.DataFrame(np.column_stack((interval_range,norm_simulation)),columns=["Interval","Normalized_freq"])
        

        
        my_plot=np.nan
        if do_plot==True:
            #plot_results_brute(result_brute,best_vals=True,varlabels=None)
            my_plot=ggplot(interval_freq_table,aes(x='Interval',y='Inst_frequency'))+geom_point(aes(color=interval_freq_table.loc[:,"Stimulus_amplitude"]))
            
            my_plot=my_plot+geom_point(median_table,aes(x='Interval',y='Inst_frequency',size=median_table["Count_weights"]),shape='s',color='red')
            my_plot=my_plot+geom_line(sim_table,aes(x='Interval',y='Normalized_freq'),color='black')
            my_plot=my_plot+geom_line(norm_sim_table,aes(x='Interval',y='Normalized_freq'),color="green")
            my_plot+=ggtitle(str("Adaptation, Cell_id: "+str(cell_id)))
            print(my_plot)


        
        return best_A,best_B,best_C
    except (StopIteration):
        print("Stopped Iteration")
        my_plot=np.nan
        A=np.nan
        B=np.nan
        C=np.nan

        return A,B,C
    except (ValueError):
        print("stopped_valueError")
        my_plot=np.nan
        A=np.nan
        B=np.nan
        C=np.nan
        return A,B,C
    except(RuntimeError):
        print("Can't fit exponential, least-square optimization failed")
        my_plot=np.nan
        A=np.nan
        B=np.nan
        C=np.nan
        return A,B,C
    except(TypeError):
        print("Stopped TypeError")
        my_plot=np.nan
        A=np.nan
        B=np.nan
        C=np.nan
        return A,B,C

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    