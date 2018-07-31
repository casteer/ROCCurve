import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math


# Alter these rates, mean and standard deviations and look at how the system performs

# The signal and background rates 
background_rate = 1.0
signal_rate = 0.25 + background_rate # Should be in the presence of the background

# Acquisition time (assumed to be identical for both signal and background measurements)
acq_time = 90

# Calculate the mean and standard deviation
background_mean_counts = acq_time*background_rate
signal_mean_counts = acq_time*signal_rate

# - Assuming a Poisson distribution... 
background_std_counts = np.sqrt(background_mean_counts)
signal_std_counts = np.sqrt(signal_mean_counts)

# Calculate the total = signal+background rates
total_rate = (signal_rate+background_rate)
bins = np.linspace(background_mean_counts-5*background_std_counts,signal_mean_counts+5*signal_std_counts,1000)

# The analytical Gaussian distributions for signal and background 
signal = (signal_rate/total_rate)*mlab.normpdf(bins, signal_mean_counts, signal_std_counts)
background = (background_rate/total_rate)*mlab.normpdf(bins, background_mean_counts, background_std_counts)

# Initialize the true and false positives
roc_tp=[]
roc_fp=[]

# Loop over the detection threshold 
for threshold in bins[10:-5]:
    
    # Find the corresponding bin (or index) of signal and background arrays
    above_threshold = np.min(np.argwhere((bins>(threshold*np.ones_like(bins)))))
    
    # Calculate the true positive fraction as the area under the signal+background distribution above the threshold 
    roc_tp.append(2.0*np.trapz(signal[above_threshold:-1],bins[above_threshold:-1]))

    # Calculate the true positive fraction as the area under the background distribution that is above the threshold 
    roc_fp.append(2.0*np.trapz(background[above_threshold:-1],bins[above_threshold:-1]))
    


# Pick a threshold on the scale (called bins here)    
threshold = bins[500]

# Create a figure 
fig = plt.figure(num=None, figsize=(18, 6), dpi=100, facecolor='w', edgecolor='k')

# The right-hand plot of the ROC curve 
plt.subplot(133)
plt.title('ROC Curve')
plt.plot(roc_fp,roc_tp,'r-')

above_threshold = np.min(np.argwhere((bins>(threshold*np.ones_like(bins)))))
this_roc_tp = np.trapz(2.0*signal[above_threshold:-1],bins[above_threshold:-1])
this_roc_fp = np.trapz(2.0*background[above_threshold:-1],bins[above_threshold:-1])

# Plot a marker on the ROC curve corresponding to this threshold  
plt.plot(this_roc_fp,this_roc_tp,'ko',markersize=10)

# Create some axes labels and add a grid 
plt.ylabel("Fraction of True Positive Trials")
plt.xlabel("Fraction of False Alarms Trials")
plt.grid()
    




# The left-hand plot of the signal+background and background distributions with the shaded area as the true positives 
ax = plt.subplot(131)
plt.title('True Positives')

# Plot the distributions 
plt.plot(bins,signal,'r-',label='Signal+Background')
plt.plot(bins,background,'g-',label='Background')        
#... and the threshold as a dashed vertical line 
plt.plot([threshold,threshold], [0,0.025] ,'k--')

# Colour in the area corresponding to the true positives     
ax.fill_between(    bins, 
                    signal, 
                    0, where=(bins>threshold),
                    facecolor='orange', interpolate=True)

# Create some axes labels and add a grid and legend
plt.xlabel("Counts in Region of Interest")
plt.ylabel("Frequency")
plt.grid()
plt.legend()
    
    





# The left-hand plot of the signal+background and background distributions with the shaded area as the false positives 
ax = plt.subplot(132)
plt.title('False Positives')

# Plot the distributions 
plt.plot(bins,signal,'r-',label='Signal+Background')
plt.plot(bins,background,'g-',label='Background')        
plt.plot([threshold,threshold], [0,0.025] ,'k--')
    
# Colour in the area corresponding to the true positives     
ax.fill_between(    bins, 
                    background, 
                    0, where=(bins>threshold),
                    facecolor='teal', interpolate=True)

# Create some axes labels and add a grid and legend
plt.xlabel("Counts in Region of Interest")
plt.ylabel("Frequency")
plt.grid()
plt.legend()
 
    
plt.show()
