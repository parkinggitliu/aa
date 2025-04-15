import serdespy as sdp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#common parameters
samples_per_symbol = 64
data_rate = 106.125e9
UI =  2/data_rate 
t_symbol = 2/data_rate
t_sample = t_symbol/samples_per_symbol
#frequency vector (rad/s)

k = 14
fmax=1/t_sample
f = np.linspace(0,fmax,2**k+1)
w = f*2*np.pi
np.save("./data/f.npy",f)
np.save("./data/w.npy",w)

h = np.load("./data/h_thru.npy")
#%% Eye Diagram
hpulse = sp.signal.convolve(h, np.ones(np.array([samples_per_symbol,])))[:np.size(h)]
np.save("./data/hpulse.npy",hpulse)
#generate binary data
data = sdp.prqs10(1)[:10000]

#generate Baud-Rate sampled signal from data
signal_BR = sdp.pam4_input_BR(data)

#oversampled signal
signal_ideal = np.repeat(signal_BR, samples_per_symbol)

#eye diagram of signal after channel
signal_out = sp.signal.convolve(h,signal_ideal)
sdp.simple_eye(signal_out[1000*samples_per_symbol:], samples_per_symbol*3, 1000, t_sample, "{}Gbps 4-PAM Signal".format(data_rate/1e9), 100 )
                      
#%% save data for next homework assignment
np.save("./data/signal.npy",signal_out)

plt.show()
