import serdespy as sdp
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import scipy as sp
from scipy.fft import fft, ifft, fftfreq

import numpy as np
def generate_prbs7(length):
    """Generates a PRBS7 sequence of the specified length."""
    if length <= 0:
        return np.array([])

    lfsr = np.array([1, 0, 0, 0, 0, 0, 1], dtype=int)  # Initial state (can be non-zero)
    sequence = np.zeros(length, dtype=int)

    for i in range(length):
        feedback = lfsr[6] ^ lfsr[5]  # XOR taps for PRBS7 (x^7 + x^6 + 1)
        sequence[i] = lfsr[0]
        lfsr = np.roll(lfsr, 1)
        lfsr[0] = feedback

    return sequence

def map_prbs7_to_pam4(prbs_sequence, mapping='gray'):
    """Maps a binary PRBS7 sequence to PAM4 symbols.

    Args:
        prbs_sequence (np.ndarray): The binary PRBS7 sequence.
        mapping (str, optional): The binary-to-PAM4 symbol mapping.
                                  'gray' (default), 'natural', or a custom dict.

    Returns:
        np.ndarray: The PAM4 symbol sequence.
    """
    if len(prbs_sequence) % 2 != 0:
        raise ValueError("PRBS sequence length must be even for PAM4 mapping.")

    num_symbols = len(prbs_sequence) // 2
    pam4_symbols = np.zeros(num_symbols)

    if mapping == 'gray':
        mapping_dict = {'00': -3, '01': -1, '11': 1, '10': 3}
    elif mapping == 'natural':
        mapping_dict = {'00': -3, '01': -1, '10': 1, '11': 3}
    elif isinstance(mapping, dict):
        mapping_dict = mapping
    else:
        raise ValueError("Invalid PAM4 mapping specified.")

    for i in range(num_symbols):
        binary_pair = str(prbs_sequence[2 * i]) + str(prbs_sequence[2 * i + 1])
        if binary_pair in mapping_dict:
            pam4_symbols[i] = mapping_dict[binary_pair]
        else:
            raise ValueError(f"Invalid binary pair '{binary_pair}' for PAM4 mapping.")

    return pam4_symbols

def pam4_lms_ffe_dfe(received_signal, desired_levels, ffe_taps, dfe_taps, step_size, decision_delay):
    """
    Performs LMS adaptation for FFE and DFE for PAM4 modulation.

    Args:
        received_signal: The received signal (numpy array).
        desired_levels: The desired signal levels for PAM4 ([-3, -1, 1, 3]).
        ffe_taps: Initial FFE filter taps (numpy array).
        dfe_taps: Initial DFE filter taps (numpy array).
        step_size: The LMS step size.
        decision_delay: The delay between receiving and making a decision.

    Returns:
        equalized_signal: The equalized signal.
        ffe_taps: The adapted FFE filter taps.
        dfe_taps: The adapted DFE filter taps.
        decisions: the decisions made by the slicer.
    """

    num_samples = len(received_signal)
    ffe_length = len(ffe_taps)
    dfe_length = len(dfe_taps)
    equalized_signal = np.zeros(num_samples)
    decisions = np.zeros(num_samples)

    for n in range(num_samples):
        # FFE output
        ffe_input = received_signal[max(0, n - ffe_length + 1):n + 1]
        ffe_input = np.pad(ffe_input, (max(0, ffe_length - n - 1), 0), 'constant')
        ffe_output = np.dot(ffe_input, ffe_taps)

        # DFE output
        dfe_input = decisions[max(0, n - dfe_length + 1 - decision_delay):n - decision_delay]
        dfe_input = np.pad(dfe_input, (max(0, dfe_length - (n - decision_delay)), 0), 'constant')
        print(f"n: {n}, dfe_input.shape: {dfe_input.shape}, dfe_taps.shape: {dfe_taps.shape}") #Debug line.
        dfe_output = np.dot(dfe_input, dfe_taps)

        # Equalized signal
        equalized_signal[n] = ffe_output - dfe_output

        # PAM4 Decision (slicer)
        distances = np.abs(equalized_signal[n] - np.array(desired_levels))
        decision_index = np.argmin(distances)
        decision = desired_levels[decision_index]
        decisions[n] = decision

        # Error calculation
        error = decision - equalized_signal[n] # the decision is the desired level now.

        # Tap update (LMS)
        ffe_taps += step_size * error * ffe_input
        dfe_taps += step_size * error * dfe_input

    return equalized_signal, ffe_taps, dfe_taps, decisions
def get_every_kth_element(data_sequence, k):
  """
  Selects every k-th element from a sequence starting from index 0.

  Args:
    data_sequence: The list, tuple, string, or NumPy array to select from.
    k: The step size (select element 0, k, 2k, ...). Must be > 0.

  Returns:
    A new sequence of the same type containing the selected elements.
    Returns an empty sequence if k is invalid or the input sequence is empty.
  """
  if k <= 0:
    print("Error: Step k must be a positive integer.")
    # Return an empty sequence of the same type if possible
    try:
      return type(data_sequence)()
    except TypeError:
      return None # Or raise ValueError("Step k must be positive")

  # Python's slicing [start:stop:step]
  # ::k means start from the beginning, go to the end, with a step of k
  return data_sequence[::k]

def plot_eye_diagram(signal, samples_per_symbol, offset=0, traces=10, title='Eye Diagram'):
    """
    Generates and plots an eye diagram for a given signal.

    Args:
        signal (numpy.ndarray): The input signal.
        samples_per_symbol (int): Number of samples per symbol.
        offset (int, optional): Starting index for plotting. Defaults to 0.
        traces (int, optional): Number of traces to plot. Defaults to 10.
    """
    plt.figure(dpi = 100)
    for i in range(traces):
        start = offset + i * samples_per_symbol
        end = start + 2 * samples_per_symbol
        if end <= len(signal):
            plt.plot(np.linspace(-1, 1, 2 * samples_per_symbol), signal[start:end], 'b-')
    plt.xlabel('Time (normalized)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True)
    plt.show()
    
#common section
samples_per_symbol = 32
data_rate = 212.5e9
nyquist_f = data_rate/4
t_symbol = 2/data_rate
t_sample = t_symbol/samples_per_symbol
fmax = 0.5/t_sample
UI = t_symbol

k = 14
f = np.linspace(0,fmax,2**k+1)
w = f*2*np.pi



#set poles and zeroes for peaking at nyquist freq
#high peaking because channel is high insertion loss
z = 24e10
#z=2.69e10
#z=1.3e10/4
#z=10e10
p = 3.4e11
k = p**2/z

#calculate Frequency response of CTLE at given frequencies
w, H_ctle = sp.signal.freqs([k/p**2, k*z/p**2], [1/p**2, 2/p, 1], w)

#bode plot of CTLE transfer function
plt.figure(dpi=100)
plt.semilogx(1*f,20*np.log10(abs(H_ctle)), color = "red", label = 'CTLE')
plt.title("CTLE Frequency Response")
plt.grid()
plt.axvline(x=25,color = 'grey', label = "Nyquist Frequency")
plt.axvline(x=z/(2*np.pi)*1,color = 'green', label = "Zero Location")
plt.axvline(x=p/(2*np.pi)*1,color = 'blue', label = "Pole Location")
plt.legend()

h_ctle, t_ctle = sdp.freq2impulse(H_ctle,f)
h_ctle = h_ctle[0:200]
plt.figure(dpi=100)
plt.plot(h_ctle)
plt.title("CTLE Impulse Response")

#%% Eye diagram of signal with and without CTLE

ntrace =250*4 
N_symbols=2100
offset = int(samples_per_symbol/2)
prbs7_length = 2*N_symbols  # Must be an even number for PAM4 mapping
prbs7_sequence = generate_prbs7(prbs7_length)
print("Generated PRBS7 Sequence (first 20 bits):", prbs7_sequence[:20])

pam4_symbols = map_prbs7_to_pam4(prbs7_sequence, mapping='gray')
print("PAM4 Symbols (first 10 symbols):", pam4_symbols[:10])
#sdp.simple_eye(aa, samples_per_symbol*3, ntrace,t_sample, "{}Gbps 4-PAM Signal".format(data_rate/1e9))

plot_eye_diagram(aa, samples_per_symbol, offset, ntrace, "{}Gbps 4-PAM Signal".format(data_rate/1e9))

signal_ctle = sp.signal.convolve(signal,h_ctle)
signal_ctle = signal_ctle*10 *2.5
aa2 = signal_ctle[100*samples_per_symbol:]
#sdp.simple_eye(aa2, samples_per_symbol*3, ntrace, t_sample, "{}Gbps 4-PAM Signal with CTLE".format(data_rate/1e9))
plot_eye_diagram(aa2, samples_per_symbol, offset, ntrace, "{}Gbps 4-PAM Signal with CTLE".format(data_rate/1e9))


FFE_pre = 4
FFE_taps = 7
FFE_post = FFE_taps - FFE_pre - 1
DFE_taps = 1

#sdp.channel_coefficients(h_pulse[:t.size],t,40,2,4)

#h = sdp.channel_coefficients(h_pulse_ctle[:t.size],t,40,2,4)
#%%
#h /= h.max()
#print('h: ',h)

#channel_main = h.argmax()

#main_cursor = h[channel_main]

main_cursor = 1
#generate binary data
#data = sdp.prqs10(1)[:10000]

#voltage_levels = np.array([-3, -1, 1, 3]*0.25)
#voltage_levels = np.array([-0.12, -0.04, 0.04, 0.12])

#generate Baud-Rate sampled signal from data
#signal_BR = sdp.pam4_input_BR(data)
cdr_pos=16+8
for cdr_pos in range (0, 9, 1):
   
    aa2_BR = get_every_kth_element(aa2[cdr_pos:], samples_per_symbol)
    titles0= str(data_rate/1e9) + "Gbps 4-PAM BR Signal with CTLE with CDR position" +str(cdr_pos)
    plot_eye_diagram(aa2_BR, 1, 0, ntrace, titles0)

    #signal_rx = sp.signal.fftconvolve(h, signal_BR)[:len(signal_BR)]
    signal_rx_cropped = aa2_BR
    #reference_signal = signal[:1000]
    w_ffe_init = np.zeros(FFE_taps)
    w_dfe_init = np.zeros(DFE_taps)

    voltage_levels = np.array([-3, -1, 1, 3])

    #w_ffe, w_dfe, v_combined_ffe, v_combined_dfe, z_combined, e_combined = \
    #    sdp.lms_equalizer(signal_rx_cropped, 0.001, len(signal_rx_cropped), w_ffe_init, FFE_pre, w_dfe_init,  voltage_levels, reference=reference_signal)
    w_ffe, w_dfe, v_combined_ffe, v_combined_dfe, z_combined, e_combined = sdp.lms_equalizer(signal_rx_cropped, 0.001, len(signal_rx_cropped), w_ffe_init, FFE_pre, w_dfe_init,  voltage_levels)
    #%%
  
    nyquist_f = data_rate/4
    RX = sdp.Receiver(signal_ctle, samples_per_symbol, nyquist_f, voltage_levels, main_cursor=main_cursor)
    #sdp.simple_eye(RX.signal, samples_per_symbol*3, 800, TX.UI/TX.samples_per_symbol, "Eye Diagram with CTLE")

    RX.FFE(w_ffe, FFE_pre)
    #plot_eye_diagram(RX.signal[int(100*samples_per_symbol):], samples_per_symbol, 0, ntrace, "{}Gbps 4-PAM BR Signal with CTLE and FFE".format(data_rate/1e9))
    #sdp.simple_eye(RX.signal[int(100*samples_per_symbol):], samples_per_symbol*3, ntrace, t_sample, "Eye Diagram with CTLE and FFE", 100)

    RX.pam4_DFE(w_dfe)
    titles= str(data_rate/1e9) + "Gbps 4-PAM BR Signal with CTLE+FFE+DFE with CDR position" +str(cdr_pos)
    #sdp.simple_eye(RX.signal[int(100*samples_per_symbol):], samples_per_symbol*3, ntrace, t_sample, "Eye Diagram with CTLE, FFE, and DFE", 100)
    plot_eye_diagram(RX.signal[int(100*samples_per_symbol):], samples_per_symbol, 0, ntrace, titles)

plt.show()