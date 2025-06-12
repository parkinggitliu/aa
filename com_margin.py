import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv

class COMCalculator:
    """
    Calculates Channel Operating Margin (COM) for a PCIe Gen5 link.

    This class implements the COM methodology, which is a standardized way to
    assess channel quality by calculating a signal-to-noise ratio. The "noise"
    term includes residual ISI, jitter, and device noise.
    """

    def __init__(self, s_params_df, data_rate_gbps=32.0):
        """
        Initializes the COM calculator with channel and spec parameters.

        Args:
            s_params_df (np.ndarray): A NumPy array with two columns:
                                      [Frequency (GHz), Insertion Loss (dB)].
            data_rate_gbps (float): The data rate in Gbps (or GT/s).
        """
        # --- PCIe Gen5 Specification Constants ---
        self.data_rate = data_rate_gbps * 1e9
        self.unit_interval = 1 / self.data_rate
        self.nyquist_freq = self.data_rate / 2
        self.target_ber = 1e-12  # Standard target BER for COM

        # --- Device Parameters (from PCIe spec) ---
        self.tx_ffe_taps = {'pre': 1, 'post': 2}  # Number of pre/post-cursor taps
        self.rx_ctle_config = {'dc_gain_db': -10, 'ac_gain_db': 10, 'f_pole_ghz': 16.0}
        self.rx_dfe_taps = 10 # Number of DFE taps to cancel post-cursor ISI
        
        # --- Noise and Jitter Parameters (from PCIe spec) ---
        self.tx_noise_rms = 1.5e-3 # V
        self.rx_noise_rms = 3.0e-3 # V
        self.random_jitter_rms = 0.15e-12 # s
        self.deterministic_jitter_pkpk = 0.25e-12 # s
        self.eta_0 = 8.8e-10 # V^2/GHz, spectral density of noise

        # --- Channel Data ---
        self.freq_ghz = s_params_df[:, 0]
        self.insertion_loss_db = s_params_df[:, 1]
        
        # --- Internal Simulation Parameters ---
        self.samples_per_ui = 64
        self.time_span_ui = 20 # How many UIs to simulate for the pulse response
        self.num_points = self.samples_per_ui * self.time_span_ui
        self.time_step = self.unit_interval / self.samples_per_ui
        self.freq_step = 1 / (self.num_points * self.time_step)
        
        self.sim_freq_vector_ghz = np.arange(0, self.num_points // 2) * self.freq_step / 1e9

    def _get_channel_transfer_function(self):
        """Interpolates S-parameters to the simulation frequency steps."""
        # Convert dB to linear magnitude
        linear_mag = 10**(self.insertion_loss_db / 20.0)
        # Interpolate to our simulation's frequency vector
        h_channel = np.interp(self.sim_freq_vector_ghz, self.freq_ghz, linear_mag, left=1.0, right=0.0)
        return h_channel

    def _get_tx_ffe_response(self, coefficients):
        """Calculates the frequency response of the Tx FFE FIR filter."""
        h_ffe = np.zeros_like(self.sim_freq_vector_ghz, dtype=np.complex128)
        w = 2 * np.pi * self.sim_freq_vector_ghz * 1e9
        
        # Main cursor
        h_ffe += coefficients['main']
        # Pre-cursors
        for i in range(self.tx_ffe_taps['pre']):
            h_ffe += coefficients['pre'][i] * np.exp(1j * w * (i + 1) * self.unit_interval)
        # Post-cursors
        for i in range(self.tx_ffe_taps['post']):
            h_ffe += coefficients['post'][i] * np.exp(-1j * w * (i + 1) * self.unit_interval)
            
        return h_ffe

    def _get_rx_ctle_response(self):
        """Calculates the frequency response of the Rx CTLE."""
        g_dc = 10**(self.rx_ctle_config['dc_gain_db'] / 20)
        g_ac = 10**(self.rx_ctle_config['ac_gain_db'] / 20)
        f_z = self.nyquist_freq * 0.5 # Zero frequency often placed around 0.5 * Nyquist
        f_p1 = self.rx_ctle_config['f_pole_ghz'] * 1e9
        
        s = 1j * 2 * np.pi * self.sim_freq_vector_ghz * 1e9
        
        h_ctle = g_dc * (1 + s / (2 * np.pi * f_z)) / (1 + s / (2 * np.pi * f_p1))
        return h_ctle
        
    def calculate_pulse_response(self, tx_ffe_coeffs):
        """
        Calculates the single-bit pulse response of the full channel + EQ.
        """
        # 1. Get transfer function for each component
        h_channel = self._get_channel_transfer_function()
        h_ffe = self._get_tx_ffe_response(tx_ffe_coeffs)
        h_ctle = self._get_rx_ctle_response()
        
        # 2. Total system transfer function (Tx + Channel + Rx)
        h_system = h_ffe * h_channel * h_ctle
        
        # 3. An ideal Tx pulse is a single dirac delta in time, which is flat in frequency
        # So we can use the system transfer function directly.
        
        # 4. Perform Inverse FFT to get time-domain pulse response
        # We use irfft for real-valued output
        pulse_response = np.fft.irfft(h_system, n=self.num_points)
        
        # 5. Create the time vector for plotting
        time_vector = np.arange(self.num_points) * self.time_step
        
        return time_vector, pulse_response

    def run_com_calculation(self, pulse_response):
        """
        Performs the final COM calculation from the pulse response.
        """
        # 1. Find the peak of the pulse response (main cursor)
        center_index = self.time_span_ui // 2 * self.samples_per_ui
        search_range = slice(center_index - self.samples_per_ui, center_index + self.samples_per_ui)
        peak_index = np.argmax(np.abs(pulse_response[search_range])) + (center_index - self.samples_per_ui)
        p_max = pulse_response[peak_index]

        # 2. Extract ISI cursors by sampling at UI intervals
        isi_indices = []
        for i in range(-self.time_span_ui // 2, self.time_span_ui // 2):
             if i == 0: continue # Skip main cursor
             index = peak_index + i * self.samples_per_ui
             if 0 <= index < len(pulse_response):
                 isi_indices.append(index)
        
        isi_cursors = pulse_response[isi_indices]

        # 3. Simulate DFE to remove post-cursor ISI
        # The DFE removes the first N post-cursors
        num_post_cursors = (self.time_span_ui//2) - 1
        num_dfe_cancelled = min(self.rx_dfe_taps, num_post_cursors)
        
        # ISI after DFE is the remaining cursors
        residual_isi = np.concatenate((
            isi_cursors[:self.tx_ffe_taps['pre']], # All pre-cursors remain
            isi_cursors[self.tx_ffe_taps['pre'] + num_dfe_cancelled:] # Remaining post-cursors
        ))
        
        # 4. Calculate RMS value of residual ISI
        sigma_isi = np.sqrt(np.sum(residual_isi**2))

        # 5. Calculate noise from jitter
        # We need the derivative of the pulse response for this
        deriv_pulse_response = np.gradient(pulse_response, self.time_step)
        slope_at_crossings = np.mean(np.abs(deriv_pulse_response[peak_index - self.samples_per_ui//2:peak_index + self.samples_per_ui//2]))
        
        # Jitter noise is jitter_in_time * slope_at_sampling_point
        sigma_rj = self.random_jitter_rms * slope_at_crossings
        # For DJ, it's more complex, often approximated as pk-pk/sqrt(12) or similar
        sigma_dj = (self.deterministic_jitter_pkpk / np.sqrt(12)) * slope_at_crossings
        
        # 6. Sum all noise sources (RSS - Root Sum Square)
        sigma_total = np.sqrt(sigma_isi**2 + sigma_rj**2 + sigma_dj**2 + self.tx_noise_rms**2 + self.rx_noise_rms**2)
        
        # 7. Calculate COM
        # Q-factor for the target BER (inverse of Gaussian CDF)
        q_factor = np.sqrt(2) * erfinv(1 - 2 * self.target_ber)
        
        signal_amplitude = p_max
        noise_for_com = q_factor * sigma_total
        
        if noise_for_com == 0: return float('inf')
        
        com = 20 * np.log10(signal_amplitude / noise_for_com)
        
        noise_breakdown = {
            'sigma_isi': sigma_isi,
            'sigma_rj': sigma_rj,
            'sigma_dj': sigma_dj,
            'sigma_tx': self.tx_noise_rms,
            'sigma_rx': self.rx_noise_rms,
            'sigma_total': sigma_total
        }
        
        return com, p_max, noise_breakdown

def generate_pcie5_channel(length_inches=10):
    """Generates a synthetic S-parameter model for a PCB channel."""
    freq = np.linspace(0.01, 40, 400) # Freq vector up to 40 GHz
    
    # Simple loss model: Loss(f) = k1*sqrt(f) + k2*f + k3*f^2
    # k1: skin effect, k2: dielectric loss
    k1 = 0.15 * length_inches / 10
    k2 = 0.01 * length_inches / 10
    
    insertion_loss = -(k1 * np.sqrt(freq) + k2 * freq)
    
    # Add some ripple to simulate connector/via impedance discontinuities
    ripple = 0.5 * np.sin(2 * np.pi * freq / 10) * (freq / 40)
    insertion_loss += ripple
    
    return np.vstack((freq, insertion_loss)).T

def plot_results(s_params, time, pulse, p_max):
    """Visualizes the channel and the calculated pulse response."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Insertion Loss
    axes[0].plot(s_params[:, 0], s_params[:, 1], 'b-')
    axes[0].set_title('Channel Insertion Loss (Sdd21)')
    axes[0].set_xlabel('Frequency (GHz)')
    axes[0].set_ylabel('Insertion Loss (dB)')
    axes[0].grid(True)
    axes[0].axvline(16, color='r', linestyle='--', label='Nyquist (16 GHz)')
    axes[0].legend()

    # Plot 2: Pulse Response
    axes[1].plot(time / 1e-12, pulse, 'g-')
    axes[1].set_title('Equalized Single-Bit Pulse Response')
    axes[1].set_xlabel('Time (ps)')
    axes[1].set_ylabel('Amplitude (V)')
    axes[1].grid(True)
    
    # Mark the peak
    peak_time_ps = time[np.argmax(pulse)] / 1e-12
    axes[1].plot(peak_time_ps, p_max, 'ro', label=f'Peak = {p_max:.3f} V')
    
    # Mark UI intervals around the peak
    ui_ps = (1/32) * 1000
    for i in range(-5, 6):
        axes[1].axvline(peak_time_ps + i * ui_ps, color='k', linestyle=':', alpha=0.5)

    axes[1].legend()
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    # 1. Define channel and simulation parameters
    channel_length_inches = 12
    # Optimal Tx FFE coefficients need to be found with an optimizer.
    # Here, we use some plausible pre-emphasis values.
    tx_ffe_coefficients = {
        'pre': [-0.15],
        'main': 0.7,
        'post': [-0.10, -0.05]
    }

    # 2. Generate a synthetic channel model
    channel_s_params = generate_pcie5_channel(length_inches=channel_length_inches)

    # 3. Initialize and run the COM calculator
    com_sim = COMCalculator(channel_s_params)
    
    time_vec, pulse_resp = com_sim.calculate_pulse_response(tx_ffe_coefficients)
    
    com_result, peak_voltage, noise_stats = com_sim.run_com_calculation(pulse_resp)

    # 4. Print results
    print("--- PCIe Gen5 COM Calculation Results ---")
    print(f"Channel Length: {channel_length_inches} inches")
    print(f"Target BER: {com_sim.target_ber}")
    print("\n" + "="*40)
    print(f"  Channel Operating Margin (COM): {com_result:.2f} dB")
    print("="*40)
    # A COM value > 3 dB is typically considered passing for PCIe.
    if com_result > 3.0:
        print("  Result: PASS")
    else:
        print("  Result: FAIL")
        
    print("\n--- Signal and Noise Breakdown ---")
    print(f"  Peak Pulse Voltage (p_max): {peak_voltage * 1000:.2f} mV")
    print("  RMS Noise Components (in mV):")
    print(f"    - ISI Noise (sigma_isi):   {noise_stats['sigma_isi'] * 1000:.3f} mV")
    print(f"    - RJ Noise (sigma_rj):     {noise_stats['sigma_rj'] * 1000:.3f} mV")
    print(f"    - DJ Noise (sigma_dj):     {noise_stats['sigma_dj'] * 1000:.3f} mV")
    print(f"    - Tx Noise (sigma_tx):     {noise_stats['sigma_tx'] * 1000:.3f} mV")
    print(f"    - Rx Noise (sigma_rx):     {noise_stats['sigma_rx'] * 1000:.3f} mV")
    print(f"  -------------------------------------")
    print(f"    - Total RSS Noise (sigma_total): {noise_stats['sigma_total'] * 1000:.3f} mV")
    
    # 5. Visualize the results
    plot_results(channel_s_params, time_vec, pulse_resp, peak_voltage)