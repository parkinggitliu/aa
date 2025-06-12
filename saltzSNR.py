import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize_scalar
from scipy.fft import fft, fftfreq, ifft

class SalzSNRCalculator:
    def __init__(self, bit_rate, pulse_shape='nrz', alpha=0.0):
        """
        Initialize Salz SNR calculator
        
        Parameters:
        bit_rate: Symbol rate in Hz
        pulse_shape: 'nrz', 'rz', or 'raised_cosine'
        alpha: Roll-off factor for raised cosine (0 <= alpha <= 1)
        """
        self.bit_rate = bit_rate
        self.T = 1 / bit_rate  # Symbol period
        self.pulse_shape = pulse_shape
        self.alpha = alpha
        
    def generate_pulse_shape(self, t):
        """Generate pulse shape function"""
        if self.pulse_shape == 'nrz':
            return np.where(np.abs(t) <= self.T/2, 1.0, 0.0)
        elif self.pulse_shape == 'rz':
            return np.where(np.abs(t) <= self.T/4, 1.0, 0.0)
        elif self.pulse_shape == 'raised_cosine':
            return self._raised_cosine_pulse(t)
        else:
            raise ValueError("Unsupported pulse shape")
    
    def _raised_cosine_pulse(self, t):
        """Raised cosine pulse shape"""
        t_norm = t / self.T
        
        # Handle special cases to avoid division by zero
        if self.alpha == 0:
            return np.sinc(t_norm)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            numerator = np.sin(np.pi * t_norm) * np.cos(np.pi * self.alpha * t_norm)
            denominator = np.pi * t_norm * (1 - (2 * self.alpha * t_norm)**2)
            
            # Handle t = 0 case
            result = np.where(t == 0, 1.0, numerator / denominator)
            
            # Handle t = ±T/(2*alpha) cases
            if self.alpha > 0:
                critical_points = np.abs(t_norm) == 1/(2*self.alpha)
                result = np.where(critical_points, 
                                self.alpha/2 * np.sin(np.pi/(2*self.alpha)), 
                                result)
        
        return result
    
    def tia_transfer_function(self, f, r_f, c_f, r_in=50, c_in=1e-12):
        """
        TIA transfer function including parasitic capacitance
        
        Parameters:
        f: frequency array
        r_f: feedback resistance
        c_f: feedback capacitance
        r_in: input resistance
        c_in: input capacitance
        """
        s = 2j * np.pi * f
        
        # TIA gain with parasitic effects
        z_f = r_f / (1 + s * r_f * c_f)  # Feedback impedance
        z_in = r_in / (1 + s * r_in * c_in)  # Input impedance
        
        # Simplified TIA transfer function
        h_tia = -z_f / (1 + z_f / z_in)
        
        return h_tia
    
    def tia_noise_psd(self, f, r_f, c_f, temp=300, i_n=1e-12):
        """
        TIA noise power spectral density
        
        Parameters:
        f: frequency array
        r_f: feedback resistance
        c_f: feedback capacitance
        temp: temperature in Kelvin
        i_n: input current noise density (A/√Hz)
        """
        k_b = 1.38e-23  # Boltzmann constant
        
        # Thermal noise from feedback resistor
        v_n_thermal = 4 * k_b * temp * r_f
        
        # Current noise contribution
        h_tia = self.tia_transfer_function(f, r_f, c_f)
        v_n_current = (i_n * np.abs(h_tia))**2
        
        # Total noise PSD
        s_n = v_n_thermal + v_n_current
        
        return s_n
    
    def tia_nonlinear_response(self, i_in, r_f, nonlin_params):
        """
        Comprehensive TIA non-linearity model including harmonics and intermodulation
        
        Parameters:
        i_in: input current (can be time-domain signal)
        r_f: feedback resistance
        nonlin_params: dictionary with non-linearity parameters
            - v_sat: saturation voltage
            - a2: second-order coefficient
            - a3: third-order coefficient
            - iip2: second-order intercept point (dBm)
            - iip3: third-order intercept point (dBm)
            - compression_point: 1dB compression point
        """
        # Linear response
        v_linear = i_in * r_f
        
        # Extract parameters
        v_sat = nonlin_params.get('v_sat', 1.0)
        a2 = nonlin_params.get('a2', 0.01)
        a3 = nonlin_params.get('a3', 0.001)
        iip2 = nonlin_params.get('iip2', 30)  # dBm
        iip3 = nonlin_params.get('iip3', 20)  # dBm
        p1db = nonlin_params.get('compression_point', 10)  # dBm
        
        # Convert intercept points to linear scale
        iip2_linear = 10**(iip2/10) * 1e-3  # Convert dBm to W
        iip3_linear = 10**(iip3/10) * 1e-3
        p1db_linear = 10**(p1db/10) * 1e-3
        
        # Polynomial non-linearity model
        v_nonlinear = v_linear.copy()
        
        # Second-order distortion
        v_nonlinear += a2 * v_linear**2
        
        # Third-order distortion
        v_nonlinear += a3 * v_linear**3
        
        # Saturation/compression
        compression_factor = 1 / (1 + np.abs(v_linear)**2 / p1db_linear)
        v_nonlinear *= compression_factor
        
        # Hard saturation
        v_nonlinear = np.clip(v_nonlinear, -v_sat, v_sat)
        
        return v_nonlinear
    
    def calculate_harmonic_distortion(self, signal_freq, signal_amp, r_f, nonlin_params, fs=None):
        """
        Calculate harmonic distortion components
        
        Parameters:
        signal_freq: fundamental frequency
        signal_amp: signal amplitude
        r_f: feedback resistance
        nonlin_params: non-linearity parameters
        fs: sampling frequency (if None, auto-calculated)
        """
        if fs is None:
            fs = 20 * signal_freq  # Nyquist criterion with margin
        
        # Generate time vector
        n_cycles = 10
        t = np.linspace(0, n_cycles/signal_freq, int(fs * n_cycles/signal_freq))
        dt = t[1] - t[0]
        
        # Generate sinusoidal input current
        i_in = signal_amp * np.sin(2 * np.pi * signal_freq * t)
        
        # Apply TIA non-linearity
        v_out = self.tia_nonlinear_response(i_in, r_f, nonlin_params)
        
        # FFT analysis
        V_fft = fft(v_out)
        freqs = fftfreq(len(t), dt)
        
        # Find harmonics
        fundamental_idx = np.argmin(np.abs(freqs - signal_freq))
        harmonic_powers = {}
        
        # Calculate power of fundamental and harmonics
        fundamental_power = np.abs(V_fft[fundamental_idx])**2
        
        for h in range(2, 6):  # Up to 5th harmonic
            harmonic_freq = h * signal_freq
            if harmonic_freq < fs/2:  # Within Nyquist limit
                harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                harmonic_power = np.abs(V_fft[harmonic_idx])**2
                harmonic_powers[f'H{h}'] = harmonic_power
        
        return fundamental_power, harmonic_powers, freqs, V_fft
    
    def calculate_intermodulation_distortion(self, f1, f2, amp1, amp2, r_f, nonlin_params, fs=None):
        """
        Calculate intermodulation distortion for two-tone test
        
        Parameters:
        f1, f2: two test frequencies
        amp1, amp2: amplitudes of test tones
        r_f: feedback resistance
        nonlin_params: non-linearity parameters
        fs: sampling frequency
        """
        if fs is None:
            fs = 20 * max(f1, f2)
        
        # Generate time vector
        n_cycles = 20
        t_max = n_cycles / min(f1, f2)
        t = np.linspace(0, t_max, int(fs * t_max))
        dt = t[1] - t[0]
        
        # Generate two-tone input current
        i_in = amp1 * np.sin(2 * np.pi * f1 * t) + amp2 * np.sin(2 * np.pi * f2 * t)
        
        # Apply TIA non-linearity
        v_out = self.tia_nonlinear_response(i_in, r_f, nonlin_params)
        
        # FFT analysis
        V_fft = fft(v_out)
        freqs = fftfreq(len(t), dt)
        
        # Find fundamental tones
        f1_idx = np.argmin(np.abs(freqs - f1))
        f2_idx = np.argmin(np.abs(freqs - f2))
        
        fundamental_power_f1 = np.abs(V_fft[f1_idx])**2
        fundamental_power_f2 = np.abs(V_fft[f2_idx])**2
        
        # Calculate intermodulation products
        imd_products = {}
        
        # Second-order IMD: f1±f2
        for sign in [1, -1]:
            imd_freq = f1 + sign * f2
            if 0 < imd_freq < fs/2:
                imd_idx = np.argmin(np.abs(freqs - imd_freq))
                imd_power = np.abs(V_fft[imd_idx])**2
                imd_products[f'IM2_{f1}{"+/-"[sign<0]}{f2}'] = imd_power
        
        # Third-order IMD: 2f1±f2, 2f2±f1
        for f_base, f_other in [(f1, f2), (f2, f1)]:
            for sign in [1, -1]:
                imd_freq = 2 * f_base + sign * f_other
                if 0 < imd_freq < fs/2:
                    imd_idx = np.argmin(np.abs(freqs - imd_freq))
                    imd_power = np.abs(V_fft[imd_idx])**2
                    imd_products[f'IM3_2*{f_base}{"+/-"[sign<0]}{f_other}'] = imd_power
        
        return (fundamental_power_f1, fundamental_power_f2), imd_products, freqs, V_fft
    
    def calculate_nonlinear_noise_power(self, signal_power, nonlin_params, r_f):
        """
        Calculate total non-linear distortion noise power
        
        Parameters:
        signal_power: input signal power
        nonlin_params: non-linearity parameters
        r_f: feedback resistance
        """
        # Estimate signal amplitude from power
        signal_amp = np.sqrt(2 * signal_power)
        
        # Calculate harmonic distortion at fundamental frequency
        fund_freq = self.bit_rate / 2  # Approximate fundamental
        fund_power, harmonic_powers, _, _ = self.calculate_harmonic_distortion(
            fund_freq, signal_amp, r_f, nonlin_params)
        
        # Total harmonic distortion power
        thd_power = sum(harmonic_powers.values())
        
        # Estimate IMD power using two-tone test at nearby frequencies
        f1 = fund_freq * 0.9
        f2 = fund_freq * 1.1
        (p1, p2), imd_powers, _, _ = self.calculate_intermodulation_distortion(
            f1, f2, signal_amp/2, signal_amp/2, r_f, nonlin_params)
        
        # Total IMD power
        imd_power = sum(imd_powers.values())
        
        # Total non-linear noise
        nonlinear_noise_power = thd_power + imd_power
        
        return nonlinear_noise_power
    
    def calculate_isi_penalty(self, channel_response, num_symbols=100):
        """
        Calculate ISI penalty using Salz method
        
        Parameters:
        channel_response: impulse response of the channel
        num_symbols: number of symbols to consider for ISI calculation
        """
        # Sample at symbol rate
        dt = self.T / 100  # Oversampling
        t = np.arange(-num_symbols*self.T, num_symbols*self.T, dt)
        
        # Generate pulse shape
        pulse = self.generate_pulse_shape(t)
        
        # Convolve with channel response
        received_pulse = np.convolve(pulse, channel_response, mode='same')
        
        # Find peak (main cursor)
        peak_idx = np.argmax(np.abs(received_pulse))
        peak_value = received_pulse[peak_idx]
        
        # Calculate ISI from pre- and post-cursors
        symbol_indices = np.arange(peak_idx - num_symbols//2 * int(self.T/dt),
                                 peak_idx + num_symbols//2 * int(self.T/dt),
                                 int(self.T/dt))
        
        isi_values = received_pulse[symbol_indices]
        isi_values[len(isi_values)//2] = 0  # Remove main cursor
        
        isi_power = np.sum(np.abs(isi_values)**2)
        signal_power = np.abs(peak_value)**2
        
        return isi_power / signal_power
    
    def calculate_salz_snr(self, optical_power, responsivity, r_f, c_f, 
                          nonlin_params=None, channel_h=None, temp=300, 
                          i_n=1e-12, include_nonlinearity=True):
        """
        Calculate Salz SNR including comprehensive TIA non-linearity effects
        
        Parameters:
        optical_power: received optical power (W)
        responsivity: photodiode responsivity (A/W)
        r_f: TIA feedback resistance (Ohm)
        c_f: TIA feedback capacitance (F)
        nonlin_params: non-linearity parameters dictionary
        channel_h: channel impulse response (optional)
        temp: temperature (K)
        i_n: TIA input current noise (A/√Hz)
        include_nonlinearity: whether to include TIA non-linearity
        """
        # Default non-linearity parameters
        if nonlin_params is None:
            nonlin_params = {
                'v_sat': 1.0,
                'a2': 0.01,
                'a3': 0.001,
                'iip2': 30,
                'iip3': 20,
                'compression_point': 10
            }
        
        # Signal current and power
        i_signal = optical_power * responsivity
        signal_power = (i_signal * r_f)**2
        
        # Frequency range for analysis
        f_max = 10 * self.bit_rate
        f = np.linspace(0.1, f_max, 1000)
        
        # Linear noise power spectral density
        s_n_linear = self.tia_noise_psd(f, r_f, c_f, temp, i_n)
        df = f[1] - f[0]
        linear_noise_power = np.trapz(s_n_linear, dx=df)
        
        # Non-linear distortion noise
        nonlinear_noise_power = 0
        if include_nonlinearity:
            nonlinear_noise_power = self.calculate_nonlinear_noise_power(
                signal_power, nonlin_params, r_f)
        
        # Total noise power
        total_noise_power = linear_noise_power + nonlinear_noise_power
        
        # ISI penalty if channel response provided
        isi_penalty_db = 0
        if channel_h is not None:
            isi_ratio = self.calculate_isi_penalty(channel_h)
            isi_penalty_db = 10 * np.log10(1 + isi_ratio)
        
        # Calculate SNR
        snr_linear = signal_power / total_noise_power
        snr_db = 10 * np.log10(snr_linear) - isi_penalty_db
        
        return {
            'snr_db': snr_db,
            'snr_linear': snr_linear,
            'signal_power': signal_power,
            'linear_noise_power': linear_noise_power,
            'nonlinear_noise_power': nonlinear_noise_power,
            'total_noise_power': total_noise_power,
            'isi_penalty_db': isi_penalty_db
        }
    
    def analyze_distortion_components(self, optical_power, responsivity, r_f, nonlin_params):
        """
        Detailed analysis of distortion components
        """
        i_signal = optical_power * responsivity
        signal_power = (i_signal * r_f)**2
        signal_amp = np.sqrt(2 * signal_power)
        
        # Harmonic analysis
        fund_freq = self.bit_rate / 2
        fund_power, harmonic_powers, _, _ = self.calculate_harmonic_distortion(
            fund_freq, signal_amp, r_f, nonlin_params)
        
        # IMD analysis
        f1, f2 = fund_freq * 0.9, fund_freq * 1.1
        (p1, p2), imd_powers, _, _ = self.calculate_intermodulation_distortion(
            f1, f2, signal_amp/2, signal_amp/2, r_f, nonlin_params)
        
        return {
            'fundamental_power': fund_power,
            'harmonic_powers': harmonic_powers,
            'imd_powers': imd_powers,
            'thd_db': 10 * np.log10(sum(harmonic_powers.values()) / fund_power),
            'total_distortion_power': sum(harmonic_powers.values()) + sum(imd_powers.values())
        }
    
    def optimize_tia_parameters(self, optical_power, responsivity, temp=300):
        """
        Optimize TIA parameters for maximum SNR considering non-linearity
        """
        def objective(log_rf):
            r_f = 10**log_rf
            c_f = 1e-12  # Fixed feedback capacitance
            
            result = self.calculate_salz_snr(optical_power, responsivity, 
                                           r_f, c_f, temp=temp)
            return -result['snr_db']  # Minimize negative SNR
        
        # Optimize over reasonable range of feedback resistance
        result = minimize_scalar(objective, bounds=(3, 7), method='bounded')
        optimal_rf = 10**result.x
        
        return optimal_rf, -result.fun

# Enhanced demonstration with non-linearity analysis
def demo_salz_snr_with_nonlinearity():
    # System parameters
    bit_rate = 10e9  # 10 Gbps
    calc = SalzSNRCalculator(bit_rate, pulse_shape='nrz')
    
    # Optical and electrical parameters
    optical_power = 1e-3  # 1 mW
    responsivity = 0.8  # A/W
    r_f = 10e3  # 10 kOhm feedback resistance
    c_f = 0.1e-12  # 0.1 pF feedback capacitance
    temp = 300  # 300 K
    i_n = 1e-12  # 1 pA/√Hz input current noise
    
    # Non-linearity parameters
    nonlin_params = {
        'v_sat': 1.0,      # 1V saturation
        'a2': 0.02,        # Second-order coefficient
        'a3': 0.005,       # Third-order coefficient
        'iip2': 35,        # dBm
        'iip3': 25,        # dBm
        'compression_point': 15  # dBm
    }
    
    # Calculate SNR with and without non-linearity
    results_linear = calc.calculate_salz_snr(optical_power, responsivity, r_f, c_f,
                                           temp=temp, i_n=i_n, include_nonlinearity=False)
    
    results_nonlinear = calc.calculate_salz_snr(optical_power, responsivity, r_f, c_f,
                                              nonlin_params=nonlin_params,
                                              temp=temp, i_n=i_n, include_nonlinearity=True)
    
    print("Salz SNR Analysis with Non-linearity Effects:")
    print(f"SNR (linear only): {results_linear['snr_db']:.2f} dB")
    print(f"SNR (with non-linearity): {results_nonlinear['snr_db']:.2f} dB")
    print(f"Non-linearity penalty: {results_linear['snr_db'] - results_nonlinear['snr_db']:.2f} dB")
    print(f"Linear noise power: {results_nonlinear['linear_noise_power']*1e12:.2f} pW")
    print(f"Non-linear distortion power: {results_nonlinear['nonlinear_noise_power']*1e12:.2f} pW")
    
    # Analyze distortion components
    distortion_analysis = calc.analyze_distortion_components(
        optical_power, responsivity, r_f, nonlin_params)
    
    print(f"\nDistortion Analysis:")
    print(f"THD: {distortion_analysis['thd_db']:.2f} dB")
    print(f"Harmonic powers: {distortion_analysis['harmonic_powers']}")
    print(f"IMD powers: {distortion_analysis['imd_powers']}")
    
    # Plot SNR vs optical power showing non-linearity impact
    powers = np.logspace(-6, -2, 50)  # 1 µW to 10 mW
    snrs_linear = []
    snrs_nonlinear = []
    
    for p in powers:
        result_lin = calc.calculate_salz_snr(p, responsivity, r_f, c_f, 
                                           include_nonlinearity=False)
        result_nonlin = calc.calculate_salz_snr(p, responsivity, r_f, c_f,
                                              nonlin_params=nonlin_params,
                                              include_nonlinearity=True)
        snrs_linear.append(result_lin['snr_db'])
        snrs_nonlinear.append(result_nonlin['snr_db'])
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.semilogx(powers*1e3, snrs_linear, 'b-', label='Linear only')
    plt.semilogx(powers*1e3, snrs_nonlinear, 'r-', label='With non-linearity')
    plt.xlabel('Optical Power (mW)')
    plt.ylabel('SNR (dB)')
    plt.title('SNR vs Optical Power')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    penalty = np.array(snrs_linear) - np.array(snrs_nonlinear)
    plt.semilogx(powers*1e3, penalty)
    plt.xlabel('Optical Power (mW)')
    plt.ylabel('Non-linearity Penalty (dB)')
    plt.title('Non-linearity Penalty vs Power')
    plt.grid(True)
    
    # Harmonic spectrum analysis
    plt.subplot(2, 2, 3)
    fund_freq = bit_rate / 2
    signal_amp = np.sqrt(2 * (optical_power * responsivity * r_f)**2)
    fund_power, harmonic_powers, freqs, V_fft = calc.calculate_harmonic_distortion(
        fund_freq, signal_amp, r_f, nonlin_params)
    
    freq_plot = freqs[:len(freqs)//2]
    spectrum_plot = 20 * np.log10(np.abs(V_fft[:len(V_fft)//2]) + 1e-12)
    plt.semilogy(freq_plot/1e9, 10**(spectrum_plot/20))
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Amplitude (V)')
    plt.title('Output Spectrum with Harmonics')
    plt.grid(True)
    
    # IMD spectrum
    plt.subplot(2, 2, 4)
    f1, f2 = fund_freq * 0.9, fund_freq * 1.1
    (p1, p2), imd_powers, freqs_imd, V_fft_imd = calc.calculate_intermodulation_distortion(
        f1, f2, signal_amp/2, signal_amp/2, r_f, nonlin_params)
    
    freq_imd_plot = freqs_imd[:len(freqs_imd)//2]
    spectrum_imd_plot = 20 * np.log10(np.abs(V_fft_imd[:len(V_fft_imd)//2]) + 1e-12)
    plt.semilogy(freq_imd_plot/1e9, 10**(spectrum_imd_plot/20))
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Amplitude (V)')
    plt.title('Two-Tone IMD Spectrum')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo_salz_snr_with_nonlinearity()