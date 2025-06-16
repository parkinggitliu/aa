import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize_scalar

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
    
    def tia_nonlinearity(self, i_in, r_f, v_sat=1.0, p3=0.1):
        """
        TIA non-linearity model
        
        Parameters:
        i_in: input current
        r_f: feedback resistance
        v_sat: saturation voltage
        p3: third-order intercept point parameter
        """
        # Linear response
        v_linear = i_in * r_f
        
        # Saturation effect
        v_sat_effect = v_sat * np.tanh(v_linear / v_sat)
        
        # Third-order non-linearity
        v_nonlinear = v_sat_effect + p3 * (v_linear / v_sat)**3
        
        return v_nonlinear
    
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
                          channel_h=None, temp=300, i_n=1e-12, 
                          include_nonlinearity=True):
        """
        Calculate Salz SNR including TIA effects
        
        Parameters:
        optical_power: received optical power (W)
        responsivity: photodiode responsivity (A/W)
        r_f: TIA feedback resistance (Ohm)
        c_f: TIA feedback capacitance (F)
        channel_h: channel impulse response (optional)
        temp: temperature (K)
        i_n: TIA input current noise (A/√Hz)
        include_nonlinearity: whether to include TIA non-linearity
        """
        # Signal current
        i_signal = optical_power * responsivity
        
        # Frequency range for analysis
        f_max = 10 * self.bit_rate
        f = np.linspace(0.1, f_max, 1000)
        
        # TIA transfer function
        h_tia = self.tia_transfer_function(f, r_f, c_f)
        
        # Signal power after TIA
        if include_nonlinearity:
            v_signal = self.tia_nonlinearity(i_signal, r_f)
            signal_power = np.abs(v_signal)**2
        else:
            signal_power = (i_signal * r_f)**2
        
        # Noise power spectral density
        s_n = self.tia_noise_psd(f, r_f, c_f, temp, i_n)
        
        # Integrate noise over signal bandwidth
        df = f[1] - f[0]
        noise_power = np.trapz(s_n, dx=df)
        
        # ISI penalty if channel response provided
        isi_penalty_db = 0
        if channel_h is not None:
            isi_ratio = self.calculate_isi_penalty(channel_h)
            isi_penalty_db = 10 * np.log10(1 + isi_ratio)
        
        # Calculate SNR
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear) - isi_penalty_db
        
        return {
            'snr_db': snr_db,
            'snr_linear': snr_linear,
            'signal_power': signal_power,
            'noise_power': noise_power,
            'isi_penalty_db': isi_penalty_db
        }
    
    def optimize_tia_parameters(self, optical_power, responsivity, temp=300):
        """
        Optimize TIA parameters for maximum SNR
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

# Example usage and demonstration
def demo_salz_snr():
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
    
    # Calculate SNR
    results = calc.calculate_salz_snr(optical_power, responsivity, r_f, c_f,
                                    temp=temp, i_n=i_n)
    
    print("Salz SNR Analysis Results:")
    print(f"SNR: {results['snr_db']:.2f} dB")
    print(f"Signal Power: {results['signal_power']*1e6:.2f} µW")
    print(f"Noise Power: {results['noise_power']*1e12:.2f} pW")
    print(f"ISI Penalty: {results['isi_penalty_db']:.2f} dB")
    
    # Optimize TIA parameters
    optimal_rf, max_snr = calc.optimize_tia_parameters(optical_power, responsivity)
    print(f"\nOptimal feedback resistance: {optimal_rf/1e3:.1f} kOhm")
    print(f"Maximum achievable SNR: {max_snr:.2f} dB")
    
    # Plot SNR vs optical power
    powers = np.logspace(-6, -2, 50)  # 1 µW to 10 mW
    snrs = []
    
    for p in powers:
        result = calc.calculate_salz_snr(p, responsivity, r_f, c_f)
        snrs.append(result['snr_db'])
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(powers*1e3, snrs)
    plt.xlabel('Optical Power (mW)')
    plt.ylabel('SNR (dB)')
    plt.title('Salz SNR vs Optical Power (including TIA effects)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    demo_salz_snr()