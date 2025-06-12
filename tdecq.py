import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize
from scipy.stats import norm
import warnings

class PAM4_TDECQ_Calculator:
    """
    PAM4 TDECQ (Transmitter and Dispersion Eye Closure Quaternary) and 
    RLM (Receiver Level Mismatch) Calculator
    
    Based on IEEE 802.3bs specifications for 50G/100G/200G optical links
    """
    
    def __init__(self, samples_per_symbol=64, symbol_rate=26.5625e9):
        """
        Initialize TDECQ calculator
        
        Parameters:
        - samples_per_symbol: Oversampling rate
        - symbol_rate: Symbol rate in Hz (26.5625 GBaud for 50G PAM4)
        """
        self.sps = samples_per_symbol
        self.symbol_rate = symbol_rate
        self.sample_rate = symbol_rate * samples_per_symbol
        
        # PAM4 levels (normalized)
        self.pam4_levels = np.array([0, 1/3, 2/3, 1])
        
        # Target BER for TDECQ (2.4e-4 for FEC threshold)
        self.target_ber = 2.4e-4
        
        # Bessel-Thomson filter parameters (4th order, 13 GHz for 26.5625 GBaud)
        self.bt_filter_bw = 13e9
        
    def create_bessel_thomson_filter(self, order=4):
        """
        Create Bessel-Thomson filter for reference receiver
        """
        # Normalized cutoff frequency
        wn = 2 * np.pi * self.bt_filter_bw / self.sample_rate
        
        # Bessel filter poles (normalized)
        if order == 4:
            # 4th order Bessel poles
            poles = np.array([
                -0.9952 + 0j,
                -0.6573 + 0.8302j,
                -0.6573 - 0.8302j,
                -0.9047 + 0j
            ])
        else:
            raise ValueError("Only 4th order Bessel-Thomson implemented")
        
        # Scale poles by cutoff frequency
        poles = poles * wn
        
        # Convert to discrete-time filter
        dt = 1 / self.sample_rate
        z_poles = np.exp(poles * dt)
        
        # Create filter (all-pole, unity DC gain)
        b = [1]
        a = np.poly(z_poles).real
        
        # Normalize for unity gain at DC
        gain = np.abs(np.polyval(b, 1) / np.polyval(a, 1))
        b = b / gain
        
        return b, a
    
    def apply_rx_filter(self, signal_in):
        """Apply reference receiver filter (Bessel-Thomson)"""
        b, a = self.create_bessel_thomson_filter()
        # Forward-backward filtering (equivalent to filtfilt)
        # First forward pass
        forward = signal.lfilter(b, a, signal_in)
        # Reverse the signal
        reversed_signal = forward[::-1]
        # Second pass on reversed signal
        backward = signal.lfilter(b, a, reversed_signal)
        # Reverse again to get final result
        return backward[::-1]
    
    def find_symbol_centers(self, waveform, num_symbols):
        """
        Find optimal sampling phase and extract symbol centers
        """
        best_phase = 0
        max_eye_opening = 0
        
        # Search for optimal sampling phase
        for phase in range(self.sps):
            samples = waveform[phase::self.sps][:num_symbols]
            
            # Calculate eye opening metric (sum of level separations)
            if len(samples) > 100:
                sorted_samples = np.sort(samples)
                # Estimate levels
                levels = []
                for i in range(4):
                    start_idx = int(i * len(sorted_samples) / 4)
                    end_idx = int((i + 1) * len(sorted_samples) / 4)
                    levels.append(np.mean(sorted_samples[start_idx:end_idx]))
                
                # Eye opening metric
                eye_opening = sum(levels[i+1] - levels[i] for i in range(3))
                
                if eye_opening > max_eye_opening:
                    max_eye_opening = eye_opening
                    best_phase = phase
        
        # Extract symbols at optimal phase
        symbols = waveform[best_phase::self.sps][:num_symbols]
        
        return symbols, best_phase
    
    def estimate_pam4_levels(self, symbols):
        """
        Estimate PAM4 levels from symbol samples using histogram method
        """
        # Create histogram
        hist, bin_edges = np.histogram(symbols, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find peaks (PAM4 levels)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist, height=len(symbols)/20, distance=10)
        
        if len(peaks) != 4:
            # Fallback: use k-means clustering
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=4, random_state=0)
            symbols_reshaped = symbols.reshape(-1, 1)
            kmeans.fit(symbols_reshaped)
            levels = np.sort(kmeans.cluster_centers_.flatten())
        else:
            levels = np.sort(bin_centers[peaks])
        
        return levels
    
    def calculate_rlm(self, levels):
        """
        Calculate Receiver Level Mismatch (RLM)
        
        RLM = max(|L1-L0-a|, |L2-L1-a|, |L3-L2-a|) / a
        where a = (L3-L0)/3
        """
        L0, L1, L2, L3 = levels
        
        # Average level spacing
        avg_spacing = (L3 - L0) / 3
        
        # Calculate deviations
        dev1 = abs(L1 - L0 - avg_spacing)
        dev2 = abs(L2 - L1 - avg_spacing)
        dev3 = abs(L3 - L2 - avg_spacing)
        
        # RLM is the maximum deviation normalized by average spacing
        rlm = max(dev1, dev2, dev3) / avg_spacing
        
        return rlm
    
    def calculate_noise_rms(self, symbols, levels):
        """
        Calculate RMS noise for each PAM4 level
        """
        noise_rms = np.zeros(4)
        
        for i in range(4):
            # Find symbols belonging to this level
            if i == 0:
                mask = symbols < (levels[0] + levels[1]) / 2
            elif i == 1:
                mask = ((symbols >= (levels[0] + levels[1]) / 2) & 
                        (symbols < (levels[1] + levels[2]) / 2))
            elif i == 2:
                mask = ((symbols >= (levels[1] + levels[2]) / 2) & 
                        (symbols < (levels[2] + levels[3]) / 2))
            else:
                mask = symbols >= (levels[2] + levels[3]) / 2
            
            level_symbols = symbols[mask]
            if len(level_symbols) > 10:
                noise_rms[i] = np.std(level_symbols)
        
        return noise_rms
    
    def ceq_optimization(self, symbols, levels, noise_rms, target_ber=2.4e-4):
        """
        Optimize Continuous-time Equalizer (CEQ) taps to minimize TDECQ
        
        This is a simplified version - real implementation would include
        actual FFE tap optimization
        """
        # For this example, we'll use a simple gain optimization
        # Real implementation would optimize multiple FFE taps
        
        def tdecq_cost(gain):
            # Apply gain
            eq_symbols = symbols * gain[0]
            
            # Calculate new levels
            new_levels = levels * gain[0]
            
            # Calculate Q-factors for each transition
            q_values = []
            
            # Lower eye (0->1 transition)
            if noise_rms[0] > 0 and noise_rms[1] > 0:
                threshold = (new_levels[0] + new_levels[1]) / 2
                q_lower = (new_levels[1] - new_levels[0]) / (noise_rms[0] + noise_rms[1])
                q_values.append(q_lower)
            
            # Middle eye (1->2 transition)
            if noise_rms[1] > 0 and noise_rms[2] > 0:
                threshold = (new_levels[1] + new_levels[2]) / 2
                q_middle = (new_levels[2] - new_levels[1]) / (noise_rms[1] + noise_rms[2])
                q_values.append(q_middle)
            
            # Upper eye (2->3 transition)
            if noise_rms[2] > 0 and noise_rms[3] > 0:
                threshold = (new_levels[2] + new_levels[3]) / 2
                q_upper = (new_levels[3] - new_levels[2]) / (noise_rms[2] + noise_rms[3])
                q_values.append(q_upper)
            
            # Find minimum Q (worst case)
            if q_values:
                q_min = min(q_values)
                # Convert to equivalent BER
                ber = 0.5 * (1 - norm.cdf(q_min))
                
                # TDECQ in dB (relative to target BER)
                if ber > 0:
                    tdecq = 20 * np.log10(norm.ppf(1 - 2*target_ber) / norm.ppf(1 - 2*ber))
                else:
                    tdecq = 0
            else:
                tdecq = 100  # Large penalty
            
            return tdecq
        
        # Optimize gain
        result = optimize.minimize(tdecq_cost, [1.0], bounds=[(0.5, 2.0)])
        optimal_gain = result.x[0]
        min_tdecq = result.fun
        
        return min_tdecq, optimal_gain
    
    def calculate_tdecq(self, waveform, num_symbols=None):
        """
        Calculate TDECQ for PAM4 waveform
        
        Parameters:
        - waveform: PAM4 signal waveform
        - num_symbols: Number of symbols to analyze
        
        Returns:
        - tdecq: TDECQ value in dB
        - rlm: Receiver Level Mismatch
        - results: Dictionary with detailed results
        """
        if num_symbols is None:
            num_symbols = len(waveform) // self.sps
        
        # Apply reference receiver filter
        filtered_waveform = self.apply_rx_filter(waveform)
        
        # Find symbol centers
        symbols, sampling_phase = self.find_symbol_centers(filtered_waveform, num_symbols)
        
        # Estimate PAM4 levels
        levels = self.estimate_pam4_levels(symbols)
        
        # Calculate RLM
        rlm = self.calculate_rlm(levels)
        
        # Calculate noise RMS for each level
        noise_rms = self.calculate_noise_rms(symbols, levels)
        
        # Optimize CEQ and calculate TDECQ
        tdecq, optimal_gain = self.ceq_optimization(symbols, levels, noise_rms, self.target_ber)
        
        # Compile results
        results = {
            'levels': levels,
            'noise_rms': noise_rms,
            'sampling_phase': sampling_phase,
            'optimal_gain': optimal_gain,
            'symbols': symbols,
            'filtered_waveform': filtered_waveform
        }
        
        return tdecq, rlm, results
    
    def plot_eye_diagram(self, waveform, title="PAM4 Eye Diagram"):
        """Plot eye diagram"""
        plt.figure(figsize=(10, 8))
        
        # Plot multiple symbol periods
        num_traces = min(1000, len(waveform) // (2 * self.sps))
        time_axis = np.linspace(0, 2, 2 * self.sps)
        
        for i in range(num_traces):
            start_idx = i * self.sps
            if start_idx + 2 * self.sps <= len(waveform):
                plt.plot(time_axis, waveform[start_idx:start_idx + 2*self.sps], 
                        'b', alpha=0.01, linewidth=0.5)
        
        plt.xlabel('Time (UI)')
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 2)
        
    def plot_histogram(self, symbols, levels, title="PAM4 Level Histogram"):
        """Plot histogram with identified levels"""
        plt.figure(figsize=(10, 6))
        
        plt.hist(symbols, bins=100, alpha=0.7, density=True)
        
        # Plot identified levels
        for i, level in enumerate(levels):
            plt.axvline(level, color='red', linestyle='--', 
                       label=f'Level {i}: {level:.3f}')
        
        plt.xlabel('Amplitude')
        plt.ylabel('Density')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

# Example usage
if __name__ == "__main__":
    # Generate example PAM4 signal
    np.random.seed(42)
    
    # Parameters
    num_symbols = 10000
    sps = 64
    snr_db = 25
    
    # Generate random PAM4 symbols
    symbols = np.random.randint(0, 4, num_symbols)
    pam4_levels = np.array([0, 1/3, 2/3, 1])
    symbol_values = pam4_levels[symbols]
    
    # Create waveform with pulse shaping (raised cosine)
    # Upsample
    upsampled = np.zeros(num_symbols * sps)
    upsampled[::sps] = symbol_values
    
    # Simple raised cosine filter
    beta = 0.3
    span = 10
    t = np.arange(-span*sps/2, span*sps/2) / sps
    h = np.sinc(t) * np.cos(np.pi*beta*t) / (1 - (2*beta*t)**2 + 1e-10)
    h = h / np.sum(h)
    
    # Apply pulse shaping
    waveform = np.convolve(upsampled, h, mode='same')
    
    # Add some ISI (Inter-Symbol Interference)
    isi_filter = np.array([1, 0.1, -0.05])
    waveform = np.convolve(waveform, isi_filter, mode='same')
    
    # Add noise
    noise_power = np.mean(waveform**2) / (10**(snr_db/10))
    noise = np.sqrt(noise_power) * np.random.randn(len(waveform))
    waveform += noise
    
    # Add level mismatch (to create non-zero RLM)
    # Distort levels slightly
    for i, sym in enumerate(symbols[:len(waveform)//sps]):
        if sym == 1:
            waveform[i*sps:(i+1)*sps] *= 0.95  # Slightly compress level 1
        elif sym == 2:
            waveform[i*sps:(i+1)*sps] *= 1.02  # Slightly expand level 2
    
    # Create TDECQ calculator
    calculator = PAM4_TDECQ_Calculator(samples_per_symbol=sps)
    
    # Calculate TDECQ and RLM
    print("Calculating TDECQ and RLM...")
    tdecq, rlm, results = calculator.calculate_tdecq(waveform, num_symbols=5000)
    
    print(f"\nResults:")
    print(f"TDECQ: {tdecq:.2f} dB")
    print(f"RLM: {rlm:.3f} ({rlm*100:.1f}%)")
    print(f"\nPAM4 Levels:")
    for i, level in enumerate(results['levels']):
        print(f"  Level {i}: {level:.4f}")
    print(f"\nNoise RMS:")
    for i, noise in enumerate(results['noise_rms']):
        print(f"  Level {i}: {noise:.4f}")
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Eye diagram
    plt.subplot(2, 2, 1)
    calculator.plot_eye_diagram(waveform[:10000], "Original Eye Diagram")
    
    # Filtered eye diagram
    plt.subplot(2, 2, 2)
    calculator.plot_eye_diagram(results['filtered_waveform'][:10000], 
                               "Filtered Eye Diagram (Bessel-Thomson)")
    
    # Histogram
    plt.subplot(2, 2, 3)
    calculator.plot_histogram(results['symbols'], results['levels'])
    
    # TDECQ visualization
    plt.subplot(2, 2, 4)
    # Plot Q-factor for each eye
    eyes = ['Lower (0-1)', 'Middle (1-2)', 'Upper (2-3)']
    q_factors = []
    
    levels = results['levels']
    noise_rms = results['noise_rms']
    
    for i in range(3):
        if noise_rms[i] > 0 and noise_rms[i+1] > 0:
            q = (levels[i+1] - levels[i]) / (noise_rms[i] + noise_rms[i+1])
            q_factors.append(q)
        else:
            q_factors.append(0)
    
    plt.bar(eyes, q_factors)
    plt.axhline(y=norm.ppf(1 - calculator.target_ber), color='r', 
                linestyle='--', label=f'Target Q (BER={calculator.target_ber})')
    plt.xlabel('Eye')
    plt.ylabel('Q-factor')
    plt.title('Q-factors for Each Eye')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis: BER estimation
    print(f"\nEstimated BER for each eye:")
    for i, (eye, q) in enumerate(zip(eyes, q_factors)):
        if q > 0:
            ber = 0.5 * (1 - norm.cdf(q))
            print(f"  {eye}: BER = {ber:.2e} (Q = {q:.2f})")


def example_optical_measurement():
    """
    Example showing how to use with actual optical measurements
    """
    print("\n" + "="*60)
    print("Example: Optical Transmitter Measurement")
    print("="*60)
    
    # In practice, you would load this from an oscilloscope
    # Here we simulate an optical PAM4 signal with typical impairments
    
    # Parameters for optical link
    symbol_rate = 26.5625e9  # 26.5625 GBaud
    sps = 64
    num_symbols = 20000
    
    # Generate PAM4 signal with optical impairments
    symbols = np.random.randint(0, 4, num_symbols)
    
    # Optical power levels (example: -6, -2, +2, +6 dBm)
    optical_powers_dbm = np.array([-6, -2, 2, 6])
    optical_powers_linear = 10**(optical_powers_dbm/10)
    
    # Normalize
    optical_powers_linear = optical_powers_linear / np.max(optical_powers_linear)
    
    # Create waveform
    symbol_values = optical_powers_linear[symbols]
    upsampled = np.zeros(num_symbols * sps)
    upsampled[::sps] = symbol_values
    
    # Optical channel effects
    # 1. Bandwidth limitation (photodiode + TIA)
    def apply_lowpass_filter(signal_in, cutoff_hz, sample_rate, order=4):
        """Apply Butterworth lowpass filter without filtfilt"""
        from scipy.signal import butter, lfilter
        fn = sample_rate / 2  # Nyquist frequency
        b, a = butter(order, cutoff_hz/fn, btype='low')
        # Forward-backward filtering
        forward = lfilter(b, a, signal_in)
        backward = lfilter(b, a, forward[::-1])
        return backward[::-1]
    
    bw_3db = 20e9  # 20 GHz bandwidth
    sample_rate = symbol_rate * sps
    waveform = apply_lowpass_filter(upsampled, bw_3db, sample_rate)
    
    # 2. Chromatic dispersion (simplified model)
    dispersion_ps_nm = 10  # 10 ps/nm dispersion
    wavelength_nm = 1310  # 1310 nm wavelength
    fiber_length_km = 10  # 10 km fiber
    
    # 3. Add optical noise (ASE noise)
    osnr_db = 25  # Optical SNR
    noise_power = np.mean(waveform**2) / (10**(osnr_db/10))
    noise = np.sqrt(noise_power) * np.random.randn(len(waveform))
    waveform += noise
    
    # 4. Nonlinear compression (typical of optical modulators)
    waveform = np.tanh(1.5 * waveform) / np.tanh(1.5)
    
    # Calculate TDECQ and RLM
    calculator = PAM4_TDECQ_Calculator(samples_per_symbol=sps, symbol_rate=symbol_rate)
    tdecq, rlm, results = calculator.calculate_tdecq(waveform, num_symbols=10000)
    
    print(f"\nOptical Transmitter Results:")
    print(f"Symbol Rate: {symbol_rate/1e9:.4f} GBaud")
    print(f"TDECQ: {tdecq:.2f} dB")
    print(f"RLM: {rlm:.3f} ({rlm*100:.1f}%)")
    
    # Check compliance
    tdecq_limit = 3.4  # IEEE 802.3bs limit for 50GBASE-SR
    rlm_limit = 0.05   # 5% RLM limit
    
    print(f"\nCompliance Check:")
    print(f"TDECQ: {'PASS' if tdecq <= tdecq_limit else 'FAIL'} (limit: {tdecq_limit} dB)")
    print(f"RLM: {'PASS' if rlm <= rlm_limit else 'FAIL'} (limit: {rlm_limit*100}%)")


if __name__ == "__main__":
    # Run the optical measurement example as well
    example_optical_measurement()