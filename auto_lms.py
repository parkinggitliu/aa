import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz
from scipy.linalg import toeplitz
import time

class NRZPAM4Equalizer:
    """
    NRZ PAM4 Equalizer with Correlation-based and LMS-based adaptation comparison.
    Specifically designed for high-speed serial links (56Gbps, 112Gbps).
    """
    
    def __init__(self, ffe_pre_taps=5, ffe_post_taps=10, dfe_taps=7, 
                 symbol_rate=28e9, oversampling=1):
        """
        Initialize NRZ PAM4 equalizer.
        
        Parameters:
        -----------
        ffe_pre_taps : int
            Number of pre-cursor FFE taps
        ffe_post_taps : int
            Number of post-cursor FFE taps (including main tap)
        dfe_taps : int
            Number of DFE taps
        symbol_rate : float
            Symbol rate in Hz (e.g., 28e9 for 56Gbps PAM4)
        oversampling : int
            Oversampling factor (1 for symbol-spaced)
        """
        self.ffe_pre_taps = ffe_pre_taps
        self.ffe_post_taps = ffe_post_taps
        self.ffe_total_taps = ffe_pre_taps + ffe_post_taps
        self.dfe_taps = dfe_taps
        self.symbol_rate = symbol_rate
        self.oversampling = oversampling
        
        # PAM4 levels: -3, -1, +1, +3 (normalized)
        self.pam4_levels = np.array([-3, -1, 1, 3]) / 3.0
        
        # Gray coding for PAM4
        self.gray_map = {
            (0, 0): -3/3,  # 00 -> -3
            (0, 1): -1/3,  # 01 -> -1
            (1, 1): 1/3,   # 11 -> +1
            (1, 0): 3/3    # 10 -> +3
        }
        
        # Initialize tap coefficients
        self.reset_coefficients()
        
        # Performance metrics storage
        self.metrics = {
            'lms': {'mse': [], 'ber': [], 'ser': []},
            'corr': {'mse': [], 'ber': [], 'ser': []}
        }
        
    def reset_coefficients(self):
        """Reset all tap coefficients to initial values."""
        # FFE initialization (center spike)
        self.ffe_lms = np.zeros(self.ffe_total_taps)
        self.ffe_lms[self.ffe_pre_taps] = 1.0  # Main tap
        
        self.ffe_corr = np.zeros(self.ffe_total_taps)
        self.ffe_corr[self.ffe_pre_taps] = 1.0
        
        # DFE initialization (all zeros)
        self.dfe_lms = np.zeros(self.dfe_taps)
        self.dfe_corr = np.zeros(self.dfe_taps)
        
    def generate_pam4_symbols(self, num_symbols, pattern='random'):
        """
        Generate PAM4 symbols with different patterns.
        
        Parameters:
        -----------
        num_symbols : int
            Number of symbols to generate
        pattern : str
            'random', 'prbs7', 'prbs15', 'stress' (worst-case patterns)
        
        Returns:
        --------
        symbols : array
            PAM4 symbols
        bits : array
            Corresponding bit pairs
        """
        if pattern == 'random':
            bits = np.random.randint(0, 2, (num_symbols, 2))
        elif pattern == 'prbs7':
            # PRBS7: x^7 + x^6 + 1
            bits = self._generate_prbs(num_symbols * 2, [7, 6])
            bits = bits.reshape(num_symbols, 2)
        elif pattern == 'prbs15':
            # PRBS15: x^15 + x^14 + 1
            bits = self._generate_prbs(num_symbols * 2, [15, 14])
            bits = bits.reshape(num_symbols, 2)
        elif pattern == 'stress':
            # Stress pattern with maximum transitions
            stress_pattern = np.array([
                [-3/3, 3/3], [3/3, -3/3],  # Maximum swing
                [-1/3, 1/3], [1/3, -1/3],  # Middle transitions
                [-3/3, -1/3, 1/3, 3/3],    # Ramp up
                [3/3, 1/3, -1/3, -3/3]     # Ramp down
            ]).flatten()
            repeats = num_symbols // len(stress_pattern) + 1
            symbols = np.tile(stress_pattern, repeats)[:num_symbols]
            # Generate corresponding bits (approximate)
            bits = np.zeros((num_symbols, 2), dtype=int)
            return symbols, bits
        
        # Map bits to PAM4 symbols using Gray coding
        symbols = np.zeros(num_symbols)
        for i in range(num_symbols):
            symbols[i] = self.gray_map[tuple(bits[i])]
        
        return symbols, bits
    
    def _generate_prbs(self, length, taps):
        """Generate PRBS sequence."""
        # Initialize with all ones
        lfsr = np.ones(max(taps), dtype=int)
        output = np.zeros(length, dtype=int)
        
        for i in range(length):
            # XOR tapped bits
            feedback = 0
            for tap in taps:
                feedback ^= lfsr[tap-1]
            
            output[i] = lfsr[0]
            # Shift and insert feedback
            lfsr = np.roll(lfsr, -1)
            lfsr[-1] = feedback
            
        return output
    
    def channel_model(self, symbols, channel_type='backplane_28g'):
        """
        Model high-speed channel for NRZ PAM4.
        
        Parameters:
        -----------
        symbols : array
            Input PAM4 symbols
        channel_type : str
            'backplane_28g', 'copper_56g', 'optical_100g'
        
        Returns:
        --------
        output : array
            Channel output with ISI and loss
        channel_response : array
            Channel impulse response
        """
        if channel_type == 'backplane_28g':
            # 28Gbaud backplane channel (20dB loss at Nyquist)
            # Model includes skin effect, dielectric loss
            t = np.arange(0, 20) / self.symbol_rate
            
            # Dominant pole for skin effect
            f_3db = 0.3 * self.symbol_rate
            h_skin = np.exp(-2 * np.pi * f_3db * t)
            
            # Secondary poles for reflections
            h_refl = 0.3 * np.exp(-2 * np.pi * 0.5 * f_3db * t) * \
                     np.cos(2 * np.pi * 0.2 * self.symbol_rate * t)
            
            h = h_skin + h_refl
            h = h[:10]  # Limit channel length
            
        elif channel_type == 'copper_56g':
            # 56Gbaud copper cable (25dB loss)
            t = np.arange(0, 25) / self.symbol_rate
            
            # Severe frequency-dependent loss
            f_3db = 0.2 * self.symbol_rate
            h = np.exp(-2 * np.pi * f_3db * t) * \
                (1 - 0.5 * t * self.symbol_rate)  # Additional linear decay
            h = h[:15]
            
        elif channel_type == 'optical_100g':
            # 100Gbaud optical channel with chromatic dispersion
            beta2 = 20e-27  # Dispersion parameter (s²/m)
            L = 10e3  # 10km fiber
            t = np.arange(-10, 10) / self.symbol_rate
            
            # Chromatic dispersion response
            h = np.sqrt(1 / (1j * beta2 * L)) * \
                np.exp(-t**2 / (2 * beta2 * L))
            h = np.real(h)
            h = h[::2]  # Downsample
            
        # Normalize channel
        h = h / np.sum(h)
        
        # Apply channel
        output = np.convolve(symbols, h, mode='same')
        
        return output, h
    
    def add_noise_and_jitter(self, signal, snr_db=25, rj_rms=0.01):
        """
        Add AWGN and random jitter to signal.
        
        Parameters:
        -----------
        signal : array
            Input signal
        snr_db : float
            Signal-to-noise ratio in dB
        rj_rms : float
            RMS random jitter as fraction of symbol period
        
        Returns:
        --------
        noisy_signal : array
            Signal with noise and jitter
        """
        # Add AWGN
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        
        # Add random jitter (simplified model)
        if rj_rms > 0:
            # Generate jitter samples
            jitter_samples = np.random.normal(0, rj_rms, len(signal))
            # Apply as phase modulation (simplified)
            phase_noise = 2 * np.pi * self.symbol_rate * jitter_samples
            signal = signal * np.exp(1j * phase_noise)
            signal = np.real(signal)
        
        return signal + noise
    
    def lms_adapt(self, rx_signal, training_symbols, mu_ffe=0.01, mu_dfe=0.005,
                  leak_factor=0.9999):
        """
        LMS adaptation with enhancements for PAM4.
        
        Parameters:
        -----------
        rx_signal : array
            Received signal
        training_symbols : array
            Known training symbols
        mu_ffe : float
            FFE adaptation step size
        mu_dfe : float  
            DFE adaptation step size
        leak_factor : float
            Leakage factor for coefficient constraint (0 < leak < 1)
        
        Returns:
        --------
        mse_history : array
            MSE vs iteration
        adapted_signal : array
            Equalized signal output
        """
        N = len(rx_signal)
        mse_history = []
        adapted_signal = np.zeros(N)
        
        # Buffers
        ffe_buffer = np.zeros(self.ffe_total_taps)
        dfe_buffer = np.zeros(self.dfe_taps)
        
        # Enhanced LMS with variable step size
        mu_ffe_var = mu_ffe
        mu_dfe_var = mu_dfe
        
        for n in range(self.ffe_total_taps, N):
            # Fill FFE buffer
            ffe_buffer = rx_signal[n-self.ffe_total_taps+1:n+1][::-1]
            
            # FFE output
            ffe_out = np.dot(self.ffe_lms, ffe_buffer)
            
            # DFE output
            dfe_out = np.dot(self.dfe_lms, dfe_buffer)
            
            # Equalizer output
            eq_out = ffe_out - dfe_out
            adapted_signal[n] = eq_out
            
            # PAM4 slicer
            decision = self.pam4_slicer(eq_out)
            
            # Use training symbols if available
            if n < len(training_symbols):
                target = training_symbols[n]
            else:
                target = decision
            
            # Error calculation
            error = target - eq_out
            mse_history.append(error**2)
            
            # Variable step size based on error magnitude
            if abs(error) > 0.5:  # Large error
                mu_ffe_var = mu_ffe * 2
                mu_dfe_var = mu_dfe * 2
            else:
                mu_ffe_var = mu_ffe
                mu_dfe_var = mu_dfe
            
            # LMS weight update with leakage
            self.ffe_lms = leak_factor * self.ffe_lms + mu_ffe_var * error * ffe_buffer
            self.dfe_lms = leak_factor * self.dfe_lms + mu_dfe_var * error * dfe_buffer
            
            # Constraint: limit DFE tap magnitude to prevent error propagation
            self.dfe_lms = np.clip(self.dfe_lms, -0.5, 0.5)
            
            # Update DFE buffer
            dfe_buffer = np.roll(dfe_buffer, 1)
            dfe_buffer[0] = decision
        
        return np.array(mse_history), adapted_signal
    
    def correlation_adapt(self, rx_signal, training_symbols, regularization=1e-6):
        """
        Correlation-based (MMSE) adaptation for PAM4.
        
        Parameters:
        -----------
        rx_signal : array
            Received signal
        training_symbols : array
            Known training symbols
        regularization : float
            Regularization parameter for matrix inversion
        
        Returns:
        --------
        mse : float
            Mean squared error
        adapted_signal : array
            Equalized signal output
        """
        N_train = len(training_symbols)
        
        # Ensure sufficient training data
        min_samples = 10 * (self.ffe_total_taps + self.dfe_taps)
        if N_train < min_samples:
            print(f"Warning: Training length {N_train} < recommended {min_samples}")
        
        # Construct data matrices
        # FFE matrix
        X_ffe = np.zeros((N_train - self.ffe_total_taps, self.ffe_total_taps))
        for i in range(N_train - self.ffe_total_taps):
            X_ffe[i, :] = rx_signal[i:i+self.ffe_total_taps][::-1]
        
        # DFE matrix (using ideal decisions from training)
        X_dfe = np.zeros((N_train - self.ffe_total_taps, self.dfe_taps))
        for i in range(N_train - self.ffe_total_taps):
            for j in range(self.dfe_taps):
                idx = i + self.ffe_total_taps - 1 - j
                if idx >= 0:
                    X_dfe[i, j] = training_symbols[idx]
        
        # Combined matrix [FFE | -DFE]
        X = np.hstack([X_ffe, -X_dfe])
        
        # Target vector
        d = training_symbols[self.ffe_total_taps:N_train]
        
        # Compute correlation matrices
        R = X.T @ X / X.shape[0]  # Autocorrelation
        p = X.T @ d / X.shape[0]  # Cross-correlation
        
        # Add regularization for numerical stability
        R_reg = R + regularization * np.eye(R.shape[0])
        
        # Solve normal equations: R * w = p
        try:
            w_opt = np.linalg.solve(R_reg, p)
        except np.linalg.LinAlgError:
            print("Matrix singular, using pseudo-inverse")
            w_opt = np.linalg.pinv(R_reg) @ p
        
        # Extract coefficients
        self.ffe_corr = w_opt[:self.ffe_total_taps]
        self.dfe_corr = w_opt[self.ffe_total_taps:]
        
        # Compute MSE on training data
        y_pred = X @ w_opt
        mse = np.mean((d - y_pred)**2)
        
        # Apply equalizer to full signal
        adapted_signal = self.apply_equalizer(rx_signal, self.ffe_corr, self.dfe_corr)
        
        return mse, adapted_signal
    
    def apply_equalizer(self, rx_signal, ffe_taps, dfe_taps):
        """Apply equalizer with given tap coefficients."""
        N = len(rx_signal)
        output = np.zeros(N)
        dfe_buffer = np.zeros(self.dfe_taps)
        
        for n in range(self.ffe_total_taps, N):
            # FFE
            ffe_buffer = rx_signal[n-self.ffe_total_taps+1:n+1][::-1]
            ffe_out = np.dot(ffe_taps, ffe_buffer)
            
            # DFE
            dfe_out = np.dot(dfe_taps, dfe_buffer)
            
            # Output
            output[n] = ffe_out - dfe_out
            
            # Decision for DFE
            decision = self.pam4_slicer(output[n])
            dfe_buffer = np.roll(dfe_buffer, 1)
            dfe_buffer[0] = decision
        
        return output
    
    def pam4_slicer(self, value):
        """PAM4 decision slicer with optimal thresholds."""
        # Slicer thresholds at -2/3, 0, 2/3
        if value < -2/3:
            return -3/3
        elif value < 0:
            return -1/3
        elif value < 2/3:
            return 1/3
        else:
            return 3/3
    
    def calculate_metrics(self, tx_symbols, rx_symbols, tx_bits=None):
        """
        Calculate performance metrics for PAM4.
        
        Parameters:
        -----------
        tx_symbols : array
            Transmitted PAM4 symbols
        rx_symbols : array
            Received/detected PAM4 symbols
        tx_bits : array
            Original bit pairs (optional, for BER calculation)
        
        Returns:
        --------
        metrics : dict
            SER, BER, eye metrics
        """
        # Symbol Error Rate
        symbol_errors = np.sum(tx_symbols != rx_symbols)
        ser = symbol_errors / len(tx_symbols)
        
        # Bit Error Rate (if bits provided)
        ber = 0
        if tx_bits is not None:
            # Convert symbols back to bits
            rx_bits = self.symbols_to_bits(rx_symbols)
            bit_errors = np.sum(tx_bits.flatten() != rx_bits.flatten())
            ber = bit_errors / (len(tx_bits) * 2)
        
        # Eye diagram metrics (simplified)
        eye_height = self.estimate_eye_height(rx_symbols)
        eye_width = self.estimate_eye_width(rx_symbols)
        
        return {
            'ser': ser,
            'ber': ber,
            'eye_height': eye_height,
            'eye_width': eye_width
        }
    
    def symbols_to_bits(self, symbols):
        """Convert PAM4 symbols back to bits using Gray coding."""
        # Inverse Gray mapping
        inv_gray = {v: k for k, v in self.gray_map.items()}
        
        bits = np.zeros((len(symbols), 2), dtype=int)
        for i, sym in enumerate(symbols):
            # Find closest level
            closest_level = self.pam4_levels[np.argmin(np.abs(self.pam4_levels - sym))]
            if closest_level in inv_gray:
                bits[i] = inv_gray[closest_level]
        
        return bits
    
    def estimate_eye_height(self, signal, num_levels=4):
        """Estimate eye height for PAM4 signal."""
        # Simple estimation based on level separation
        hist, bins = np.histogram(signal, bins=100)
        
        # Find peaks (PAM4 levels)
        peaks = []
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append(bins[i])
        
        if len(peaks) >= 2:
            # Minimum eye height is smallest level separation
            eye_heights = np.diff(sorted(peaks))
            return np.min(eye_heights) if len(eye_heights) > 0 else 0
        
        return 0
    
    def estimate_eye_width(self, signal):
        """Estimate eye width (timing margin)."""
        # Simplified: based on transition density
        transitions = np.abs(np.diff(signal)) > 0.5
        transition_rate = np.sum(transitions) / len(transitions)
        
        # Higher transition rate -> smaller eye width
        eye_width = 1.0 - transition_rate
        return eye_width
    
    def compare_adaptation_methods(self, num_symbols=20000, channel_type='backplane_28g',
                                 snr_db=25, training_fraction=0.1):
        """
        Comprehensive comparison of LMS vs Correlation adaptation.
        
        Parameters:
        -----------
        num_symbols : int
            Total number of symbols to simulate
        channel_type : str
            Channel model type
        snr_db : float
            Signal-to-noise ratio
        training_fraction : float
            Fraction of symbols used for training
        
        Returns:
        --------
        results : dict
            Detailed comparison results
        """
        print(f"\n=== PAM4 Adaptation Comparison ===")
        print(f"Channel: {channel_type}, SNR: {snr_db} dB")
        
        # Generate PAM4 symbols
        tx_symbols, tx_bits = self.generate_pam4_symbols(num_symbols, pattern='prbs15')
        
        # Channel simulation
        rx_signal, channel_resp = self.channel_model(tx_symbols, channel_type)
        rx_signal = self.add_noise_and_jitter(rx_signal, snr_db, rj_rms=0.01)
        
        # Training sequence
        num_training = int(num_symbols * training_fraction)
        training_symbols = tx_symbols[:num_training]
        
        # Reset coefficients
        self.reset_coefficients()
        
        # LMS Adaptation
        print("\nRunning LMS adaptation...")
        t_start = time.time()
        mse_lms, eq_signal_lms = self.lms_adapt(rx_signal, training_symbols)
        t_lms = time.time() - t_start
        
        # Detect symbols
        detected_lms = np.array([self.pam4_slicer(x) for x in eq_signal_lms])
        metrics_lms = self.calculate_metrics(tx_symbols, detected_lms, tx_bits)
        
        # Correlation Adaptation
        print("Running correlation adaptation...")
        t_start = time.time()
        mse_corr, eq_signal_corr = self.correlation_adapt(rx_signal, training_symbols)
        t_corr = time.time() - t_start
        
        # Detect symbols
        detected_corr = np.array([self.pam4_slicer(x) for x in eq_signal_corr])
        metrics_corr = self.calculate_metrics(tx_symbols, detected_corr, tx_bits)
        
        # Store results
        results = {
            'channel_response': channel_resp,
            'lms': {
                'mse_history': mse_lms,
                'final_mse': np.mean(mse_lms[-100:]) if len(mse_lms) > 100 else np.mean(mse_lms),
                'metrics': metrics_lms,
                'ffe_taps': self.ffe_lms.copy(),
                'dfe_taps': self.dfe_lms.copy(),
                'eq_signal': eq_signal_lms,
                'time': t_lms
            },
            'corr': {
                'mse': mse_corr,
                'metrics': metrics_corr,
                'ffe_taps': self.ffe_corr.copy(),
                'dfe_taps': self.dfe_corr.copy(),
                'eq_signal': eq_signal_corr,
                'time': t_corr
            },
            'tx_symbols': tx_symbols,
            'rx_signal': rx_signal
        }
        
        # Print summary
        print(f"\n--- Results Summary ---")
        print(f"LMS: SER={metrics_lms['ser']:.6f}, BER={metrics_lms['ber']:.6f}, "
              f"Time={t_lms:.3f}s")
        print(f"Corr: SER={metrics_corr['ser']:.6f}, BER={metrics_corr['ber']:.6f}, "
              f"Time={t_corr:.3f}s")
        
        return results
    
    def plot_detailed_comparison(self, results):
        """Generate comprehensive visualization plots."""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Channel and Equalizer Frequency Response
        ax1 = plt.subplot(3, 3, 1)
        freq = np.fft.fftfreq(1024, 1/self.symbol_rate)
        
        # Channel
        H_ch = np.fft.fft(results['channel_response'], 1024)
        ax1.plot(freq[:512]/1e9, 20*np.log10(np.abs(H_ch[:512])), 'k-', 
                linewidth=2, label='Channel')
        
        # Equalizers
        H_ffe_lms = np.fft.fft(results['lms']['ffe_taps'], 1024)
        H_ffe_corr = np.fft.fft(results['corr']['ffe_taps'], 1024)
        ax1.plot(freq[:512]/1e9, 20*np.log10(np.abs(H_ffe_lms[:512])), 'b--', 
                linewidth=2, label='FFE (LMS)')
        ax1.plot(freq[:512]/1e9, 20*np.log10(np.abs(H_ffe_corr[:512])), 'r--', 
                linewidth=2, label='FFE (Corr)')
        
        ax1.set_xlabel('Frequency (GHz)')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title('Frequency Response')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim([0, self.symbol_rate/2e9])
        
        # 2. LMS Convergence
        ax2 = plt.subplot(3, 3, 2)
        ax2.semilogy(results['lms']['mse_history'][:2000], 'b-', linewidth=1)
        ax2.axhline(y=results['corr']['mse'], color='r', linestyle='--', 
                   linewidth=2, label='Correlation MSE')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('MSE')
        ax2.set_title('LMS Convergence')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. FFE Tap Coefficients
        ax3 = plt.subplot(3, 3, 3)
        tap_indices = np.arange(self.ffe_total_taps)
        width = 0.35
        ax3.bar(tap_indices - width/2, results['lms']['ffe_taps'], width, 
               label='LMS', alpha=0.7)
        ax3.bar(tap_indices + width/2, results['corr']['ffe_taps'], width, 
               label='Correlation', alpha=0.7)
        ax3.axvline(x=self.ffe_pre_taps, color='k', linestyle=':', alpha=0.5)
        ax3.text(self.ffe_pre_taps, ax3.get_ylim()[1]*0.9, 'Main Tap', 
                ha='center', fontsize=9)
        ax3.set_xlabel('Tap Index')
        ax3.set_ylabel('Coefficient Value')
        ax3.set_title('FFE Tap Coefficients')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. DFE Tap Coefficients
        ax4 = plt.subplot(3, 3, 4)
        if self.dfe_taps > 0:
            dfe_indices = np.arange(self.dfe_taps)
            ax4.bar(dfe_indices - width/2, results['lms']['dfe_taps'], width, 
                   label='LMS', alpha=0.7)
            ax4.bar(dfe_indices + width/2, results['corr']['dfe_taps'], width, 
                   label='Correlation', alpha=0.7)
        ax4.set_xlabel('Tap Index')
        ax4.set_ylabel('Coefficient Value')
        ax4.set_title('DFE Tap Coefficients')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Eye Diagram - Before Equalization
        ax5 = plt.subplot(3, 3, 5)
        self.plot_eye_diagram(results['rx_signal'][1000:3000], ax5, 
                            title='Before Equalization')
        
        # 6. Eye Diagram - LMS
        ax6 = plt.subplot(3, 3, 6)
        self.plot_eye_diagram(results['lms']['eq_signal'][1000:3000], ax6, 
                            title='After LMS')
        
        # 7. Eye Diagram - Correlation
        ax7 = plt.subplot(3, 3, 7)
        self.plot_eye_diagram(results['corr']['eq_signal'][1000:3000], ax7, 
                            title='After Correlation')
        
        # 8. Performance Metrics Comparison
        ax8 = plt.subplot(3, 3, 8)
        metrics_names = ['SER', 'BER', 'Eye Height', 'Eye Width']
        lms_values = [
            results['lms']['metrics']['ser'],
            results['lms']['metrics']['ber'],
            results['lms']['metrics']['eye_height'],
            results['lms']['metrics']['eye_width']
        ]
        corr_values = [
            results['corr']['metrics']['ser'],
            results['corr']['metrics']['ber'],
            results['corr']['metrics']['eye_height'],
            results['corr']['metrics']['eye_width']
        ]
        
        x = np.arange(len(metrics_names))
        ax8.bar(x - width/2, lms_values, width, label='LMS', alpha=0.7)
        ax8.bar(x + width/2, corr_values, width, label='Correlation', alpha=0.7)
        ax8.set_ylabel('Value')
        ax8.set_title('Performance Metrics')
        ax8.set_xticks(x)
        ax8.set_xticklabels(metrics_names, rotation=45)
        ax8.legend()
        ax8.set_yscale('log')
        ax8.grid(True, alpha=0.3)
        
        # 9. Adaptation Time and Complexity
        ax9 = plt.subplot(3, 3, 9)
        categories = ['Time (s)', 'Complexity\n(Relative)', 'Memory\n(Relative)']
        lms_vals = [results['lms']['time'], 1, 1]  # Normalized values
        corr_vals = [results['corr']['time'], 5, 3]  # Relative complexity
        
        x = np.arange(len(categories))
        ax9.bar(x - width/2, lms_vals, width, label='LMS', alpha=0.7)
        ax9.bar(x + width/2, corr_vals, width, label='Correlation', alpha=0.7)
        ax9.set_ylabel('Value')
        ax9.set_title('Implementation Comparison')
        ax9.set_xticks(x)
        ax9.set_xticklabels(categories)
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_eye_diagram(self, signal, ax, title='Eye Diagram', 
                        num_symbols_per_trace=2):
        """Plot PAM4 eye diagram."""
        # Reshape signal for eye diagram
        samples_per_symbol = 1  # Symbol-spaced
        trace_length = num_symbols_per_trace * samples_per_symbol
        
        num_traces = len(signal) // trace_length
        
        # Plot overlapped traces
        for i in range(min(num_traces, 200)):  # Limit traces for clarity
            trace = signal[i*trace_length:(i+1)*trace_length]
            t = np.linspace(0, num_symbols_per_trace, len(trace))
            ax.plot(t, trace, 'b-', alpha=0.1, linewidth=0.5)
        
        # Add PAM4 decision levels
        for level in self.pam4_levels:
            ax.axhline(y=level, color='r', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Symbol Period')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, num_symbols_per_trace])
        ax.set_ylim([-1.5, 1.5])
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis with multiple scenarios."""
        print("\n" + "="*60)
        print("NRZ PAM4 DFE/FFE: Correlation vs LMS Adaptation Analysis")
        print("="*60)
        
        # Test scenarios
        scenarios = [
            {'name': '28Gbaud Backplane', 'channel': 'backplane_28g', 'snr': 25},
            {'name': '56Gbaud Copper', 'channel': 'copper_56g', 'snr': 20},
            {'name': '100Gbaud Optical', 'channel': 'optical_100g', 'snr': 30}
        ]
        
        all_results = {}
        
        for scenario in scenarios:
            print(f"\n### Testing: {scenario['name']} ###")
            
            results = self.compare_adaptation_methods(
                num_symbols=10000,
                channel_type=scenario['channel'],
                snr_db=scenario['snr'],
                training_fraction=0.1
            )
            
            all_results[scenario['name']] = results
            
            # Plot detailed comparison
            self.plot_detailed_comparison(results)
        
        # SNR sweep analysis
        self.snr_sweep_analysis()
        
        # Training length analysis
        self.training_length_analysis()
        
        return all_results
    
    def snr_sweep_analysis(self):
        """Analyze performance vs SNR."""
        print("\n### SNR Sweep Analysis ###")
        
        snr_range = np.arange(15, 35, 2)
        ser_lms = []
        ser_corr = []
        ber_lms = []
        ber_corr = []
        
        for snr in snr_range:
            results = self.compare_adaptation_methods(
                num_symbols=5000,
                channel_type='backplane_28g',
                snr_db=snr,
                training_fraction=0.1
            )
            
            ser_lms.append(results['lms']['metrics']['ser'])
            ser_corr.append(results['corr']['metrics']['ser'])
            ber_lms.append(results['lms']['metrics']['ber'])
            ber_corr.append(results['corr']['metrics']['ber'])
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # SER plot
        ax1.semilogy(snr_range, ser_lms, 'bo-', label='LMS', linewidth=2, markersize=8)
        ax1.semilogy(snr_range, ser_corr, 'rs-', label='Correlation', linewidth=2, markersize=8)
        ax1.set_xlabel('SNR (dB)')
        ax1.set_ylabel('Symbol Error Rate (SER)')
        ax1.set_title('PAM4 SER vs SNR')
        ax1.grid(True, which="both", ls="-", alpha=0.3)
        ax1.legend()
        
        # BER plot
        ax2.semilogy(snr_range, ber_lms, 'bo-', label='LMS', linewidth=2, markersize=8)
        ax2.semilogy(snr_range, ber_corr, 'rs-', label='Correlation', linewidth=2, markersize=8)
        ax2.set_xlabel('SNR (dB)')
        ax2.set_ylabel('Bit Error Rate (BER)')
        ax2.set_title('PAM4 BER vs SNR')
        ax2.grid(True, which="both", ls="-", alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def training_length_analysis(self):
        """Analyze impact of training length."""
        print("\n### Training Length Analysis ###")
        
        training_fractions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
        total_symbols = 10000
        
        ser_lms = []
        ser_corr = []
        convergence_time_lms = []
        
        for frac in training_fractions:
            results = self.compare_adaptation_methods(
                num_symbols=total_symbols,
                channel_type='backplane_28g',
                snr_db=25,
                training_fraction=frac
            )
            
            ser_lms.append(results['lms']['metrics']['ser'])
            ser_corr.append(results['corr']['metrics']['ser'])
            
            # Find convergence point for LMS (within 10% of final MSE)
            final_mse = results['lms']['final_mse']
            mse_history = results['lms']['mse_history']
            conv_idx = np.where(mse_history < 1.1 * final_mse)[0]
            conv_time = conv_idx[0] if len(conv_idx) > 0 else len(mse_history)
            convergence_time_lms.append(conv_time)
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        training_symbols = [int(frac * total_symbols) for frac in training_fractions]
        
        # SER vs training length
        ax1.semilogy(training_symbols, ser_lms, 'bo-', label='LMS', 
                    linewidth=2, markersize=8)
        ax1.semilogy(training_symbols, ser_corr, 'rs-', label='Correlation', 
                    linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Training Symbols')
        ax1.set_ylabel('Symbol Error Rate (SER)')
        ax1.set_title('Impact of Training Length')
        ax1.grid(True, which="both", ls="-", alpha=0.3)
        ax1.legend()
        
        # Convergence time
        ax2.plot(training_symbols, convergence_time_lms, 'bo-', 
                linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Training Symbols')
        ax2.set_ylabel('LMS Convergence Time (symbols)')
        ax2.set_title('LMS Convergence vs Training Length')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary_table(self):
        """Print comprehensive comparison summary."""
        print("\n" + "="*80)
        print("PAM4 DFE/FFE ADAPTATION METHOD COMPARISON SUMMARY")
        print("="*80)
        
        print("\n┌─────────────────────────┬─────────────────────────┬─────────────────────────┐")
        print("│ Characteristic          │ LMS Adaptation          │ Correlation Adaptation  │")
        print("├─────────────────────────┼─────────────────────────┼─────────────────────────┤")
        print("│ Algorithm Type          │ Stochastic Gradient     │ Block Processing (MMSE) │")
        print("│ Computational Order     │ O(N) per symbol         │ O(N²) matrix ops        │")
        print("│ Memory Requirement      │ 2×(FFE+DFE) taps        │ (FFE+DFE)² matrix       │")
        print("│ Convergence Speed       │ Slow (iterative)        │ Immediate (one-shot)    │")
        print("│ Tracking Capability     │ Excellent               │ Poor (static only)      │")
        print("│ Noise Sensitivity       │ Moderate                │ High (regularization)   │")
        print("│ Training Data Required  │ Minimal (100s symbols)  │ Substantial (1000s)     │")
        print("│ Hardware Implementation │ Simple MAC units        │ Complex matrix solver   │")
        print("│ Power Consumption       │ Low                     │ High                    │")
        print("│ Adaptation Time         │ Continuous              │ Batch                   │")
        print("└─────────────────────────┴─────────────────────────┴─────────────────────────┘")
        
        print("\n┌─────────────────────────┬─────────────────────────┬─────────────────────────┐")
        print("│ PAM4-Specific Features  │ LMS                     │ Correlation             │")
        print("├─────────────────────────┼─────────────────────────┼─────────────────────────┤")
        print("│ Multi-level Support     │ Native with PAM4 slicer │ Requires all levels     │")
        print("│ Error Propagation       │ Can adapt/recover       │ Fixed after training    │")
        print("│ Level-dependent Noise   │ Adapts automatically    │ Assumes uniform         │")
        print("│ Pattern Dependencies    │ Handles naturally       │ Training must cover all │")
        print("│ Jitter Tolerance        │ Good (continuous adapt) │ Poor (static solution)  │")
        print("└─────────────────────────┴─────────────────────────┴─────────────────────────┘")
        
        print("\n### Recommendations for NRZ PAM4 Systems ###")
        print("1. LMS: Best for production systems with time-varying channels")
        print("2. Correlation: Suitable for static channels with good training sequences")
        print("3. Hybrid: Use correlation for initialization, then switch to LMS")
        print("4. Consider variable step-size LMS for faster initial convergence")
        print("5. Implement DFE tap limiting to prevent error propagation in PAM4")


# Main execution
if __name__ == "__main__":
    # Create PAM4 equalizer instance
    equalizer = NRZPAM4Equalizer(
        ffe_pre_taps=5,
        ffe_post_taps=10,
        dfe_taps=5,
        symbol_rate=28e9  # 28 Gbaud
    )
    
    # Run comprehensive analysis
    results = equalizer.run_comprehensive_analysis()
    
    # Print summary table
    equalizer.print_summary_table()