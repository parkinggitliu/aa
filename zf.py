import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, solve_toeplitz
from scipy.signal import freqz, group_delay
from scipy.fft import fft, fftfreq, ifft
import time

class ToeplitzZeroForcingFFE:
    """
    Zero Forcing Feed-Forward Equalizer using Toeplitz matrix methods.
    Optimized for high-speed serial links with ISI compensation.
    """
    
    def __init__(self, num_taps=21, tap_spacing=1, symbol_rate=25e9):
        """
        Initialize ZF-FFE.
        
        Parameters:
        -----------
        num_taps : int
            Number of FFE taps (should be odd for symmetric design)
        tap_spacing : int
            Tap spacing (1 for symbol-spaced, 2 for half-symbol-spaced)
        symbol_rate : float
            Symbol rate in Hz
        """
        self.num_taps = num_taps
        self.tap_spacing = tap_spacing
        self.symbol_rate = symbol_rate
        self.Ts = 1 / symbol_rate  # Symbol period
        
        # Ensure odd number of taps for center tap
        if num_taps % 2 == 0:
            self.num_taps += 1
            print(f"Adjusted to {self.num_taps} taps (odd number required)")
        
        self.center_tap = num_taps // 2
        
        # FFE coefficients
        self.coefficients = np.zeros(num_taps)
        self.coefficients[self.center_tap] = 1.0  # Initialize with center spike
        
        # Performance metrics
        self.metrics = {
            'mse': [],
            'peak_distortion': [],
            'eye_opening': []
        }
        
    def compute_channel_matrix(self, channel_response, num_symbols):
        """
        Construct Toeplitz channel matrix from impulse response.
        
        Parameters:
        -----------
        channel_response : array
            Channel impulse response
        num_symbols : int
            Number of symbols for matrix construction
        
        Returns:
        --------
        H : array
            Channel matrix (Toeplitz structure)
        """
        h = np.array(channel_response)
        L = len(h)
        
        # Construct first column and row
        # First column: [h[0], h[1], ..., h[L-1], 0, 0, ...]
        col = np.zeros(num_symbols)
        col[:L] = h
        
        # First row: [h[0], 0, 0, ...]
        row = np.zeros(num_symbols)
        row[0] = h[0]
        
        # Create Toeplitz matrix
        H = toeplitz(col, row)
        
        return H
    
    def design_zf_equalizer(self, channel_response, desired_delay=None, 
                           regularization=1e-6):
        """
        Design Zero Forcing equalizer using Toeplitz matrix inversion.
        
        Parameters:
        -----------
        channel_response : array
            Channel impulse response
        desired_delay : int
            Desired system delay (None for automatic)
        regularization : float
            Regularization parameter for numerical stability
        
        Returns:
        --------
        coefficients : array
            Optimized FFE tap coefficients
        """
        h = np.array(channel_response)
        n_h = len(h)
        n_w = self.num_taps
        
        # Set desired delay (typically center tap)
        if desired_delay is None:
            desired_delay = self.center_tap + n_h // 2
        
        # Construct convolution matrix (Toeplitz structure)
        # Size: (n_h + n_w - 1) x n_w
        n_y = n_h + n_w - 1
        
        # Build Toeplitz matrix for convolution
        # First column
        col = np.zeros(n_y)
        col[:n_h] = h
        
        # First row  
        row = np.zeros(n_w)
        row[0] = h[0]
        
        # Channel convolution matrix
        H = toeplitz(col, row)
        
        # Desired response vector (unit impulse at desired delay)
        d = np.zeros(n_y)
        if desired_delay < n_y:
            d[desired_delay] = 1.0
        else:
            print(f"Warning: desired delay {desired_delay} exceeds output length")
            d[n_y//2] = 1.0
        
        # Solve normal equations: H^T H w = H^T d
        # Using regularization for stability
        HTH = H.T @ H + regularization * np.eye(n_w)
        HTd = H.T @ d
        
        # Solve using optimized Toeplitz solver if possible
        if self._is_toeplitz_system(HTH):
            # Use fast Toeplitz solver
            self.coefficients = self._solve_toeplitz_system(HTH, HTd)
        else:
            # Standard linear solver
            self.coefficients = np.linalg.solve(HTH, HTd)
        
        # Normalize to prevent overflow
        max_coeff = np.max(np.abs(self.coefficients))
        if max_coeff > 0:
            self.coefficients /= max_coeff
        
        return self.coefficients
    
    def _is_toeplitz_system(self, matrix, tol=1e-10):
        """Check if matrix is Toeplitz."""
        n = matrix.shape[0]
        for i in range(1, n):
            for j in range(1, n):
                if abs(matrix[i, j] - matrix[i-1, j-1]) > tol:
                    return False
        return True
    
    def _solve_toeplitz_system(self, T, b):
        """Solve Toeplitz system using Levinson recursion."""
        n = len(b)
        
        # Extract first column and row
        c = T[:, 0]
        r = T[0, :]
        
        try:
            # Use scipy's optimized Toeplitz solver
            x = solve_toeplitz((c, r), b)
            return x
        except:
            # Fallback to standard solver
            return np.linalg.solve(T, b)
    
    def design_mmse_equalizer(self, channel_response, noise_variance=0.01,
                            signal_power=1.0):
        """
        Design MMSE equalizer (includes noise in optimization).
        
        Parameters:
        -----------
        channel_response : array
            Channel impulse response
        noise_variance : float
            Noise variance (σ²)
        signal_power : float
            Signal power
        
        Returns:
        --------
        coefficients : array
            MMSE-optimized FFE coefficients
        """
        h = np.array(channel_response)
        n_h = len(h)
        n_w = self.num_taps
        
        # Construct channel matrix
        n_y = n_h + n_w - 1
        col = np.zeros(n_y)
        col[:n_h] = h
        row = np.zeros(n_w)
        row[0] = h[0]
        H = toeplitz(col, row)
        
        # MMSE solution: (H^T H + (σ²/P_s) I)^(-1) H^T d
        regularization = noise_variance / signal_power
        HTH = H.T @ H + regularization * np.eye(n_w)
        
        # Desired response
        d = np.zeros(n_y)
        d[self.center_tap + n_h // 2] = 1.0
        HTd = H.T @ d
        
        # Solve
        self.coefficients = np.linalg.solve(HTH, HTd)
        
        # Normalize
        self.coefficients /= np.max(np.abs(self.coefficients))
        
        return self.coefficients
    
    def apply_equalizer(self, signal):
        """
        Apply FFE to input signal.
        
        Parameters:
        -----------
        signal : array
            Input signal
        
        Returns:
        --------
        equalized : array
            Equalized output signal
        """
        # Apply FFE using convolution
        equalized = np.convolve(signal, self.coefficients, mode='same')
        return equalized
    
    def compute_frequency_response(self, num_points=1024):
        """
        Compute frequency response of the equalizer.
        
        Returns:
        --------
        freq : array
            Normalized frequency points
        H_eq : array
            Complex frequency response
        mag_db : array
            Magnitude response in dB
        phase : array
            Phase response in radians
        gd : array
            Group delay in samples
        """
        # Frequency response
        w, H_eq = freqz(self.coefficients, worN=num_points)
        freq = w / (2 * np.pi)  # Normalized frequency
        
        # Magnitude and phase
        mag_db = 20 * np.log10(np.abs(H_eq) + 1e-12)
        phase = np.unwrap(np.angle(H_eq))
        
        # Group delay
        _, gd = group_delay((self.coefficients, 1), w=w)
        
        return freq, H_eq, mag_db, phase, gd
    
    def optimize_tap_weights(self, channel_response, training_sequence,
                           method='gradient', iterations=100, mu=0.01):
        """
        Optimize tap weights using iterative methods.
        
        Parameters:
        -----------
        channel_response : array
            Channel impulse response
        training_sequence : array
            Known training symbols
        method : str
            Optimization method ('gradient', 'newton', 'conjugate')
        iterations : int
            Number of iterations
        mu : float
            Step size for gradient methods
        
        Returns:
        --------
        coefficients : array
            Optimized coefficients
        convergence : array
            MSE vs iteration
        """
        # Channel output for training sequence
        channel_output = np.convolve(training_sequence, channel_response, mode='same')
        
        # Add noise
        noise = np.random.normal(0, 0.01, len(channel_output))
        received = channel_output + noise
        
        # Initialize
        w = self.coefficients.copy()
        convergence = []
        
        if method == 'gradient':
            # Gradient descent optimization
            for i in range(iterations):
                # Forward pass
                equalized = np.convolve(received, w, mode='same')
                
                # Error
                error = training_sequence - equalized
                mse = np.mean(error**2)
                convergence.append(mse)
                
                # Gradient computation
                gradient = np.zeros_like(w)
                for k in range(len(w)):
                    if k < len(received):
                        gradient[k] = -2 * np.mean(error * np.roll(received, k)[:len(error)])
                
                # Update weights
                w = w - mu * gradient
                
                # Adaptive step size
                if i > 0 and convergence[i] > convergence[i-1]:
                    mu *= 0.5  # Reduce step size if error increases
                    
        elif method == 'newton':
            # Newton's method (second-order)
            for i in range(iterations):
                equalized = np.convolve(received, w, mode='same')
                error = training_sequence - equalized
                mse = np.mean(error**2)
                convergence.append(mse)
                
                # Hessian approximation
                R = self._compute_autocorrelation_matrix(received, len(w))
                p = self._compute_crosscorrelation(received, training_sequence, len(w))
                
                # Newton update: w = w + inv(R) * (p - R*w)
                try:
                    delta = np.linalg.solve(R + 0.01 * np.eye(len(w)), p - R @ w)
                    w = w + 0.5 * delta  # Damped Newton
                except:
                    # Fall back to gradient if singular
                    gradient = -2 * (p - R @ w)
                    w = w - mu * gradient
                    
        elif method == 'conjugate':
            # Conjugate gradient method
            R = self._compute_autocorrelation_matrix(received, len(w))
            p = self._compute_crosscorrelation(received, training_sequence, len(w))
            
            # CG initialization
            r = p - R @ w
            d = r.copy()
            
            for i in range(min(iterations, len(w))):
                equalized = np.convolve(received, w, mode='same')
                error = training_sequence - equalized
                mse = np.mean(error**2)
                convergence.append(mse)
                
                # CG iterations
                Rd = R @ d
                alpha = (r.T @ r) / (d.T @ Rd + 1e-10)
                w = w + alpha * d
                r_new = r - alpha * Rd
                
                if np.linalg.norm(r_new) < 1e-10:
                    break
                    
                beta = (r_new.T @ r_new) / (r.T @ r + 1e-10)
                d = r_new + beta * d
                r = r_new
        
        self.coefficients = w / np.max(np.abs(w))  # Normalize
        return self.coefficients, np.array(convergence)
    
    def _compute_autocorrelation_matrix(self, signal, size):
        """Compute autocorrelation matrix of signal."""
        R = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i - j >= 0 and i - j < len(signal):
                    R[i, j] = np.mean(signal[i-j:] * signal[:len(signal)-(i-j)])
        return R
    
    def _compute_crosscorrelation(self, signal, desired, size):
        """Compute cross-correlation vector."""
        p = np.zeros(size)
        for i in range(size):
            if i < len(signal) and i < len(desired):
                p[i] = np.mean(signal[i:] * desired[:len(signal)-i])
        return p
    
    def analyze_equalization_performance(self, channel_response, 
                                       test_signal=None, num_symbols=1000):
        """
        Analyze equalizer performance with various metrics.
        
        Parameters:
        -----------
        channel_response : array
            Channel impulse response
        test_signal : array
            Test signal (None for random)
        num_symbols : int
            Number of symbols for testing
        
        Returns:
        --------
        metrics : dict
            Performance metrics
        """
        if test_signal is None:
            # Generate random PAM4 test signal
            test_signal = np.random.choice([-3, -1, 1, 3], num_symbols)
        
        # Channel output
        channel_out = np.convolve(test_signal, channel_response, mode='same')
        
        # Add realistic noise
        snr_db = 25
        signal_power = np.mean(channel_out**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.sqrt(noise_power) * np.random.randn(len(channel_out))
        received = channel_out + noise
        
        # Apply equalizer
        equalized = self.apply_equalizer(received)
        
        # Compute metrics
        # 1. Residual ISI
        impulse_response = np.convolve(channel_response, self.coefficients)
        peak_idx = np.argmax(np.abs(impulse_response))
        main_tap = impulse_response[peak_idx]
        isi_taps = np.concatenate([impulse_response[:peak_idx], 
                                  impulse_response[peak_idx+1:]])
        residual_isi = np.sum(np.abs(isi_taps)) / np.abs(main_tap)
        
        # 2. Eye opening
        eye_opening = self._estimate_eye_opening(equalized, num_symbols=min(1000, len(equalized)))
        
        # 3. Peak distortion
        peak_distortion = np.max(np.abs(isi_taps)) / np.abs(main_tap)
        
        # 4. Noise enhancement
        noise_enhancement = np.sqrt(np.sum(self.coefficients**2))
        
        # 5. Frequency response flatness
        freq, H_eq, mag_db, _, _ = self.compute_frequency_response()
        passband_idx = freq < 0.7  # 70% of Nyquist
        flatness = np.std(mag_db[passband_idx])
        
        metrics = {
            'residual_isi': residual_isi,
            'eye_opening': eye_opening,
            'peak_distortion': peak_distortion,
            'noise_enhancement': noise_enhancement,
            'frequency_flatness_db': flatness,
            'combined_response': impulse_response,
            'main_tap_value': main_tap,
            'effective_snr_db': snr_db - 10*np.log10(noise_enhancement**2)
        }
        
        return metrics
    
    def _estimate_eye_opening(self, signal, num_symbols=1000):
        """Estimate eye opening from equalized signal."""
        # Simple eye opening estimation
        # In practice, would use eye diagram analysis
        samples_per_symbol = len(signal) // num_symbols
        if samples_per_symbol < 1:
            return 0
        
        # Sample at optimal point (center of eye)
        sampled = signal[samples_per_symbol//2::samples_per_symbol][:num_symbols]
        
        # Find levels (assuming PAM4)
        levels = [-3, -1, 1, 3]
        
        # Classify samples to nearest level
        classified = np.zeros_like(sampled)
        for i, samp in enumerate(sampled):
            classified[i] = min(levels, key=lambda x: abs(x - samp))
        
        # Compute eye opening as minimum distance between levels
        eye_heights = []
        for i in range(len(levels)-1):
            level_samples_low = sampled[classified == levels[i]]
            level_samples_high = sampled[classified == levels[i+1]]
            
            if len(level_samples_low) > 0 and len(level_samples_high) > 0:
                eye_height = (np.mean(level_samples_high) - np.mean(level_samples_low) -
                            np.std(level_samples_high) - np.std(level_samples_low))
                eye_heights.append(eye_height)
        
        return np.min(eye_heights) if eye_heights else 0
    
    def plot_results(self, channel_response=None, show_all=True):
        """
        Plot comprehensive equalization results.
        
        Parameters:
        -----------
        channel_response : array
            Channel impulse response for combined response plot
        show_all : bool
            Show all plots
        """
        if show_all:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 1. FFE tap coefficients
            ax = axes[0, 0]
            tap_indices = np.arange(self.num_taps) - self.center_tap
            ax.stem(tap_indices, self.coefficients, basefmt=' ')
            ax.set_xlabel('Tap Index')
            ax.set_ylabel('Coefficient Value')
            ax.set_title('FFE Tap Coefficients')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            
            # 2. Frequency response
            ax = axes[0, 1]
            freq, _, mag_db, _, _ = self.compute_frequency_response()
            ax.plot(freq, mag_db, 'b-', linewidth=2)
            ax.set_xlabel('Normalized Frequency')
            ax.set_ylabel('Magnitude (dB)')
            ax.set_title('FFE Frequency Response')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 0.5])
            
            # 3. Group delay
            ax = axes[0, 2]
            freq, _, _, _, gd = self.compute_frequency_response()
            ax.plot(freq, gd, 'g-', linewidth=2)
            ax.set_xlabel('Normalized Frequency')
            ax.set_ylabel('Group Delay (samples)')
            ax.set_title('FFE Group Delay')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 0.5])
            
            # 4. Combined impulse response
            if channel_response is not None:
                ax = axes[1, 0]
                combined = np.convolve(channel_response, self.coefficients)
                time_indices = np.arange(len(combined)) - len(combined)//2
                ax.stem(time_indices, combined, basefmt=' ')
                ax.set_xlabel('Time (symbols)')
                ax.set_ylabel('Amplitude')
                ax.set_title('Combined Channel + FFE Response')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linewidth=0.5)
                
                # Mark main tap and ISI
                peak_idx = np.argmax(np.abs(combined))
                ax.axvline(x=time_indices[peak_idx], color='g', linestyle='--', 
                          alpha=0.7, label='Main tap')
                ax.legend()
            
            # 5. Convergence plot (if available)
            if hasattr(self, 'convergence_history'):
                ax = axes[1, 1]
                ax.semilogy(self.convergence_history, 'b-', linewidth=2)
                ax.set_xlabel('Iteration')
                ax.set_ylabel('MSE')
                ax.set_title('Optimization Convergence')
                ax.grid(True, alpha=0.3)
            
            # 6. Performance metrics
            if channel_response is not None:
                ax = axes[1, 2]
                metrics = self.analyze_equalization_performance(channel_response)
                
                metric_names = ['Residual ISI', 'Peak Distortion', 
                              'Noise Enhancement', 'Freq Flatness (dB)']
                metric_values = [metrics['residual_isi'], metrics['peak_distortion'],
                               metrics['noise_enhancement'], metrics['frequency_flatness_db']]
                
                y_pos = np.arange(len(metric_names))
                bars = ax.barh(y_pos, metric_values, alpha=0.7)
                
                # Color code bars
                for i, bar in enumerate(bars):
                    if metric_values[i] < 0.1:
                        bar.set_color('green')
                    elif metric_values[i] < 0.3:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(metric_names)
                ax.set_xlabel('Value')
                ax.set_title('Performance Metrics')
                ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.show()


# Example usage and demonstrations
if __name__ == "__main__":
    print("=== Toeplitz Zero Forcing FFE Demo ===\n")
    
    # Create FFE instance
    ffe = ToeplitzZeroForcingFFE(num_taps=21, symbol_rate=25e9)
    
    # Example 1: Simple channel with ISI
    print("Example 1: Simple ISI Channel")
    print("-" * 40)
    
    # Channel impulse response (normalized)
    channel_simple = np.array([0.1, 0.3, 0.8, 1.0, 0.7, 0.4, 0.2, 0.1])
    channel_simple = channel_simple / np.max(channel_simple)
    
    # Design equalizer
    coeffs_zf = ffe.design_zf_equalizer(channel_simple)
    print(f"Number of taps: {len(coeffs_zf)}")
    print(f"Main tap location: {ffe.center_tap}")
    
    # Analyze performance
    metrics = ffe.analyze_equalization_performance(channel_simple)
    print(f"Residual ISI: {metrics['residual_isi']:.4f}")
    print(f"Peak distortion: {metrics['peak_distortion']:.4f}")
    print(f"Noise enhancement: {metrics['noise_enhancement']:.4f}")
    print(f"Effective SNR loss: {25 - metrics['effective_snr_db']:.2f} dB\n")
    
    # Example 2: Severe ISI channel
    print("Example 2: Severe ISI Channel")
    print("-" * 40)
    
    # More challenging channel
    t = np.linspace(0, 10, 15)
    channel_severe = np.exp(-t/3) * np.sin(2*np.pi*0.3*t)
    channel_severe = channel_severe / np.max(np.abs(channel_severe))
    
    # Compare ZF and MMSE
    coeffs_zf = ffe.design_zf_equalizer(channel_severe, regularization=1e-6)
    metrics_zf = ffe.analyze_equalization_performance(channel_severe)
    
    coeffs_mmse = ffe.design_mmse_equalizer(channel_severe, noise_variance=0.01)
    metrics_mmse = ffe.analyze_equalization_performance(channel_severe)
    
    print("Zero Forcing:")
    print(f"  Residual ISI: {metrics_zf['residual_isi']:.4f}")
    print(f"  Noise enhancement: {metrics_zf['noise_enhancement']:.4f}")
    
    print("\nMMSE:")
    print(f"  Residual ISI: {metrics_mmse['residual_isi']:.4f}") 
    print(f"  Noise enhancement: {metrics_mmse['noise_enhancement']:.4f}\n")
    
    # Example 3: Optimization comparison
    print("Example 3: Optimization Methods Comparison")
    print("-" * 40)
    
    # Generate training sequence
    np.random.seed(42)
    training_seq = np.random.choice([-3, -1, 1, 3], 1000)
    
    # Test different optimization methods
    methods = ['gradient', 'newton', 'conjugate']
    results = {}
    
    for method in methods:
        ffe_opt = ToeplitzZeroForcingFFE(num_taps=21)
        start_time = time.time()
        
        coeffs, convergence = ffe_opt.optimize_tap_weights(
            channel_severe, training_seq, method=method, iterations=50
        )
        
        elapsed = time.time() - start_time
        final_mse = convergence[-1] if len(convergence) > 0 else np.inf
        
        results[method] = {
            'time': elapsed,
            'final_mse': final_mse,
            'iterations': len(convergence),
            'convergence': convergence
        }
        
        print(f"{method.capitalize()}:")
        print(f"  Time: {elapsed:.3f} seconds")
        print(f"  Final MSE: {final_mse:.6f}")
        print(f"  Iterations: {len(convergence)}")
    
    # Plot convergence comparison
    plt.figure(figsize=(10, 6))
    for method in methods:
        plt.semilogy(results[method]['convergence'], label=method.capitalize(), 
                    linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('Optimization Method Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Example 4: Tap spacing analysis
    print("\nExample 4: Tap Spacing Analysis")
    print("-" * 40)
    
    tap_spacings = [1, 2, 4]  # Symbol-spaced, half-symbol, quarter-symbol
    
    for spacing in tap_spacings:
        ffe_spaced = ToeplitzZeroForcingFFE(num_taps=21//spacing, tap_spacing=spacing)
        coeffs = ffe_spaced.design_zf_equalizer(channel_severe)
        metrics = ffe_spaced.analyze_equalization_performance(channel_severe)
        
        print(f"Tap spacing T/{spacing}:")
        print(f"  Residual ISI: {metrics['residual_isi']:.4f}")
        print(f"  Noise enhancement: {metrics['noise_enhancement']:.4f}")
    
    # Plot final results
    print("\nPlotting comprehensive results...")
    ffe.plot_results(channel_severe, show_all=True)
    
    # Performance summary
    print("\n=== Performance Summary ===")
    print("1. Zero Forcing provides perfect ISI cancellation at Nyquist frequency")
    print("2. MMSE trades off ISI cancellation for noise enhancement")
    print("3. Toeplitz structure enables efficient computation")
    print("4. Fractional tap spacing improves performance for band-limited channels")
    print("5. Iterative optimization can fine-tune coefficients for non-ideal conditions")