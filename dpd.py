import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class PowerAmplifierModel:
    """
    Power Amplifier model with memory effects
    Implements Volterra series and memory polynomial models
    """
    
    def __init__(self, memory_depth=5, nonlinearity_order=9, saturation_level=1.0):
        """
        Initialize PA model
        
        Parameters:
        - memory_depth: Number of memory taps
        - nonlinearity_order: Maximum nonlinearity order (odd numbers)
        - saturation_level: PA saturation level
        """
        self.M = memory_depth
        self.K = nonlinearity_order if nonlinearity_order % 2 == 1 else nonlinearity_order + 1
        self.sat_level = saturation_level
        
        # Memory polynomial coefficients
        self.coefficients = self._generate_pa_coefficients()
        
        # Thermal memory effects
        self.thermal_tau = 100e-6  # Thermal time constant (100 μs)
        self.thermal_gain = 0.1    # Thermal memory strength
        
    def _generate_pa_coefficients(self):
        """Generate realistic PA coefficients with memory"""
        coeffs = {}
        
        # Linear term (fundamental)
        coeffs[(1, 0)] = 1.0 + 0.05j
        
        # Nonlinear terms with memory
        for k in range(1, self.K + 1, 2):  # Odd orders only
            for m in range(self.M):
                if k == 1:
                    # Linear terms with memory
                    coeffs[(k, m)] = (0.8 - 0.1*m) * np.exp(-1j * 0.1 * m)
                elif k == 3:
                    # 3rd order terms
                    coeffs[(k, m)] = (-0.2 + 0.02*m) * np.exp(-1j * 0.2 * m)
                elif k == 5:
                    # 5th order terms  
                    coeffs[(k, m)] = (0.05 - 0.005*m) * np.exp(-1j * 0.3 * m)
                else:
                    # Higher order terms
                    coeffs[(k, m)] = (0.01 / k) * np.exp(-1j * 0.1 * k * m)
                    
        return coeffs
    
    def memory_polynomial(self, x):
        """
        Memory polynomial PA model
        y(n) = Σ_k Σ_m a_{k,m} * x(n-m) * |x(n-m)|^{k-1}
        """
        N = len(x)
        y = np.zeros(N, dtype=complex)
        
        # Pad input for memory
        x_padded = np.concatenate([np.zeros(self.M-1), x])
        
        for n in range(N):
            for k in range(1, self.K + 1, 2):  # Odd orders
                for m in range(self.M):
                    if (k, m) in self.coefficients:
                        x_delayed = x_padded[n + self.M - 1 - m]
                        y[n] += self.coefficients[(k, m)] * x_delayed * (np.abs(x_delayed) ** (k-1))
        
        return y
    
    def thermal_memory_effect(self, x, fs):
        """Add thermal memory effects"""
        # Low-pass filter to model thermal dynamics
        tau_samples = int(self.thermal_tau * fs)
        thermal_filter = signal.butter(1, 1/tau_samples, btype='low', output='sos')
        
        # Envelope detection
        envelope = np.abs(x)
        
        # Apply thermal filtering
        thermal_response = signal.sosfilt(thermal_filter, envelope)
        
        # Thermal gain compression
        thermal_effect = 1 - self.thermal_gain * thermal_response / np.max(thermal_response)
        
        return x * thermal_effect
    
    def amplify(self, x, fs=1e6, add_noise=True, snr_db=40):
        """
        Amplify signal through PA model
        
        Parameters:
        - x: Input signal
        - fs: Sampling frequency
        - add_noise: Add noise to output
        - snr_db: Signal-to-noise ratio
        """
        # Memory polynomial nonlinearity
        y = self.memory_polynomial(x)
        
        # Add thermal memory effects
        y = self.thermal_memory_effect(y, fs)
        
        # Saturation
        y_magnitude = np.abs(y)
        saturation_mask = y_magnitude > self.sat_level
        y[saturation_mask] = self.sat_level * y[saturation_mask] / y_magnitude[saturation_mask]
        
        # Add noise
        if add_noise:
            signal_power = np.mean(np.abs(y)**2)
            noise_power = signal_power / (10**(snr_db/10))
            noise = np.sqrt(noise_power/2) * (np.random.randn(len(y)) + 1j*np.random.randn(len(y)))
            y += noise
            
        return y

class DigitalPredistorter:
    """
    Digital Predistortion System
    Implements various DPD algorithms including MP, GMP, and Volterra series
    """
    
    def __init__(self, memory_depth=5, nonlinearity_order=9, algorithm='memory_polynomial'):
        """
        Initialize DPD system
        
        Parameters:
        - memory_depth: Memory depth for DPD
        - nonlinearity_order: Nonlinearity order
        - algorithm: DPD algorithm ('memory_polynomial', 'generalized_memory_polynomial', 'volterra')
        """
        self.M = memory_depth
        self.K = nonlinearity_order if nonlinearity_order % 2 == 1 else nonlinearity_order + 1
        self.algorithm = algorithm
        
        # DPD coefficients
        self.dpd_coeffs = {}
        self.is_trained = False
        
        # Algorithm-specific parameters
        if algorithm == 'generalized_memory_polynomial':
            self.L = 3  # Cross-term memory depth
            self.Q = 3  # Cross-term lag depth
            
    def generate_basis_functions(self, x):
        """Generate basis functions for different DPD algorithms"""
        N = len(x)
        
        if self.algorithm == 'memory_polynomial':
            return self._mp_basis_functions(x)
        elif self.algorithm == 'generalized_memory_polynomial':
            return self._gmp_basis_functions(x)
        elif self.algorithm == 'volterra':
            return self._volterra_basis_functions(x)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _mp_basis_functions(self, x):
        """Memory Polynomial basis functions"""
        N = len(x)
        x_padded = np.concatenate([np.zeros(self.M-1), x])
        
        # Count number of basis functions
        num_basis = sum(self.M for k in range(1, self.K + 1, 2))
        basis_matrix = np.zeros((N, num_basis), dtype=complex)
        
        col_idx = 0
        for k in range(1, self.K + 1, 2):  # Odd orders
            for m in range(self.M):
                for n in range(N):
                    x_delayed = x_padded[n + self.M - 1 - m]
                    basis_matrix[n, col_idx] = x_delayed * (np.abs(x_delayed) ** (k-1))
                col_idx += 1
                
        return basis_matrix
    
    def _gmp_basis_functions(self, x):
        """Generalized Memory Polynomial basis functions"""
        N = len(x)
        x_padded = np.concatenate([np.zeros(max(self.M, self.L + self.Q)-1), x])
        
        basis_functions = []
        
        # Main MP terms
        mp_basis = self._mp_basis_functions(x)
        basis_functions.append(mp_basis)
        
        # Leading cross-terms: x(n-m) * |x(n-m-l)|^k
        for k in range(1, self.K + 1, 2):
            for m in range(self.M):
                for l in range(1, self.L + 1):
                    basis_col = np.zeros(N, dtype=complex)
                    for n in range(N):
                        idx1 = n + max(self.M, self.L + self.Q) - 1 - m
                        idx2 = n + max(self.M, self.L + self.Q) - 1 - m - l
                        if idx1 >= 0 and idx2 >= 0:
                            x1 = x_padded[idx1]
                            x2 = x_padded[idx2]
                            basis_col[n] = x1 * (np.abs(x2) ** k)
                    basis_functions.append(basis_col.reshape(-1, 1))
        
        # Lagging cross-terms: x(n-m) * |x(n-m+l)|^k  
        for k in range(1, self.K + 1, 2):
            for m in range(self.M):
                for l in range(1, self.Q + 1):
                    basis_col = np.zeros(N, dtype=complex)
                    for n in range(N):
                        idx1 = n + max(self.M, self.L + self.Q) - 1 - m
                        idx2 = n + max(self.M, self.L + self.Q) - 1 - m + l
                        if idx1 >= 0 and idx2 < len(x_padded):
                            x1 = x_padded[idx1]
                            x2 = x_padded[idx2]
                            basis_col[n] = x1 * (np.abs(x2) ** k)
                    basis_functions.append(basis_col.reshape(-1, 1))
        
        return np.hstack(basis_functions)
    
    def _volterra_basis_functions(self, x):
        """Volterra series basis functions (simplified 2nd order)"""
        N = len(x)
        x_padded = np.concatenate([np.zeros(self.M-1), x])
        
        basis_functions = []
        
        # Linear terms
        for m in range(self.M):
            basis_col = np.zeros(N, dtype=complex)
            for n in range(N):
                basis_col[n] = x_padded[n + self.M - 1 - m]
            basis_functions.append(basis_col.reshape(-1, 1))
        
        # Second-order terms
        for m1 in range(self.M):
            for m2 in range(m1, self.M):
                basis_col = np.zeros(N, dtype=complex)
                for n in range(N):
                    x1 = x_padded[n + self.M - 1 - m1]
                    x2 = x_padded[n + self.M - 1 - m2]
                    basis_col[n] = x1 * np.conj(x2)
                basis_functions.append(basis_col.reshape(-1, 1))
        
        return np.hstack(basis_functions)
    
    def train(self, x_train, y_train, method='least_squares', regularization=1e-6):
        """
        Train DPD coefficients
        
        Parameters:
        - x_train: Input training data
        - y_train: PA output training data
        - method: Training method ('least_squares', 'recursive_least_squares')
        - regularization: Regularization parameter
        """
        print(f"Training DPD using {self.algorithm} algorithm...")
        
        # Generate basis functions
        basis_matrix = self.generate_basis_functions(x_train)
        
        if method == 'least_squares':
            # Regularized least squares
            A = basis_matrix
            b = y_train
            
            # Solve: min ||Ac - b||^2 + λ||c||^2
            AtA = np.conj(A.T) @ A
            Atb = np.conj(A.T) @ b
            
            # Add regularization
            AtA += regularization * np.eye(AtA.shape[0])
            
            # Solve for coefficients
            coeffs = np.linalg.solve(AtA, Atb)
            
        elif method == 'recursive_least_squares':
            coeffs = self._rls_training(basis_matrix, y_train, regularization)
        else:
            raise ValueError(f"Unknown training method: {method}")
        
        # Store coefficients
        self.dpd_coeffs = coeffs
        self.is_trained = True
        
        # Calculate training error
        y_pred = basis_matrix @ coeffs
        nmse = self._calculate_nmse(y_train, y_pred)
        
        print(f"Training completed. NMSE: {nmse:.2f} dB")
        return nmse
    
    def _rls_training(self, A, b, forgetting_factor=0.99):
        """Recursive Least Squares training"""
        N, M = A.shape
        
        # Initialize
        coeffs = np.zeros(M, dtype=complex)
        P = np.eye(M) / 0.01  # Inverse correlation matrix
        
        for n in range(N):
            a_n = A[n, :].reshape(-1, 1)
            b_n = b[n]
            
            # RLS update
            k = P @ a_n / (forgetting_factor + np.conj(a_n.T) @ P @ a_n)
            alpha = b_n - np.conj(a_n.T) @ coeffs
            coeffs = coeffs + (k * alpha).flatten()
            P = (P - k @ np.conj(a_n.T) @ P) / forgetting_factor
            
        return coeffs
    
    def predistort(self, x):
        """Apply digital predistortion to input signal"""
        if not self.is_trained:
            raise ValueError("DPD must be trained before use")
        
        # Generate basis functions for input
        basis_matrix = self.generate_basis_functions(x)
        
        # Apply DPD
        x_predistorted = basis_matrix @ self.dpd_coeffs
        
        return x_predistorted
    
    def indirect_learning(self, x_ref, pa_output, pa_model, iterations=5, step_size=0.1):
        """
        Indirect learning architecture for DPD training
        """
        x_dpd = x_ref.copy()
        
        for iteration in range(iterations):
            print(f"Indirect learning iteration {iteration + 1}/{iterations}")
            
            # Train post-inverse
            basis_matrix = self.generate_basis_functions(pa_output)
            
            # Solve for post-inverse coefficients
            AtA = np.conj(basis_matrix.T) @ basis_matrix
            Atb = np.conj(basis_matrix.T) @ x_ref
            AtA += 1e-6 * np.eye(AtA.shape[0])  # Regularization
            
            post_inverse_coeffs = np.linalg.solve(AtA, Atb)
            
            # Update DPD coefficients
            self.dpd_coeffs = post_inverse_coeffs
            self.is_trained = True
            
            # Apply current DPD
            x_dpd = self.predistort(x_ref)
            
            # Get new PA output
            pa_output = pa_model.amplify(x_dpd, add_noise=False)
            
            # Calculate error
            error = self._calculate_nmse(x_ref, pa_output)
            print(f"  NMSE: {error:.2f} dB")
            
            if error < -30:  # Convergence criterion
                break
                
        return x_dpd
    
    def _calculate_nmse(self, reference, actual):
        """Calculate Normalized Mean Square Error in dB"""
        mse = np.mean(np.abs(reference - actual)**2)
        signal_power = np.mean(np.abs(reference)**2)
        nmse_db = 10 * np.log10(mse / signal_power)
        return nmse_db

class DPDPerformanceAnalyzer:
    """
    Analyze DPD performance with various metrics
    """
    
    def __init__(self):
        self.metrics = {}
    
    def analyze_linearity(self, x_ref, y_pa, fs, title="PA Analysis"):
        """Analyze PA linearity and distortion"""
        # Calculate ACPR (Adjacent Channel Power Ratio)
        acpr = self._calculate_acpr(y_pa, fs)
        
        # Calculate EVM (Error Vector Magnitude)
        evm = self._calculate_evm(x_ref, y_pa)
        
        # Calculate AM/AM and AM/PM characteristics
        am_am, am_pm, power_levels = self._calculate_am_characteristics(x_ref, y_pa)
        
        # Store metrics
        self.metrics[title] = {
            'acpr': acpr,
            'evm': evm,
            'am_am': am_am,
            'am_pm': am_pm,
            'power_levels': power_levels
        }
        
        return acpr, evm
    
    def _calculate_acpr(self, signal, fs, channel_bw=20e6):
        """Calculate Adjacent Channel Power Ratio"""
        # FFT of signal
        N = len(signal)
        freqs = np.fft.fftfreq(N, 1/fs)
        spectrum = np.abs(np.fft.fft(signal))**2
        
        # Define frequency bands
        main_channel = (np.abs(freqs) <= channel_bw/2)
        adj_channel_lower = ((freqs >= -1.5*channel_bw) & (freqs <= -0.5*channel_bw))
        adj_channel_upper = ((freqs >= 0.5*channel_bw) & (freqs <= 1.5*channel_bw))
        
        # Calculate power in each band
        main_power = np.sum(spectrum[main_channel])
        adj_power_lower = np.sum(spectrum[adj_channel_lower])
        adj_power_upper = np.sum(spectrum[adj_channel_upper])
        
        # ACPR in dB
        acpr_lower = 10 * np.log10(adj_power_lower / main_power)
        acpr_upper = 10 * np.log10(adj_power_upper / main_power)
        
        return {'lower': acpr_lower, 'upper': acpr_upper}
    
    def _calculate_evm(self, reference, actual):
        """Calculate Error Vector Magnitude"""
        # Align signals (simple delay compensation)
        correlation = np.correlate(actual, reference, mode='full')
        delay = np.argmax(np.abs(correlation)) - len(reference) + 1
        
        if delay > 0:
            actual_aligned = actual[delay:]
            reference_aligned = reference[:-delay] if delay < len(reference) else reference
        elif delay < 0:
            actual_aligned = actual[:delay]
            reference_aligned = reference[-delay:]
        else:
            actual_aligned = actual
            reference_aligned = reference
        
        # Ensure same length
        min_len = min(len(actual_aligned), len(reference_aligned))
        actual_aligned = actual_aligned[:min_len]
        reference_aligned = reference_aligned[:min_len]
        
        # Calculate EVM
        error_power = np.mean(np.abs(reference_aligned - actual_aligned)**2)
        signal_power = np.mean(np.abs(reference_aligned)**2)
        evm_percent = 100 * np.sqrt(error_power / signal_power)
        
        return evm_percent
    
    def _calculate_am_characteristics(self, x_ref, y_pa):
        """Calculate AM/AM and AM/PM characteristics"""
        # Input and output magnitudes
        input_mag = np.abs(x_ref)
        output_mag = np.abs(y_pa)
        
        # Input and output phases
        input_phase = np.angle(x_ref)
        output_phase = np.angle(y_pa)
        phase_diff = np.angle(np.exp(1j * (output_phase - input_phase)))
        
        # Bin by input power levels
        power_levels = np.linspace(0, np.max(input_mag), 50)
        am_am = np.zeros_like(power_levels)
        am_pm = np.zeros_like(power_levels)
        
        for i, power_level in enumerate(power_levels[:-1]):
            mask = (input_mag >= power_level) & (input_mag < power_levels[i+1])
            if np.any(mask):
                am_am[i] = np.mean(output_mag[mask])
                am_pm[i] = np.mean(phase_diff[mask]) * 180 / np.pi
        
        return am_am, am_pm, power_levels
    
    def plot_performance_comparison(self, x_ref, y_pa_no_dpd, y_pa_with_dpd, fs):
        """Plot comprehensive performance comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Time domain comparison
        t = np.arange(min(1000, len(x_ref))) / fs * 1e6  # μs
        axes[0,0].plot(t, np.real(x_ref[:len(t)]), 'b-', label='Reference')
        axes[0,0].plot(t, np.real(y_pa_no_dpd[:len(t)]), 'r--', alpha=0.7, label='PA without DPD')
        axes[0,0].plot(t, np.real(y_pa_with_dpd[:len(t)]), 'g-', alpha=0.7, label='PA with DPD')
        axes[0,0].set_xlabel('Time (μs)')
        axes[0,0].set_ylabel('Amplitude')
        axes[0,0].set_title('Time Domain Comparison')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Frequency domain comparison
        N = 2048
        freqs = np.fft.fftfreq(N, 1/fs) / 1e6  # MHz
        X_ref = 20 * np.log10(np.abs(np.fft.fft(x_ref[:N])) + 1e-12)
        Y_no_dpd = 20 * np.log10(np.abs(np.fft.fft(y_pa_no_dpd[:N])) + 1e-12)
        Y_with_dpd = 20 * np.log10(np.abs(np.fft.fft(y_pa_with_dpd[:N])) + 1e-12)
        
        # Plot positive frequencies only
        pos_idx = freqs >= 0
        axes[0,1].plot(freqs[pos_idx], X_ref[pos_idx], 'b-', label='Reference')
        axes[0,1].plot(freqs[pos_idx], Y_no_dpd[pos_idx], 'r--', alpha=0.7, label='PA without DPD')
        axes[0,1].plot(freqs[pos_idx], Y_with_dpd[pos_idx], 'g-', alpha=0.7, label='PA with DPD')
        axes[0,1].set_xlabel('Frequency (MHz)')
        axes[0,1].set_ylabel('Magnitude (dB)')
        axes[0,1].set_title('Frequency Domain Comparison')
        axes[0,1].legend()
        axes[0,1].grid(True)
        axes[0,1].set_xlim(0, fs/2/1e6)
        
        # Constellation diagram
        axes[0,2].scatter(np.real(x_ref[::10]), np.imag(x_ref[::10]), 
                         c='blue', alpha=0.5, s=1, label='Reference')
        axes[0,2].scatter(np.real(y_pa_no_dpd[::10]), np.imag(y_pa_no_dpd[::10]), 
                         c='red', alpha=0.5, s=1, label='PA without DPD')
        axes[0,2].scatter(np.real(y_pa_with_dpd[::10]), np.imag(y_pa_with_dpd[::10]), 
                         c='green', alpha=0.5, s=1, label='PA with DPD')
        axes[0,2].set_xlabel('In-phase')
        axes[0,2].set_ylabel('Quadrature')
        axes[0,2].set_title('Constellation Diagram')
        axes[0,2].legend()
        axes[0,2].grid(True)
        axes[0,2].axis('equal')
        
        # AM/AM characteristics
        _, am_am_no_dpd, power_levels = self._calculate_am_characteristics(x_ref, y_pa_no_dpd)
        _, am_am_with_dpd, _ = self._calculate_am_characteristics(x_ref, y_pa_with_dpd)
        
        # Ideal AM/AM (linear)
        ideal_am_am = power_levels * np.mean(am_am_no_dpd[:10] / power_levels[:10])
        
        axes[1,0].plot(power_levels, ideal_am_am, 'b-', label='Ideal (Linear)')
        axes[1,0].plot(power_levels, am_am_no_dpd, 'r--', label='PA without DPD')
        axes[1,0].plot(power_levels, am_am_with_dpd, 'g-', label='PA with DPD')
        axes[1,0].set_xlabel('Input Magnitude')
        axes[1,0].set_ylabel('Output Magnitude')
        axes[1,0].set_title('AM/AM Characteristics')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # AM/PM characteristics
        _, am_pm_no_dpd, _ = self._calculate_am_characteristics(x_ref, y_pa_no_dpd)
        _, am_pm_with_dpd, _ = self._calculate_am_characteristics(x_ref, y_pa_with_dpd)
        
        axes[1,1].plot(power_levels, am_pm_no_dpd, 'r--', label='PA without DPD')
        axes[1,1].plot(power_levels, am_pm_with_dpd, 'g-', label='PA with DPD')
        axes[1,1].axhline(y=0, color='b', linestyle='-', label='Ideal (0°)')
        axes[1,1].set_xlabel('Input Magnitude')
        axes[1,1].set_ylabel('Phase Deviation (degrees)')
        axes[1,1].set_title('AM/PM Characteristics')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        # Performance metrics comparison
        acpr_no_dpd = self._calculate_acpr(y_pa_no_dpd, fs)
        acpr_with_dpd = self._calculate_acpr(y_pa_with_dpd, fs)
        evm_no_dpd = self._calculate_evm(x_ref, y_pa_no_dpd)
        evm_with_dpd = self._calculate_evm(x_ref, y_pa_with_dpd)
        
        metrics = ['ACPR Lower', 'ACPR Upper', 'EVM (%)']
        no_dpd_values = [acpr_no_dpd['lower'], acpr_no_dpd['upper'], evm_no_dpd]
        with_dpd_values = [acpr_with_dpd['lower'], acpr_with_dpd['upper'], evm_with_dpd]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        axes[1,2].bar(x_pos - width/2, no_dpd_values, width, label='Without DPD', alpha=0.7)
        axes[1,2].bar(x_pos + width/2, with_dpd_values, width, label='With DPD', alpha=0.7)
        axes[1,2].set_xlabel('Metrics')
        axes[1,2].set_ylabel('Value (dB or %)')
        axes[1,2].set_title('Performance Metrics Comparison')
        axes[1,2].set_xticks(x_pos)
        axes[1,2].set_xticklabels(metrics)
        axes[1,2].legend()
        axes[1,2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print performance summary
        print("\nPerformance Summary:")
        print("=" * 50)
        print(f"{'Metric':<15} {'Without DPD':<15} {'With DPD':<15} {'Improvement':<15}")
        print("-" * 60)
        print(f"{'ACPR Lower':<15} {acpr_no_dpd['lower']:<15.2f} {acpr_with_dpd['lower']:<15.2f} {acpr_with_dpd['lower']-acpr_no_dpd['lower']:<15.2f}")
        print(f"{'ACPR Upper':<15} {acpr_no_dpd['upper']:<15.2f} {acpr_with_dpd['upper']:<15.2f} {acpr_with_dpd['upper']-acpr_no_dpd['upper']:<15.2f}")
        print(f"{'EVM (%)':<15} {evm_no_dpd:<15.2f} {evm_with_dpd:<15.2f} {evm_no_dpd-evm_with_dpd:<15.2f}")

def generate_test_signal(signal_type='qam64', num_symbols=10000, samples_per_symbol=4, 
                        carrier_freq=1e9, fs=100e6, papr_reduction=False):
    """Generate test signals for DPD evaluation"""
    
    if signal_type == 'qam64':
        # Generate 64-QAM symbols
        symbols = np.random.randint(0, 64, num_symbols)
        
        # 64-QAM constellation mapping
        qam64_constellation = []
        for i in range(8):
            for q in range(8):
                qam64_constellation.append(complex(2*i - 7, 2*q - 7))
        qam64_constellation = np.array(qam64_constellation)
        
        # Map symbols to constellation
        modulated_symbols = qam64_constellation[symbols]
        
    elif signal_type == 'qam16':
        # Generate 16-QAM symbols
        symbols = np.random.randint(0, 16, num_symbols)
        qam16_constellation = []
        for i in range(4):
            for q in range(4):
                qam16_constellation.append(complex(2*i - 3, 2*q - 3))
        qam16_constellation = np.array(qam16_constellation)
        modulated_symbols = qam16_constellation[symbols]
        
    elif signal_type == 'ofdm':
        # Generate OFDM signal
        num_subcarriers = 64
        num_ofdm_symbols = num_symbols // num_subcarriers
        
        # Generate random QAM data for each subcarrier
        qam_data = np.random.randint(0, 16, (num_ofdm_symbols, num_subcarriers))
        qam16_constellation = np.array([complex(2*i - 3, 2*q - 3) 
                                      for i in range(4) for q in range(4)])
        
        ofdm_symbols = []
        for symbol_idx in range(num_ofdm_symbols):
            # Map to constellation
            freq_domain = qam16_constellation[qam_data[symbol_idx, :]]
            
            # IFFT to time domain
            time_domain = np.fft.ifft(freq_domain)
            
            # Add cyclic prefix (25% of symbol length)
            cp_length = num_subcarriers // 4
            time_domain_with_cp = np.concatenate([time_domain[-cp_length:], time_domain])
            
            ofdm_symbols.extend(time_domain_with_cp)
            
        modulated_symbols = np.array(ofdm_symbols)
        
    elif signal_type == 'lte':
        # Simplified LTE-like signal
        num_subcarriers = 72  # 1.4 MHz LTE
        num_ofdm_symbols = num_symbols // num_subcarriers
        
        # Generate reference signals and data
        lte_symbols = []
        for symbol_idx in range(num_ofdm_symbols):
            freq_domain = np.zeros(num_subcarriers, dtype=complex)
            
            # Add reference signals every 6th subcarrier
            for k in range(0, num_subcarriers, 6):
                freq_domain[k] = 1 + 1j  # Reference signal
                
            # Fill data subcarriers with QAM
            data_indices = [k for k in range(num_subcarriers) if k % 6 != 0]
            qam_data = np.random.randint(0, 16, len(data_indices))
            qam16_constellation = np.array([complex(2*i - 3, 2*q - 3) 
                                          for i in range(4) for q in range(4)])
            freq_domain[data_indices] = qam16_constellation[qam_data]
            
            # IFFT and add cyclic prefix
            time_domain = np.fft.ifft(freq_domain)
            cp_length = num_subcarriers // 4
            time_domain_with_cp = np.concatenate([time_domain[-cp_length:], time_domain])
            
            lte_symbols.extend(time_domain_with_cp)
            
        modulated_symbols = np.array(lte_symbols)
        
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
    
    # Normalize to unit power
    modulated_symbols = modulated_symbols / np.sqrt(np.mean(np.abs(modulated_symbols)**2))
    
    # Upsample
    upsampled = np.zeros(len(modulated_symbols) * samples_per_symbol, dtype=complex)
    upsampled[::samples_per_symbol] = modulated_symbols
    
    # Apply pulse shaping filter (root raised cosine)
    beta = 0.35  # Roll-off factor
    span = 10    # Filter span in symbols
    sps = samples_per_symbol
    
    # Create RRC filter
    t = np.arange(-span*sps, span*sps + 1) / sps
    rrc_filter = np.zeros_like(t)
    
    for i, time in enumerate(t):
        if time == 0:
            rrc_filter[i] = 1 + beta*(4/np.pi - 1)
        elif abs(time) == 1/(4*beta):
            rrc_filter[i] = (beta/np.sqrt(2)) * ((1 + 2/np.pi)*np.sin(np.pi/(4*beta)) + 
                                                (1 - 2/np.pi)*np.cos(np.pi/(4*beta)))
        else:
            numerator = np.sin(np.pi*time*(1-beta)) + 4*beta*time*np.cos(np.pi*time*(1+beta))
            denominator = np.pi*time*(1-(4*beta*time)**2)
            rrc_filter[i] = numerator / denominator
    
    # Apply pulse shaping
    filtered_signal = np.convolve(upsampled, rrc_filter, mode='same')
    
    # PAPR reduction (optional)
    if papr_reduction:
        filtered_signal = apply_papr_reduction(filtered_signal)
    
    # Normalize power
    filtered_signal = filtered_signal / np.sqrt(np.mean(np.abs(filtered_signal)**2))
    
    return filtered_signal

def apply_papr_reduction(signal, method='clipping', clip_ratio=1.5):
    """Apply PAPR reduction techniques"""
    
    if method == 'clipping':
        # Simple amplitude clipping
        magnitude = np.abs(signal)
        max_amplitude = clip_ratio * np.sqrt(np.mean(magnitude**2))
        
        clipping_mask = magnitude > max_amplitude
        clipped_signal = signal.copy()
        clipped_signal[clipping_mask] = (max_amplitude * 
                                       signal[clipping_mask] / magnitude[clipping_mask])
        
        return clipped_signal
        
    elif method == 'companding':
        # μ-law companding
        mu = 100
        magnitude = np.abs(signal)
        phase = np.angle(signal)
        
        # Apply μ-law compression
        max_mag = np.max(magnitude)
        normalized_mag = magnitude / max_mag
        compressed_mag = max_mag * np.log(1 + mu * normalized_mag) / np.log(1 + mu)
        
        return compressed_mag * np.exp(1j * phase)
    
    else:
        return signal

def main():
    """Main demonstration of Digital Predistortion system"""
    
    print("Digital Predistortion for Power Amplifier with Memory Effects")
    print("=" * 65)
    
    # System parameters
    fs = 100e6          # Sampling frequency (100 MHz)
    num_symbols = 5000   # Number of symbols
    sps = 4             # Samples per symbol
    
    # Generate test signal
    print("Generating test signal...")
    test_signal = generate_test_signal('lte', num_symbols, sps, fs=fs)
    
    # Scale to desired power level
    desired_power_dbm = 10  # dBm
    desired_power_watts = 10**(desired_power_dbm/10) / 1000
    current_power = np.mean(np.abs(test_signal)**2)
    scale_factor = np.sqrt(desired_power_watts / current_power)
    test_signal = test_signal * scale_factor
    
    print(f"Generated {len(test_signal)} samples")
    print(f"Signal power: {desired_power_dbm} dBm")
    print(f"PAPR: {10*np.log10(np.max(np.abs(test_signal)**2) / np.mean(np.abs(test_signal)**2)):.1f} dB")
    
    # Create PA model
    print("\nCreating Power Amplifier model...")
    pa = PowerAmplifierModel(memory_depth=5, nonlinearity_order=9, saturation_level=1.2)
    
    # Generate PA output without DPD
    print("Amplifying signal without DPD...")
    pa_output_no_dpd = pa.amplify(test_signal, fs, add_noise=True, snr_db=35)
    
    # Create DPD system
    algorithms = ['memory_polynomial', 'generalized_memory_polynomial']
    
    for algorithm in algorithms:
        print(f"\n{algorithm.replace('_', ' ').title()} DPD Algorithm")
        print("-" * 50)
        
        # Initialize DPD
        dpd = DigitalPredistorter(memory_depth=5, nonlinearity_order=9, algorithm=algorithm)
        
        # Split data for training and testing
        train_size = len(test_signal) // 2
        x_train = test_signal[:train_size]
        x_test = test_signal[train_size:]
        
        # Get PA output for training
        y_train = pa.amplify(x_train, fs, add_noise=False)
        
        # Train DPD using direct learning
        print("Training DPD (Direct Learning)...")
        training_nmse = dpd.train(x_train, y_train, method='least_squares', regularization=1e-6)
        
        # Test DPD performance
        print("Testing DPD performance...")
        x_predistorted = dpd.predistort(x_test)
        pa_output_with_dpd = pa.amplify(x_predistorted, fs, add_noise=True, snr_db=35)
        
        # Indirect learning architecture
        print("Training DPD (Indirect Learning)...")
        dpd_indirect = DigitalPredistorter(memory_depth=5, nonlinearity_order=9, algorithm=algorithm)
        x_predistorted_indirect = dpd_indirect.indirect_learning(
            x_test, pa_output_no_dpd[train_size:], pa, iterations=3)
        pa_output_indirect = pa.amplify(x_predistorted_indirect, fs, add_noise=True, snr_db=35)
        
        # Performance analysis
        analyzer = DPDPerformanceAnalyzer()
        
        # Analyze performance
        acpr_no_dpd, evm_no_dpd = analyzer.analyze_linearity(
            x_test, pa_output_no_dpd[train_size:], fs, "Without DPD")
        
        acpr_direct, evm_direct = analyzer.analyze_linearity(
            x_test, pa_output_with_dpd, fs, f"With {algorithm} DPD (Direct)")
        
        acpr_indirect, evm_indirect = analyzer.analyze_linearity(
            x_test, pa_output_indirect, fs, f"With {algorithm} DPD (Indirect)")
        
        print(f"\nPerformance Results for {algorithm}:")
        print(f"  Without DPD - ACPR: {acpr_no_dpd['lower']:.1f}/{acpr_no_dpd['upper']:.1f} dB, EVM: {evm_no_dpd:.2f}%")
        print(f"  Direct DPD  - ACPR: {acpr_direct['lower']:.1f}/{acpr_direct['upper']:.1f} dB, EVM: {evm_direct:.2f}%")
        print(f"  Indirect DPD- ACPR: {acpr_indirect['lower']:.1f}/{acpr_indirect['upper']:.1f} dB, EVM: {evm_indirect:.2f}%")
        
        # Plot results for the first algorithm
        if algorithm == algorithms[0]:
            print("\nGenerating performance plots...")
            analyzer.plot_performance_comparison(
                x_test, pa_output_no_dpd[train_size:], pa_output_with_dpd, fs)
    
    # Advanced DPD techniques demonstration
    print("\nAdvanced DPD Techniques:")
    print("-" * 30)
    
    # Adaptive DPD with tracking
    print("1. Adaptive DPD with coefficient tracking...")
    dpd_adaptive = DigitalPredistorter(memory_depth=5, nonlinearity_order=7, 
                                     algorithm='memory_polynomial')
    
    # Simulate time-varying PA characteristics
    time_varying_signal = test_signal.copy()
    adaptive_outputs = []
    
    block_size = 1000
    num_blocks = len(time_varying_signal) // block_size
    
    for block in range(num_blocks):
        start_idx = block * block_size
        end_idx = (block + 1) * block_size
        
        x_block = time_varying_signal[start_idx:end_idx]
        
        # Simulate PA drift (temperature, aging effects)
        pa.coefficients[(1, 0)] *= (1 + 0.01 * np.sin(block * 0.1))  # Slow gain drift
        
        if block == 0:
            # Initial training
            y_block = pa.amplify(x_block, fs, add_noise=False)
            dpd_adaptive.train(x_block, y_block, method='least_squares')
        else:
            # Adaptive tracking with RLS
            x_predistorted_block = dpd_adaptive.predistort(x_block)
            y_block = pa.amplify(x_predistorted_block, fs, add_noise=True, snr_db=35)
            
            # Update coefficients with RLS
            basis_matrix = dpd_adaptive.generate_basis_functions(x_predistorted_block)
            dpd_adaptive.dpd_coeffs = dpd_adaptive._rls_training(basis_matrix, x_block, 0.95)
        
        adaptive_outputs.extend(y_block)
    
    adaptive_outputs = np.array(adaptive_outputs)
    
    # Calculate performance
    test_length = min(len(time_varying_signal), len(adaptive_outputs))
    acpr_adaptive, evm_adaptive = analyzer.analyze_linearity(
        time_varying_signal[:test_length], adaptive_outputs[:test_length], fs, "Adaptive DPD")
    
    print(f"   Adaptive DPD - ACPR: {acpr_adaptive['lower']:.1f}/{acpr_adaptive['upper']:.1f} dB, EVM: {evm_adaptive:.2f}%")
    
    # Multi-band DPD
    print("2. Multi-band DPD simulation...")
    
    # Generate two-carrier signal
    carrier1 = generate_test_signal('qam16', num_symbols//2, sps, fs=fs)
    carrier2 = generate_test_signal('qam64', num_symbols//2, sps, fs=fs)
    
    # Frequency shift carriers
    t = np.arange(len(carrier1)) / fs
    f1, f2 = -10e6, 10e6  # ±10 MHz offset
    carrier1_shifted = carrier1 * np.exp(1j * 2 * np.pi * f1 * t)
    carrier2_shifted = carrier2 * np.exp(1j * 2 * np.pi * f2 * t)
    
    multi_band_signal = carrier1_shifted + carrier2_shifted
    multi_band_signal = multi_band_signal / np.sqrt(np.mean(np.abs(multi_band_signal)**2))
    
    # Apply DPD to multi-band signal
    dpd_multiband = DigitalPredistorter(memory_depth=7, nonlinearity_order=9, 
                                      algorithm='generalized_memory_polynomial')
    
    mb_train_size = len(multi_band_signal) // 2
    mb_pa_output = pa.amplify(multi_band_signal[:mb_train_size], fs, add_noise=False)
    dpd_multiband.train(multi_band_signal[:mb_train_size], mb_pa_output)
    
    mb_predistorted = dpd_multiband.predistort(multi_band_signal[mb_train_size:])
    mb_final_output = pa.amplify(mb_predistorted, fs, add_noise=True, snr_db=35)
    
    acpr_mb, evm_mb = analyzer.analyze_linearity(
        multi_band_signal[mb_train_size:], mb_final_output, fs, "Multi-band DPD")
    
    print(f"   Multi-band DPD - ACPR: {acpr_mb['lower']:.1f}/{acpr_mb['upper']:.1f} dB, EVM: {evm_mb:.2f}%")
    
    print("\nDPD demonstration completed!")
    print("Key Benefits Demonstrated:")
    print("- Linearization of PA nonlinearities")
    print("- Memory effect compensation")
    print("- Improved spectral efficiency (reduced ACPR)")
    print("- Enhanced signal quality (reduced EVM)")
    print("- Adaptive tracking of PA variations")
    print("- Multi-band operation capability")

if __name__ == "__main__":
    main()