import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

class ChannelModel:
    """
    Channel model for high-speed digital communications
    Includes ISI, crosstalk, and frequency-dependent losses
    """
    
    def __init__(self, channel_type='copper_trace', length=10, data_rate=25e9):
        """
        Initialize channel model
        
        Parameters:
        - channel_type: 'copper_trace', 'backplane', 'cable'
        - length: Physical length in inches/meters
        - data_rate: Data rate in bps
        """
        self.channel_type = channel_type
        self.length = length
        self.data_rate = data_rate
        self.fs = data_rate * 2  # Oversampling factor of 2
        
        # Generate channel impulse response
        self.impulse_response = self._generate_channel_response()
        
    def _generate_channel_response(self):
        """Generate realistic channel impulse response"""
        
        if self.channel_type == 'copper_trace':
            # PCB trace with skin effect and dielectric losses
            # Approximate impulse response for FR4 PCB trace
            
            # Dominant pole frequency (GHz)
            f_pole = 5e9 / (self.length * 0.1)  # Approximate scaling
            
            # Create frequency response
            freqs = np.logspace(6, 12, 1000)  # 1 MHz to 1 THz
            
            # Skin effect loss (√f dependency)
            skin_loss_db = 0.05 * self.length * np.sqrt(freqs / 1e9)
            
            # Dielectric loss (f dependency)  
            dielectric_loss_db = 0.02 * self.length * (freqs / 1e9)
            
            # Total loss
            total_loss_db = skin_loss_db + dielectric_loss_db
            
            # Convert to linear magnitude
            magnitude = 10**(-total_loss_db / 20)
            
            # Add phase (minimum phase approximation)
            phase = -np.imag(signal.hilbert(np.log(magnitude)))
            
            # Create complex frequency response
            H = magnitude * np.exp(1j * phase)
            
            # Convert to impulse response
            # Pad with zeros for IFFT
            n_fft = 2048
            freqs_padded = np.linspace(0, self.fs/2, n_fft//2 + 1)
            H_padded = np.interp(freqs_padded, freqs, np.real(H)) + \
                      1j * np.interp(freqs_padded, freqs, np.imag(H))
            
            # Create full spectrum (conjugate symmetry)
            H_full = np.concatenate([H_padded, np.conj(H_padded[-2:0:-1])])
            
            # IFFT to get impulse response
            impulse_response = np.real(np.fft.ifft(H_full))
            
            # Keep only causal part and normalize
            impulse_response = impulse_response[:n_fft//4]
            impulse_response = impulse_response / np.sum(impulse_response)
            
        elif self.channel_type == 'backplane':
            # Backplane channel with multiple reflections
            # Simplified model with main tap and reflections
            
            main_delay = int(self.length * 1e-10 * self.fs)  # Propagation delay
            impulse_response = np.zeros(main_delay + 50)
            
            # Main pulse
            impulse_response[main_delay] = 0.8
            
            # Pre-cursor ISI (reflections)
            impulse_response[main_delay - 5:main_delay] = 0.05 * np.random.randn(5)
            
            # Post-cursor ISI (reflections and losses)
            post_cursor_length = 30
            post_cursor = 0.3 * np.exp(-np.arange(post_cursor_length) / 10) * \
                         (1 + 0.1 * np.random.randn(post_cursor_length))
            impulse_response[main_delay + 1:main_delay + 1 + post_cursor_length] = post_cursor
            
        elif self.channel_type == 'cable':
            # Coaxial or differential cable
            # Frequency-dependent loss model
            
            # Cable parameters
            R = 0.1 * self.length  # Resistance per unit length
            L = 0.5e-6 * self.length  # Inductance per unit length  
            C = 100e-12 * self.length  # Capacitance per unit length
            G = 1e-6 * self.length  # Conductance per unit length
            
            # Create s-domain transfer function
            # H(s) = 1 / (1 + s*R*C + s^2*L*C)
            
            # Convert to discrete time
            dt = 1 / self.fs
            n_taps = 100
            t = np.arange(n_taps) * dt
            
            # Approximate impulse response (simplified)
            tau = np.sqrt(L * C)
            impulse_response = (1 / tau) * np.exp(-t / tau) * \
                             np.exp(-R * t / (2 * L))
            
        else:
            # Default: simple 3-tap channel
            impulse_response = np.array([0.1, 0.8, 0.1])
            
        return impulse_response
    
    def add_channel_effects(self, tx_signal, add_noise=True, snr_db=20, add_jitter=True):
        """
        Apply channel effects to transmitted signal
        
        Parameters:
        - tx_signal: Transmitted NRZ signal
        - add_noise: Add AWGN
        - snr_db: Signal-to-noise ratio
        - add_jitter: Add timing jitter
        """
        # Apply channel impulse response (ISI)
        rx_signal = np.convolve(tx_signal, self.impulse_response, mode='same')
        
        # Add timing jitter
        if add_jitter:
            jitter_std = 0.05  # 5% of UI
            jitter_samples = np.random.normal(0, jitter_std, len(rx_signal))
            
            # Apply jitter by interpolating
            t_ideal = np.arange(len(rx_signal))
            t_jittered = t_ideal + jitter_samples
            
            # Ensure we don't go out of bounds
            t_jittered = np.clip(t_jittered, 0, len(rx_signal) - 1)
            
            rx_signal = np.interp(t_ideal, t_jittered, rx_signal)
        
        # Add AWGN
        if add_noise:
            signal_power = np.mean(rx_signal**2)
            noise_power = signal_power / (10**(snr_db/10))
            noise = np.sqrt(noise_power) * np.random.randn(len(rx_signal))
            rx_signal += noise
        
        return rx_signal

class AutoCorrelationEqualizer:
    """
    Adaptive equalizer using auto-correlation based coefficient adaptation
    Implements both FFE and DFE structures with optimized 2-tap DFE
    """
    
    def __init__(self, ffe_taps=15, dfe_taps=2, step_size=0.001):
        """
        Initialize adaptive equalizer
        
        Parameters:
        - ffe_taps: Number of feed-forward equalizer taps
        - dfe_taps: Number of decision feedback equalizer taps (optimized for 2)
        - step_size: Adaptation step size
        """
        self.N_ffe = ffe_taps
        self.N_dfe = dfe_taps
        self.mu = step_size
        
        # Initialize coefficients
        self.w_ffe = np.zeros(self.N_ffe)
        self.w_dfe = np.zeros(self.N_dfe)
        
        # Set center tap of FFE to 1 (initial condition)
        center_tap = self.N_ffe // 2
        self.w_ffe[center_tap] = 1.0
        
        # Buffers for signal history
        self.ffe_buffer = np.zeros(self.N_ffe)
        self.dfe_buffer = np.zeros(self.N_dfe)
        
        # Adaptation parameters
        self.adaptation_enabled = True
        self.decision_directed = True
        
        # Performance tracking
        self.mse_history = []
        self.coefficient_history = {'ffe': [], 'dfe': []}
        
        # DFE-specific optimizations for 2-tap structure
        if self.N_dfe == 2:
            self._optimize_dfe_2tap = True
            # Pre-calculate common terms for 2-tap DFE
            self._dfe_alpha = 0.9  # Forgetting factor for correlation estimation
            self._dfe_correlation_history = np.zeros(2)
        else:
            self._optimize_dfe_2tap = False
        
    def hard_decision(self, x):
        """Hard decision slicer for NRZ (+1/-1)"""
        return np.sign(x)
    
    def soft_decision(self, x, alpha=0.8):
        """Soft decision with limiting"""
        return np.clip(x, -1/alpha, 1/alpha) * alpha
    
    def update_buffers(self, input_sample, decision):
        """Update FFE and DFE buffers"""
        # Shift FFE buffer and add new input
        self.ffe_buffer[1:] = self.ffe_buffer[:-1]
        self.ffe_buffer[0] = input_sample
        
        # Shift DFE buffer and add new decision
        if self.N_dfe > 0:
            self.dfe_buffer[1:] = self.dfe_buffer[:-1]
            self.dfe_buffer[0] = decision
    
    def equalize_sample(self, input_sample):
        """Process single sample through equalizer"""
        # FFE output
        ffe_output = np.dot(self.w_ffe, self.ffe_buffer)
        
        # DFE output (subtract ISI from previous decisions)
        dfe_output = np.dot(self.w_dfe, self.dfe_buffer) if self.N_dfe > 0 else 0
        
        # Equalizer output
        equalizer_output = ffe_output - dfe_output
        
        return equalizer_output, ffe_output, dfe_output
    
    def autocorrelation_adaptation(self, error, ffe_output, dfe_output):
        """
        Auto-correlation based coefficient adaptation
        Optimized for 2-tap DFE structure
        
        The key insight is that for optimal equalization, the error should be
        uncorrelated with the equalizer inputs (orthogonality principle)
        """
        
        if not self.adaptation_enabled:
            return
            
        # FFE coefficient update using auto-correlation
        # Δw_ffe = μ * error * ffe_input
        ffe_gradient = error * self.ffe_buffer
        self.w_ffe += self.mu * ffe_gradient
        
        # DFE coefficient update using auto-correlation  
        # Optimized for 2-tap DFE
        if self.N_dfe > 0:
            if self._optimize_dfe_2tap and self.N_dfe == 2:
                # Specialized 2-tap DFE adaptation
                # Use exponential averaging for correlation estimation
                current_correlation = error * self.dfe_buffer
                self._dfe_correlation_history = (self._dfe_alpha * self._dfe_correlation_history + 
                                               (1 - self._dfe_alpha) * current_correlation)
                
                # Enhanced gradient with correlation history
                dfe_gradient = self._dfe_correlation_history
                
                # Apply adaptive step size based on correlation strength
                correlation_strength = np.abs(self._dfe_correlation_history)
                adaptive_step = self.mu * (1 + 0.5 * correlation_strength)
                
                self.w_dfe += adaptive_step * dfe_gradient
            else:
                # Standard DFE adaptation
                dfe_gradient = error * self.dfe_buffer
                self.w_dfe += self.mu * dfe_gradient
    
    def wiener_solution_adaptation(self, input_buffer, desired_buffer, regularization=1e-6):
        """
        Direct Wiener solution for optimal coefficients
        Uses auto-correlation matrix and cross-correlation vector
        """
        
        # Build auto-correlation matrix for FFE
        R_ffe = np.zeros((self.N_ffe, self.N_ffe))
        for i in range(self.N_ffe):
            for j in range(self.N_ffe):
                if len(input_buffer) > max(i, j):
                    R_ffe[i, j] = np.correlate(input_buffer[i:], input_buffer[j:], mode='valid')[0]
        
        # Cross-correlation vector for FFE
        p_ffe = np.zeros(self.N_ffe)
        for i in range(self.N_ffe):
            if len(input_buffer) > i and len(desired_buffer) > i:
                p_ffe[i] = np.correlate(desired_buffer, input_buffer[i:], mode='valid')[0]
        
        # Solve Wiener-Hopf equation: R * w = p
        # Add regularization for numerical stability
        R_ffe_reg = R_ffe + regularization * np.eye(self.N_ffe)
        
        try:
            self.w_ffe = np.linalg.solve(R_ffe_reg, p_ffe)
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix in Wiener solution, using pseudoinverse")
            self.w_ffe = np.linalg.pinv(R_ffe_reg) @ p_ffe
    
    def process_block(self, input_signal, reference_signal=None, adaptation_mode='lms'):
        """
        Process a block of samples through the equalizer
        
        Parameters:
        - input_signal: Received signal samples
        - reference_signal: Known reference (for training mode)
        - adaptation_mode: 'lms', 'wiener', 'rls'
        """
        
        N = len(input_signal)
        equalizer_output = np.zeros(N)
        error_signal = np.zeros(N)
        decisions = np.zeros(N)
        
        # Initialize buffers with zeros
        self.ffe_buffer = np.zeros(self.N_ffe)
        self.dfe_buffer = np.zeros(self.N_dfe)
        
        for n in range(N):
            # Update input buffer
            self.update_buffers(input_signal[n], 
                              self.dfe_buffer[0] if n > 0 else 0)
            
            # Equalize current sample
            eq_out, ffe_out, dfe_out = self.equalize_sample(input_signal[n])
            equalizer_output[n] = eq_out
            
            # Make decision
            if self.decision_directed and reference_signal is None:
                decision = self.hard_decision(eq_out)
            elif reference_signal is not None:
                decision = reference_signal[n]  # Training mode
            else:
                decision = self.soft_decision(eq_out)
            
            decisions[n] = decision
            
            # Calculate error
            error = decision - eq_out
            error_signal[n] = error
            
            # Adapt coefficients
            if adaptation_mode == 'lms':
                self.autocorrelation_adaptation(error, ffe_out, dfe_out)
            elif adaptation_mode == 'wiener' and n > max(self.N_ffe, self.N_dfe):
                # Apply Wiener solution periodically
                if n % 100 == 0:  # Update every 100 samples
                    start_idx = max(0, n - 500)
                    self.wiener_solution_adaptation(
                        input_signal[start_idx:n], 
                        decisions[start_idx:n] if reference_signal is None else reference_signal[start_idx:n])
            
            # Update DFE buffer with current decision
            if self.N_dfe > 0:
                self.dfe_buffer[1:] = self.dfe_buffer[:-1]
                self.dfe_buffer[0] = decision
            
            # Track performance
            if n % 10 == 0:  # Subsample for efficiency
                mse = np.mean(error_signal[max(0, n-100):n+1]**2)
                self.mse_history.append(mse)
                self.coefficient_history['ffe'].append(self.w_ffe.copy())
                self.coefficient_history['dfe'].append(self.w_dfe.copy())
        
        return equalizer_output, error_signal, decisions
    
    def calculate_eye_diagram(self, signal, samples_per_symbol=2):
        """Calculate eye diagram from equalized signal"""
        
        # Reshape signal into symbol periods
        n_symbols = len(signal) // samples_per_symbol
        eye_data = signal[:n_symbols * samples_per_symbol].reshape(n_symbols, samples_per_symbol)
        
        return eye_data
    
    def analyze_performance(self, input_signal, output_signal, decisions, reference=None):
        """Analyze equalizer performance"""
        
        # Calculate SNR improvement
        input_snr = 10 * np.log10(np.var(input_signal) / np.var(input_signal - np.mean(input_signal)))
        output_snr = 10 * np.log10(np.var(output_signal) / np.var(output_signal - decisions))
        snr_improvement = output_snr - input_snr
        
        # Calculate BER (if reference available)
        ber = None
        if reference is not None:
            bit_errors = np.sum(np.sign(decisions) != np.sign(reference))
            ber = bit_errors / len(reference)
        
        # Calculate residual ISI
        # Measure eye opening
        eye_data = self.calculate_eye_diagram(output_signal)
        eye_height = np.max(eye_data) - np.min(eye_data)
        eye_width_ratio = 0.8  # Assume 80% eye opening is acceptable
        
        # Timing margins
        mid_sample = eye_data.shape[1] // 2
        eye_opening = eye_data[:, mid_sample]
        eye_opening_height = np.percentile(eye_opening, 95) - np.percentile(eye_opening, 5)
        
        performance_metrics = {
            'snr_improvement_db': snr_improvement,
            'ber': ber,
            'eye_height': eye_height,
            'eye_opening_height': eye_opening_height,
            'final_mse': self.mse_history[-1] if self.mse_history else None,
            'ffe_coefficients': self.w_ffe.copy(),
            'dfe_coefficients': self.w_dfe.copy()
        }
        
        return performance_metrics

class NRZTransceiver:
    """
    Complete NRZ transceiver with adaptive equalization
    """
    
    def __init__(self, data_rate=25e9, samples_per_symbol=2):
        """
        Initialize NRZ transceiver
        
        Parameters:
        - data_rate: Data rate in bps
        - samples_per_symbol: Oversampling factor
        """
        self.data_rate = data_rate
        self.sps = samples_per_symbol
        self.fs = data_rate * samples_per_symbol
        
        # Transmit filter (optional)
        self.use_tx_filter = True
        if self.use_tx_filter:
            self.tx_filter = self._design_tx_filter()
        
    def _design_tx_filter(self):
        """Design transmit pulse shaping filter"""
        # Simple 4th order Bessel filter for pulse shaping
        # Cutoff at 0.7 * Nyquist frequency
        fc = 0.7 * self.data_rate / 2
        sos = signal.bessel(4, fc, btype='low', fs=self.fs, output='sos')
        return sos
    
    def generate_prbs(self, length, polynomial='prbs15'):
        """Generate PRBS (Pseudo-Random Binary Sequence)"""
        
        if polynomial == 'prbs7':
            # x^7 + x^6 + 1
            feedback_taps = [7, 6]
            register_length = 7
        elif polynomial == 'prbs15':
            # x^15 + x^14 + 1  
            feedback_taps = [15, 14]
            register_length = 15
        elif polynomial == 'prbs23':
            # x^23 + x^18 + 1
            feedback_taps = [23, 18]
            register_length = 23
        elif polynomial == 'prbs31':
            # x^31 + x^28 + 1
            feedback_taps = [31, 28]
            register_length = 31
        else:
            raise ValueError(f"Unknown PRBS polynomial: {polynomial}")
        
        # Initialize shift register with seed
        shift_register = np.ones(register_length, dtype=int)
        prbs_sequence = np.zeros(length, dtype=int)
        
        for i in range(length):
            # Output current bit (convert to NRZ: 0 -> -1, 1 -> +1)
            output_bit = shift_register[-1]
            prbs_sequence[i] = 2 * output_bit - 1
            
            # Calculate feedback
            feedback = 0
            for tap in feedback_taps:
                feedback ^= shift_register[tap - 1]
            
            # Shift register and insert feedback
            shift_register[1:] = shift_register[:-1]
            shift_register[0] = feedback
        
        return prbs_sequence
    
    def transmit(self, data_bits):
        """
        Transmit NRZ signal
        
        Parameters:
        - data_bits: Binary data (+1/-1 for NRZ)
        """
        
        # Upsample to symbol rate
        upsampled = np.zeros(len(data_bits) * self.sps)
        upsampled[::self.sps] = data_bits
        
        # Apply transmit filter
        if self.use_tx_filter:
            tx_signal = signal.sosfilt(self.tx_filter, upsampled)
        else:
            tx_signal = upsampled
        
        return tx_signal
    
    def receive_with_equalizer(self, rx_signal, channel, equalizer, 
                              training_length=1000, reference_data=None):
        """
        Receive signal with adaptive equalization
        
        Parameters:
        - rx_signal: Received signal
        - channel: Channel model
        - equalizer: Adaptive equalizer
        - training_length: Length of training sequence
        - reference_data: Known training data
        """
        
        # Training phase
        if reference_data is not None and training_length > 0:
            print("Training equalizer...")
            
            # Use known reference for training
            train_signal = rx_signal[:training_length * self.sps]
            train_reference = np.repeat(reference_data[:training_length], self.sps)
            
            equalizer.adaptation_enabled = True
            equalizer.decision_directed = False
            
            eq_out_train, error_train, decisions_train = equalizer.process_block(
                train_signal, train_reference, adaptation_mode='lms')
            
            print(f"Training MSE: {np.mean(error_train**2):.6f}")
        
        # Switch to decision-directed mode
        print("Switching to decision-directed mode...")
        equalizer.decision_directed = True
        
        # Process remaining signal
        remaining_signal = rx_signal[training_length * self.sps:]
        
        eq_output, error_signal, decisions = equalizer.process_block(
            remaining_signal, adaptation_mode='lms')
        
        # Downsample to symbol rate
        symbol_decisions = decisions[::self.sps]
        
        return eq_output, error_signal, decisions, symbol_decisions

def plot_results(transceiver, channel, equalizer, tx_data, rx_signal, 
                eq_output, decisions, performance_metrics):
    """Plot comprehensive results"""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    # Plot 1: Channel impulse response
    axes[0,0].stem(channel.impulse_response)
    axes[0,0].set_title('Channel Impulse Response')
    axes[0,0].set_xlabel('Sample')
    axes[0,0].set_ylabel('Amplitude')
    axes[0,0].grid(True)
    
    # Plot 2: Received signal (time domain)
    t = np.arange(min(500, len(rx_signal))) / transceiver.fs * 1e9  # ns
    axes[0,1].plot(t, rx_signal[:len(t)])
    axes[0,1].set_title('Received Signal')
    axes[0,1].set_xlabel('Time (ns)')
    axes[0,1].set_ylabel('Amplitude')
    axes[0,1].grid(True)
    
    # Plot 3: Equalized signal
    t_eq = np.arange(min(500, len(eq_output))) / transceiver.fs * 1e9  # ns
    axes[0,2].plot(t_eq, eq_output[:len(t_eq)])
    axes[0,2].set_title('Equalized Signal')
    axes[0,2].set_xlabel('Time (ns)')
    axes[0,2].set_ylabel('Amplitude')
    axes[0,2].grid(True)
    
    # Plot 4: FFE coefficients
    axes[1,0].stem(equalizer.w_ffe)
    axes[1,0].set_title('FFE Coefficients')
    axes[1,0].set_xlabel('Tap')
    axes[1,0].set_ylabel('Coefficient Value')
    axes[1,0].grid(True)
    
    # Plot 5: DFE coefficients
    if equalizer.N_dfe > 0:
        axes[1,1].stem(equalizer.w_dfe)
        axes[1,1].set_title('DFE Coefficients')
        axes[1,1].set_xlabel('Tap')
        axes[1,1].set_ylabel('Coefficient Value')
        axes[1,1].grid(True)
    else:
        axes[1,1].text(0.5, 0.5, 'No DFE', ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('DFE Coefficients')
    
    # Plot 6: MSE convergence
    if equalizer.mse_history:
        axes[1,2].semilogy(equalizer.mse_history)
        axes[1,2].set_title('MSE Convergence')
        axes[1,2].set_xlabel('Iteration')
        axes[1,2].set_ylabel('MSE')
        axes[1,2].grid(True)
    
    # Plot 7: Eye diagram (before equalization)
    eye_data_rx = equalizer.calculate_eye_diagram(rx_signal[:1000])
    for i in range(min(50, eye_data_rx.shape[0])):
        axes[2,0].plot(eye_data_rx[i, :], 'b-', alpha=0.3)
    axes[2,0].set_title('Eye Diagram (Before EQ)')
    axes[2,0].set_xlabel('Sample within Symbol')
    axes[2,0].set_ylabel('Amplitude')
    axes[2,0].grid(True)
    
    # Plot 8: Eye diagram (after equalization)
    eye_data_eq = equalizer.calculate_eye_diagram(eq_output[:1000])
    for i in range(min(50, eye_data_eq.shape[0])):
        axes[2,1].plot(eye_data_eq[i, :], 'g-', alpha=0.3)
    axes[2,1].set_title('Eye Diagram (After EQ)')
    axes[2,1].set_xlabel('Sample within Symbol')
    axes[2,1].set_ylabel('Amplitude')
    axes[2,1].grid(True)
    
    # Plot 9: Frequency response comparison
    # Channel frequency response
    freqs = np.linspace(0, transceiver.fs/2, 512)
    h_channel = np.fft.fft(channel.impulse_response, 1024)
    h_channel = h_channel[:512]
    
    # Equalizer frequency response
    h_equalizer = np.fft.fft(equalizer.w_ffe, 1024)
    h_equalizer = h_equalizer[:512]
    
    # Combined response
    h_combined = h_channel * h_equalizer
    
    axes[2,2].plot(freqs/1e9, 20*np.log10(np.abs(h_channel) + 1e-12), 'r-', label='Channel')
    axes[2,2].plot(freqs/1e9, 20*np.log10(np.abs(h_equalizer) + 1e-12), 'b-', label='Equalizer')
    axes[2,2].plot(freqs/1e9, 20*np.log10(np.abs(h_combined) + 1e-12), 'g-', label='Combined')
    axes[2,2].set_title('Frequency Response')
    axes[2,2].set_xlabel('Frequency (GHz)')
    axes[2,2].set_ylabel('Magnitude (dB)')
    axes[2,2].legend()
    axes[2,2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print performance summary
    print("\nPerformance Summary:")
    print("=" * 50)
    print(f"SNR Improvement: {performance_metrics['snr_improvement_db']:.2f} dB")
    if performance_metrics['ber'] is not None:
        print(f"Bit Error Rate: {performance_metrics['ber']:.2e}")
    print(f"Eye Height: {performance_metrics['eye_height']:.4f}")
    print(f"Eye Opening Height: {performance_metrics['eye_opening_height']:.4f}")
    print(f"Final MSE: {performance_metrics['final_mse']:.6f}")
    print(f"FFE Taps: {len(performance_metrics['ffe_coefficients'])}")
    print(f"DFE Taps: {len(performance_metrics['dfe_coefficients'])}")

def main():
    """Main demonstration of NRZ DFE/FFE with auto-correlation adaptation"""
    
    print("NRZ DFE and FFE Coefficient Adaptation using Auto-Correlation")
    print("=" * 65)
    
    # System parameters
    data_rate = 25e9  # 25 Gbps
    samples_per_symbol = 2
    num_bits = 5000
    
    # Create transceiver
    transceiver = NRZTransceiver(data_rate, samples_per_symbol)
    
    # Generate test data
    print("Generating PRBS data...")
    tx_data = transceiver.generate_prbs(num_bits, 'prbs15')
    
    # Transmit signal
    print("Transmitting NRZ signal...")
    tx_signal = transceiver.transmit(tx_data)
    
    # Create channel models
    channel_types = ['copper_trace', 'backplane']
    
    for channel_type in channel_types:
        print(f"\n--- Testing {channel_type.replace('_', ' ').title()} Channel ---")
        
        # Create channel
        if channel_type == 'copper_trace':
            channel = ChannelModel('copper_trace', length=8, data_rate=data_rate)
        else:
            channel = ChannelModel('backplane', length=20, data_rate=data_rate)
        
        print(f"Channel impulse response length: {len(channel.impulse_response)} taps")
        
        # Apply channel effects
        print("Applying channel effects...")
        rx_signal = channel.add_channel_effects(tx_signal, add_noise=True, snr_db=25, add_jitter=True)
        
        # Test different equalizer configurations
        equalizer_configs = [
            {'ffe_taps': 13, 'dfe_taps': 2, 'name': 'FFE + DFE (2-tap)'},
            {'ffe_taps': 19, 'dfe_taps': 2, 'name': 'Extended FFE + DFE (2-tap)'}
        ]
        
        for config in equalizer_configs:
            print(f"\n  Testing {config['name']} Configuration:")
            print(f"    FFE taps: {config['ffe_taps']}, DFE taps: {config['dfe_taps']}")
            
            # Create equalizer
            equalizer = AutoCorrelationEqualizer(
                ffe_taps=config['ffe_taps'],
                dfe_taps=config['dfe_taps'],
                step_size=0.001
            )
            
            # Training parameters
            training_length = 1000  # symbols
            reference_data = tx_data[:training_length]
            
            # Receive with equalization
            eq_output, error_signal, decisions, symbol_decisions = transceiver.receive_with_equalizer(
                rx_signal, channel, equalizer, training_length, reference_data)
            
            # Analyze performance
            performance_metrics = equalizer.analyze_performance(
                rx_signal[training_length*samples_per_symbol:],
                eq_output,
                decisions,
                reference=tx_data[training_length:training_length + len(symbol_decisions)]
            )
            
            print(f"    SNR Improvement: {performance_metrics['snr_improvement_db']:.2f} dB")
            if performance_metrics['ber'] is not None:
                print(f"    BER: {performance_metrics['ber']:.2e}")
            print(f"    Final MSE: {performance_metrics['final_mse']:.6f}")
            print(f"    Eye Opening: {performance_metrics['eye_opening_height']:.4f}")
            
            # Plot results for the first configuration
            if config == equalizer_configs[0]:
                print(f"  Generating plots for {config['name']}...")
                plot_results(transceiver, channel, equalizer, tx_data, rx_signal,
                           eq_output, decisions, performance_metrics)
    
    # Advanced techniques demonstration
    print("\n--- Advanced Adaptation Techniques ---")
    
    # 1. Comparison of adaptation algorithms
    print("\n1. Adaptation Algorithm Comparison:")
    
    channel = ChannelModel('copper_trace', length=10, data_rate=data_rate)
    rx_signal = channel.add_channel_effects(tx_signal, add_noise=True, snr_db=20)
    
    adaptation_methods = ['lms', 'wiener']
    
    for method in adaptation_methods:
        print(f"\n  Testing {method.upper()} Adaptation:")
        
        equalizer = AutoCorrelationEqualizer(ffe_taps=15, dfe_taps=2, step_size=0.002)
        
        # Training phase
        training_length = 500
        train_signal = rx_signal[:training_length * samples_per_symbol]
        train_reference = np.repeat(tx_data[:training_length], samples_per_symbol)
        
        equalizer.adaptation_enabled = True
        equalizer.decision_directed = False
        
        eq_out, error_sig, decisions = equalizer.process_block(
            train_signal, train_reference, adaptation_mode=method)
        
        final_mse = np.mean(error_sig[-100:]**2)
        print(f"    Final Training MSE: {final_mse:.6f}")
        
        # Test phase
        test_signal = rx_signal[training_length * samples_per_symbol:]
        equalizer.decision_directed = True
        
        eq_test, error_test, dec_test = equalizer.process_block(test_signal)
        test_mse = np.mean(error_test**2)
        print(f"    Test MSE: {test_mse:.6f}")
    
    # 2. Step size optimization
    print("\n2. Step Size Optimization:")
    
    step_sizes = [0.0001, 0.001, 0.01, 0.1]
    convergence_results = {}
    
    for step_size in step_sizes:
        equalizer = AutoCorrelationEqualizer(ffe_taps=15, dfe_taps=2, step_size=step_size)
        
        # Quick training test
        train_signal = rx_signal[:1000]
        train_reference = np.repeat(tx_data[:500], samples_per_symbol)
        
        equalizer.adaptation_enabled = True
        equalizer.decision_directed = False
        
        eq_out, error_sig, _ = equalizer.process_block(train_signal, train_reference)
        
        # Measure convergence speed (samples to reach 90% of final performance)
        final_mse = np.mean(error_sig[-50:]**2)
        target_mse = final_mse * 1.1  # 10% above final
        
        convergence_point = len(error_sig)
        for i, mse_val in enumerate(equalizer.mse_history):
            if mse_val <= target_mse:
                convergence_point = i * 10  # Account for subsampling
                break
        
        convergence_results[step_size] = {
            'final_mse': final_mse,
            'convergence_samples': convergence_point
        }
        
        print(f"    μ = {step_size}: Final MSE = {final_mse:.6f}, Convergence = {convergence_point} samples")
    
    # 3. Time-varying channel adaptation
    print("\n3. Time-Varying Channel Adaptation:")
    
    # Create time-varying channel
    base_channel = ChannelModel('copper_trace', length=8, data_rate=data_rate)
    
    # Simulate slow channel variations
    num_blocks = 10
    block_size = 500
    
    equalizer_adaptive = AutoCorrelationEqualizer(ffe_taps=15, dfe_taps=2, step_size=0.005)
    equalizer_static = AutoCorrelationEqualizer(ffe_taps=15, dfe_taps=2, step_size=0.001)
    
    adaptive_mse_history = []
    static_mse_history = []
    
    # Initial training
    initial_rx = base_channel.add_channel_effects(tx_signal[:block_size], add_noise=True, snr_db=25)
    initial_ref = np.repeat(tx_data[:block_size//2], samples_per_symbol)
    
    # Train both equalizers initially
    equalizer_adaptive.process_block(initial_rx, initial_ref)
    equalizer_static.process_block(initial_rx, initial_ref)
    
    # Disable adaptation for static equalizer
    equalizer_static.adaptation_enabled = False
    
    for block in range(num_blocks):
        # Modify channel characteristics
        variation_factor = 1 + 0.1 * np.sin(block * 0.5)  # ±10% variation
        modified_channel = ChannelModel('copper_trace', length=8*variation_factor, data_rate=data_rate)
        
        # Generate block data
        block_start = block_size + block * block_size
        block_tx = tx_signal[block_start:block_start + block_size]
        block_rx = modified_channel.add_channel_effects(block_tx, add_noise=True, snr_db=25)
        
        # Process with both equalizers
        equalizer_adaptive.decision_directed = True
        eq_out_adaptive, error_adaptive, _ = equalizer_adaptive.process_block(block_rx)
        
        equalizer_static.decision_directed = True
        eq_out_static, error_static, _ = equalizer_static.process_block(block_rx)
        
        # Record performance
        adaptive_mse = np.mean(error_adaptive**2)
        static_mse = np.mean(error_static**2)
        
        adaptive_mse_history.append(adaptive_mse)
        static_mse_history.append(static_mse)
        
        print(f"    Block {block+1}: Adaptive MSE = {adaptive_mse:.6f}, Static MSE = {static_mse:.6f}")
    
    # Plot time-varying results
    plt.figure(figsize=(12, 8))
    
    # MSE comparison
    plt.subplot(2, 2, 1)
    plt.plot(adaptive_mse_history, 'g-o', label='Adaptive Equalizer')
    plt.plot(static_mse_history, 'r-s', label='Static Equalizer')
    plt.xlabel('Block Number')
    plt.ylabel('MSE')
    plt.title('Time-Varying Channel Performance')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Coefficient evolution
    plt.subplot(2, 2, 2)
    ffe_evolution = np.array(equalizer_adaptive.coefficient_history['ffe'])
    if len(ffe_evolution) > 0:
        for tap in [0, len(equalizer_adaptive.w_ffe)//2, -1]:  # Show first, middle, last taps
            plt.plot(ffe_evolution[:, tap], label=f'FFE Tap {tap}')
    plt.xlabel('Adaptation Step')
    plt.ylabel('Coefficient Value')
    plt.title('FFE Coefficient Evolution')
    plt.legend()
    plt.grid(True)
    
    # Step size sensitivity
    plt.subplot(2, 2, 3)
    step_sizes_plot = list(convergence_results.keys())
    final_mses = [convergence_results[mu]['final_mse'] for mu in step_sizes_plot]
    convergence_speeds = [convergence_results[mu]['convergence_samples'] for mu in step_sizes_plot]
    
    plt.semilogx(step_sizes_plot, final_mses, 'bo-')
    plt.xlabel('Step Size (μ)')
    plt.ylabel('Final MSE')
    plt.title('Step Size vs Final MSE')
    plt.grid(True)
    
    # Convergence speed
    plt.subplot(2, 2, 4)
    plt.semilogx(step_sizes_plot, convergence_speeds, 'ro-')
    plt.xlabel('Step Size (μ)')
    plt.ylabel('Convergence Time (samples)')
    plt.title('Step Size vs Convergence Speed')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 4. Multi-level signaling (PAM4)
    print("\n4. PAM4 Signaling Extension:")
    
    # Generate PAM4 data
    pam4_symbols = np.random.randint(0, 4, num_bits//2)
    pam4_levels = np.array([-3, -1, 1, 3])
    pam4_signal = pam4_levels[pam4_symbols]
    
    # Transmit PAM4
    tx_pam4 = transceiver.transmit(pam4_signal)
    
    # Channel effects
    channel_pam4 = ChannelModel('copper_trace', length=6, data_rate=data_rate)
    rx_pam4 = channel_pam4.add_channel_effects(tx_pam4, add_noise=True, snr_db=30)
    
    # PAM4 equalizer (modified slicer)
    class PAM4Equalizer(AutoCorrelationEqualizer):
        def hard_decision(self, x):
            """PAM4 hard decision slicer"""
            if x < -2:
                return -3
            elif x < 0:
                return -1
            elif x < 2:
                return 1
            else:
                return 3
    
    equalizer_pam4 = PAM4Equalizer(ffe_taps=21, dfe_taps=2, step_size=0.002)
    
    # Training
    train_length_pam4 = 500
    train_rx_pam4 = rx_pam4[:train_length_pam4 * samples_per_symbol]
    train_ref_pam4 = np.repeat(pam4_signal[:train_length_pam4], samples_per_symbol)
    
    equalizer_pam4.adaptation_enabled = True
    equalizer_pam4.decision_directed = False
    
    eq_pam4_train, error_pam4_train, _ = equalizer_pam4.process_block(train_rx_pam4, train_ref_pam4)
    
    # Test
    test_rx_pam4 = rx_pam4[train_length_pam4 * samples_per_symbol:]
    equalizer_pam4.decision_directed = True
    
    eq_pam4_test, error_pam4_test, dec_pam4_test = equalizer_pam4.process_block(test_rx_pam4)
    
    # Calculate PAM4 performance
    symbol_decisions_pam4 = dec_pam4_test[::samples_per_symbol]
    reference_pam4 = pam4_signal[train_length_pam4:train_length_pam4 + len(symbol_decisions_pam4)]
    
    symbol_errors = np.sum(symbol_decisions_pam4 != reference_pam4)
    ser_pam4 = symbol_errors / len(symbol_decisions_pam4)
    
    print(f"    PAM4 Training MSE: {np.mean(error_pam4_train**2):.6f}")
    print(f"    PAM4 Test MSE: {np.mean(error_pam4_test**2):.6f}")
    print(f"    PAM4 Symbol Error Rate: {ser_pam4:.2e}")
    
    print(f"\nDemonstration completed!")
    print("\nKey Features Demonstrated:")
    print("- Auto-correlation based coefficient adaptation")
    print("- FFE and DFE structures for ISI cancellation")
    print("- Multiple adaptation algorithms (LMS, Wiener)")
    print("- Step size optimization for convergence")
    print("- Time-varying channel tracking")
    print("- Multi-level signaling (PAM4) support")
    print("- Comprehensive performance analysis")

if __name__ == "__main__":
    main()