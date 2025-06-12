import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class LMSEchoCanceller:
    def __init__(self, filter_length=128, step_size=0.01):
        """
        Initialize LMS Echo Canceller
        
        Parameters:
        filter_length: Length of adaptive filter (taps)
        step_size: LMS step size (mu) - controls convergence speed vs stability
        """
        self.filter_length = filter_length
        self.step_size = step_size
        self.weights = np.zeros(filter_length)  # Adaptive filter weights
        self.input_buffer = np.zeros(filter_length)  # Input signal buffer
        
    def update(self, input_signal, desired_signal):
        """
        Single iteration of LMS algorithm
        
        Parameters:
        input_signal: Current input sample (near-end signal)
        desired_signal: Desired output (far-end signal with echo)
        
        Returns:
        output_signal: Echo-cancelled signal
        error_signal: Error between desired and estimated echo
        """
        # Shift input buffer and add new sample
        self.input_buffer[1:] = self.input_buffer[:-1]
        self.input_buffer[0] = input_signal
        
        # Calculate filter output (estimated echo)
        estimated_echo = np.dot(self.weights, self.input_buffer)
        
        # Calculate error (echo-cancelled signal)
        error_signal = desired_signal - estimated_echo
        
        # Update filter weights using LMS algorithm
        self.weights += self.step_size * error_signal * self.input_buffer
        
        return error_signal, estimated_echo
    
    def process_block(self, input_block, desired_block):
        """
        Process a block of samples
        
        Parameters:
        input_block: Array of input samples
        desired_block: Array of desired output samples
        
        Returns:
        output_block: Echo-cancelled signal block
        error_power: Mean squared error for the block
        """
        output_block = np.zeros(len(input_block))
        estimated_echo_block = np.zeros(len(input_block))
        
        for i in range(len(input_block)):
            output_block[i], estimated_echo_block[i] = self.update(
                input_block[i], desired_block[i]
            )
        
        # Calculate error power for convergence monitoring
        error_power = np.mean(output_block**2)
        
        return output_block, estimated_echo_block, error_power

class EthernetEchoSimulator:
    """Simulate echo characteristics typical in Ethernet applications"""
    
    def __init__(self, echo_delay=10, echo_attenuation=0.3, noise_level=0.01):
        """
        Parameters:
        echo_delay: Echo delay in samples
        echo_attenuation: Echo amplitude relative to original (0-1)
        noise_level: Background noise level
        """
        self.echo_delay = echo_delay
        self.echo_attenuation = echo_attenuation
        self.noise_level = noise_level
        self.delay_line = np.zeros(echo_delay)
    
    def add_echo(self, signal):
        """Add simulated echo to input signal"""
        echoed_signal = np.zeros_like(signal)
        
        for i, sample in enumerate(signal):
            # Add delayed and attenuated echo
            if i >= self.echo_delay:
                echo_component = signal[i - self.echo_delay] * self.echo_attenuation
            else:
                echo_component = 0
            
            # Add noise
            noise = np.random.normal(0, self.noise_level)
            
            echoed_signal[i] = sample + echo_component + noise
            
        return echoed_signal

def ethernet_echo_cancellation_demo():
    """Demonstrate LMS echo cancellation for Ethernet application"""
    
    # Simulation parameters
    fs = 125e6  # 125 MHz sampling rate (typical for Gigabit Ethernet)
    duration = 0.001  # 1ms simulation
    samples = int(fs * duration)
    
    # Generate test signal (simulated Ethernet data)
    t = np.linspace(0, duration, samples)
    
    # Create pseudo-random binary sequence (typical Ethernet data pattern)
    np.random.seed(42)  # For reproducible results
    data_bits = np.random.choice([-1, 1], size=samples//10)
    
    # Upsample and shape the signal (simplified NRZ encoding)
    upsampling_factor = 10
    near_end_signal = np.repeat(data_bits, upsampling_factor)
    
    # Add some band-limiting (typical in real systems)
    b, a = signal.butter(4, 0.4, 'low')  # 4th order Butterworth filter
    near_end_signal = signal.filtfilt(b, a, near_end_signal)
    
    # Create echo simulator
    echo_sim = EthernetEchoSimulator(
        echo_delay=50,      # 50 sample delay (~400ps at 125MHz)
        echo_attenuation=0.2,  # -14dB echo
        noise_level=0.05    # Background noise
    )
    
    # Add echo to create far-end received signal
    far_end_signal = echo_sim.add_echo(near_end_signal)
    
    # Initialize LMS echo canceller
    echo_canceller = LMSEchoCanceller(
        filter_length=128,
        step_size=0.001  # Conservative step size for stability
    )
    
    # Process signal through echo canceller
    cancelled_signal, estimated_echo, error_power = echo_canceller.process_block(
        near_end_signal, far_end_signal
    )
    
    # Calculate performance metrics
    echo_suppression_db = 10 * np.log10(
        np.mean(far_end_signal**2) / np.mean(cancelled_signal**2)
    )
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Time domain plots
    time_ms = t * 1000  # Convert to milliseconds
    
    plt.subplot(2, 3, 1)
    plt.plot(time_ms, near_end_signal)
    plt.title('Near-End Signal (Transmitted)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(time_ms, far_end_signal, label='With Echo', alpha=0.7)
    plt.plot(time_ms, cancelled_signal, label='Echo Cancelled', alpha=0.7)
    plt.title('Far-End Signal')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(time_ms, estimated_echo)
    plt.title('Estimated Echo')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Frequency domain analysis
    freqs = np.fft.fftfreq(len(near_end_signal), 1/fs)
    freqs_mhz = freqs[:len(freqs)//2] / 1e6
    
    near_end_fft = np.fft.fft(near_end_signal)
    far_end_fft = np.fft.fft(far_end_signal)
    cancelled_fft = np.fft.fft(cancelled_signal)
    
    plt.subplot(2, 3, 4)
    plt.semilogy(freqs_mhz, np.abs(far_end_fft[:len(freqs)//2]), 
                 label='With Echo', alpha=0.7)
    plt.semilogy(freqs_mhz, np.abs(cancelled_fft[:len(freqs)//2]), 
                 label='Echo Cancelled', alpha=0.7)
    plt.title('Frequency Domain')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    
    # Convergence plot
    plt.subplot(2, 3, 5)
    block_size = 100
    convergence_data = []
    temp_canceller = LMSEchoCanceller(filter_length=128, step_size=0.001)
    
    for i in range(0, len(near_end_signal), block_size):
        end_idx = min(i + block_size, len(near_end_signal))
        _, _, err_power = temp_canceller.process_block(
            near_end_signal[i:end_idx], far_end_signal[i:end_idx]
        )
        convergence_data.append(10 * np.log10(err_power + 1e-10))
    
    plt.plot(convergence_data)
    plt.title('LMS Convergence')
    plt.xlabel('Block Number')
    plt.ylabel('Error Power (dB)')
    plt.grid(True)
    
    # Filter coefficients
    plt.subplot(2, 3, 6)
    plt.plot(echo_canceller.weights)
    plt.title('Final Filter Coefficients')
    plt.xlabel('Tap Number')
    plt.ylabel('Weight Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Echo Suppression: {echo_suppression_db:.2f} dB")
    print(f"Final Error Power: {10*np.log10(error_power):.2f} dB")
    print(f"Filter Length: {echo_canceller.filter_length} taps")
    print(f"Step Size: {echo_canceller.step_size}")

# Alternative implementation with better numerical stability
class NormalizedLMSEchoCanceller(LMSEchoCanceller):
    """NLMS (Normalized LMS) variant with better convergence properties"""
    
    def __init__(self, filter_length=128, step_size=0.5, regularization=1e-6):
        super().__init__(filter_length, step_size)
        self.regularization = regularization
    
    def update(self, input_signal, desired_signal):
        # Shift input buffer
        self.input_buffer[1:] = self.input_buffer[:-1]
        self.input_buffer[0] = input_signal
        
        # Calculate filter output
        estimated_echo = np.dot(self.weights, self.input_buffer)
        error_signal = desired_signal - estimated_echo
        
        # Normalized LMS update (better for varying input power)
        input_power = np.dot(self.input_buffer, self.input_buffer)
        normalized_step = self.step_size / (input_power + self.regularization)
        
        self.weights += normalized_step * error_signal * self.input_buffer
        
        return error_signal, estimated_echo

if __name__ == "__main__":
    # Run the demonstration
    ethernet_echo_cancellation_demo()
    
    # Example of using the canceller in a streaming application
    print("\nStreaming Example:")
    canceller = NormalizedLMSEchoCanceller(filter_length=64, step_size=0.8)
    
    # Simulate processing samples one by one
    for i in range(10):
        input_sample = np.random.randn()  # Simulated input
        desired_sample = input_sample + 0.3 * np.random.randn()  # With echo/noise
        
        output, estimated_echo = canceller.update(input_sample, desired_sample)
        print(f"Sample {i}: Input={input_sample:.3f}, "
              f"Desired={desired_sample:.3f}, Output={output:.3f}")