import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class TimeInterleavedSARADC:
    def __init__(self, num_channels=32, bits=12, fs=1e9, reference_voltage=1.0):
        """
        Initialize 32x Time-Interleaved SAR ADC
        
        Parameters:
        - num_channels: Number of interleaved channels (32)
        - bits: ADC resolution in bits
        - fs: Sampling frequency (Hz)
        - reference_voltage: ADC reference voltage
        """
        self.num_channels = num_channels
        self.bits = bits
        self.fs = fs
        self.vref = reference_voltage
        self.lsb = reference_voltage / (2**bits)
        
        # Initialize timing skews (in seconds)
        self.timing_skews = np.zeros(num_channels)
        self.estimated_skews = np.zeros(num_channels)
        
        # Channel sampling period
        self.ts_channel = num_channels / fs
        
    def add_timing_skew(self, skew_std=1e-12):
        """Add random timing skews to channels"""
        self.timing_skews = np.random.normal(0, skew_std, self.num_channels)
        # First channel is reference (no skew)
        self.timing_skews[0] = 0
        
    def generate_test_signal(self, freq, amplitude, duration, noise_level=0.001):
        """Generate test signal for calibration"""
        t = np.arange(0, duration, 1/self.fs)
        signal = amplitude * np.sin(2 * np.pi * freq * t)
        noise = np.random.normal(0, noise_level * amplitude, len(t))
        return t, signal + noise
    
    def sample_with_skew(self, t, signal):
        """Sample signal with timing skews applied"""
        samples = []
        
        for i in range(len(t)):
            channel = i % self.num_channels
            # Apply timing skew
            skewed_time = t[i] + self.timing_skews[channel]
            
            # Interpolate signal at skewed time
            if skewed_time >= 0 and skewed_time <= t[-1]:
                skewed_sample = np.interp(skewed_time, t, signal)
            else:
                skewed_sample = 0
                
            # Quantize to ADC resolution
            quantized = np.round(skewed_sample / self.lsb) * self.lsb
            quantized = np.clip(quantized, -self.vref, self.vref - self.lsb)
            
            samples.append(quantized)
            
        return np.array(samples)
    
    def extract_channel_data(self, samples):
        """Extract individual channel data from interleaved samples"""
        channel_data = []
        for ch in range(self.num_channels):
            ch_samples = samples[ch::self.num_channels]
            channel_data.append(ch_samples)
        return channel_data
    
    def correlation_based_calibration(self, samples, reference_freq):
        """
        Correlation-based timing skew estimation
        Uses cross-correlation between channels to estimate timing offsets
        """
        channel_data = self.extract_channel_data(samples)
        
        # Use channel 0 as reference
        ref_channel = channel_data[0]
        
        estimated_skews = np.zeros(self.num_channels)
        
        for ch in range(1, self.num_channels):
            # Cross-correlate with reference channel
            correlation = np.correlate(channel_data[ch], ref_channel, mode='full')
            
            # Find peak correlation
            peak_idx = np.argmax(np.abs(correlation))
            offset_samples = peak_idx - (len(ref_channel) - 1)
            
            # Convert to time offset
            time_offset = offset_samples * self.ts_channel
            estimated_skews[ch] = time_offset
            
        self.estimated_skews = estimated_skews
        return estimated_skews
    
    def frequency_domain_calibration(self, samples, test_freq):
        """
        Frequency domain calibration using spectral analysis
        Estimates timing skew from spurious tones in the spectrum
        """
        N = len(samples)
        fft_samples = np.fft.fft(samples)
        freqs = np.fft.fftfreq(N, 1/self.fs)
        
        # Find test frequency bin
        test_bin = np.argmin(np.abs(freqs - test_freq))
        
        # Extract channel phases at test frequency
        channel_data = self.extract_channel_data(samples)
        channel_phases = np.zeros(self.num_channels)
        
        for ch in range(self.num_channels):
            ch_fft = np.fft.fft(channel_data[ch])
            ch_freqs = np.fft.fftfreq(len(channel_data[ch]), self.ts_channel)
            ch_test_bin = np.argmin(np.abs(ch_freqs - test_freq))
            
            # Extract phase
            channel_phases[ch] = np.angle(ch_fft[ch_test_bin])
        
        # Convert phase differences to timing skews
        estimated_skews = np.zeros(self.num_channels)
        ref_phase = channel_phases[0]
        
        for ch in range(1, self.num_channels):
            phase_diff = channel_phases[ch] - ref_phase
            # Unwrap phase
            phase_diff = np.angle(np.exp(1j * phase_diff))
            # Convert to timing skew
            estimated_skews[ch] = -phase_diff / (2 * np.pi * test_freq)
            
        self.estimated_skews = estimated_skews
        return estimated_skews
    
    def adaptive_calibration(self, samples, test_freq, max_iterations=10):
        """
        Adaptive calibration algorithm that iteratively refines skew estimates
        """
        def cost_function(skews):
            # Apply skew correction and measure spectral purity
            corrected_samples = self.apply_skew_correction(samples, skews)
            fft_corrected = np.fft.fft(corrected_samples)
            
            # Cost is the sum of spurious tone power
            N = len(fft_corrected)
            freqs = np.fft.fftfreq(N, 1/self.fs)
            
            # Find spurious tones (multiples of fs/M)
            spurious_power = 0
            for k in range(1, 5):  # Check first few spurious harmonics
                spur_freq = k * self.fs / self.num_channels
                spur_bin = np.argmin(np.abs(freqs - spur_freq))
                spurious_power += np.abs(fft_corrected[spur_bin])**2
                
            return spurious_power
        
        # Initial guess from frequency domain method
        initial_skews = self.frequency_domain_calibration(samples, test_freq)
        
        # Optimize
        result = minimize(cost_function, initial_skews, method='BFGS')
        
        self.estimated_skews = result.x
        return result.x
    
    def apply_skew_correction(self, samples, skew_corrections):
        """Apply timing skew corrections to samples"""
        corrected_samples = np.zeros_like(samples)
        t = np.arange(len(samples)) / self.fs
        
        for i in range(len(samples)):
            channel = i % self.num_channels
            # Correct timing by interpolating at corrected time
            corrected_time = t[i] - skew_corrections[channel]
            
            if corrected_time >= 0 and corrected_time <= t[-1]:
                corrected_samples[i] = np.interp(corrected_time, t, samples)
            else:
                corrected_samples[i] = samples[i]
                
        return corrected_samples
    
    def analyze_performance(self, original_samples, corrected_samples, test_freq):
        """Analyze calibration performance"""
        def calculate_sndr(samples, signal_freq):
            N = len(samples)
            fft_data = np.fft.fft(samples)
            freqs = np.fft.fftfreq(N, 1/self.fs)
            
            # Find signal bin
            signal_bin = np.argmin(np.abs(freqs - signal_freq))
            signal_power = np.abs(fft_data[signal_bin])**2
            
            # Calculate noise + distortion power
            total_power = np.sum(np.abs(fft_data)**2)
            nd_power = total_power - signal_power
            
            # SNDR in dB
            sndr = 10 * np.log10(signal_power / nd_power)
            return sndr
        
        # Calculate SNDR before and after calibration
        sndr_before = calculate_sndr(original_samples, test_freq)
        sndr_after = calculate_sndr(corrected_samples, test_freq)
        
        # Calculate timing skew estimation error
        skew_error_rms = np.sqrt(np.mean((self.estimated_skews - self.timing_skews)**2))
        
        return {
            'sndr_before': sndr_before,
            'sndr_after': sndr_after,
            'sndr_improvement': sndr_after - sndr_before,
            'skew_error_rms': skew_error_rms,
            'true_skews': self.timing_skews,
            'estimated_skews': self.estimated_skews
        }
    
    def plot_results(self, original_samples, corrected_samples, test_freq):
        """Plot calibration results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Time domain comparison
        t_plot = np.arange(min(1000, len(original_samples))) / self.fs * 1e9  # ns
        axes[0,0].plot(t_plot, original_samples[:len(t_plot)], 'b-', alpha=0.7, label='Before calibration')
        axes[0,0].plot(t_plot, corrected_samples[:len(t_plot)], 'r-', alpha=0.7, label='After calibration')
        axes[0,0].set_xlabel('Time (ns)')
        axes[0,0].set_ylabel('Amplitude (V)')
        axes[0,0].set_title('Time Domain Comparison')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Plot 2: Frequency domain comparison
        N = len(original_samples)
        freqs = np.fft.fftfreq(N, 1/self.fs) / 1e6  # MHz
        fft_orig = 20 * np.log10(np.abs(np.fft.fft(original_samples)) + 1e-12)
        fft_corr = 20 * np.log10(np.abs(np.fft.fft(corrected_samples)) + 1e-12)
        
        # Plot positive frequencies only
        pos_idx = freqs >= 0
        axes[0,1].plot(freqs[pos_idx], fft_orig[pos_idx], 'b-', alpha=0.7, label='Before calibration')
        axes[0,1].plot(freqs[pos_idx], fft_corr[pos_idx], 'r-', alpha=0.7, label='After calibration')
        axes[0,1].set_xlabel('Frequency (MHz)')
        axes[0,1].set_ylabel('Magnitude (dB)')
        axes[0,1].set_title('Frequency Domain Comparison')
        axes[0,1].legend()
        axes[0,1].grid(True)
        axes[0,1].set_xlim(0, self.fs/2/1e6)
        
        # Plot 3: Timing skew comparison
        channels = np.arange(self.num_channels)
        axes[1,0].bar(channels - 0.2, self.timing_skews * 1e12, 0.4, 
                     label='True skews', alpha=0.7, color='blue')
        axes[1,0].bar(channels + 0.2, self.estimated_skews * 1e12, 0.4, 
                     label='Estimated skews', alpha=0.7, color='red')
        axes[1,0].set_xlabel('Channel')
        axes[1,0].set_ylabel('Timing Skew (ps)')
        axes[1,0].set_title('Timing Skew Calibration')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Plot 4: Skew estimation error
        skew_error = (self.estimated_skews - self.timing_skews) * 1e12
        axes[1,1].bar(channels, skew_error, alpha=0.7, color='green')
        axes[1,1].set_xlabel('Channel')
        axes[1,1].set_ylabel('Estimation Error (ps)')
        axes[1,1].set_title('Skew Estimation Error')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage and demonstration
def main():
    # Initialize 32x Time-Interleaved SAR ADC
    adc = TimeInterleavedSARADC(num_channels=32, bits=12, fs=1e9, reference_voltage=1.0)
    
    # Add timing skews (1ps RMS)
    adc.add_timing_skew(skew_std=1e-12)
    
    print("32x Time-Interleaved SAR ADC Timing Skew Calibration")
    print("=" * 55)
    print(f"Number of channels: {adc.num_channels}")
    print(f"Sampling frequency: {adc.fs/1e9:.1f} GHz")
    print(f"Channel sampling rate: {adc.fs/adc.num_channels/1e6:.1f} MHz")
    print(f"Added timing skew RMS: {np.std(adc.timing_skews)*1e12:.2f} ps")
    
    # Generate test signal (100 MHz sine wave)
    test_freq = 100e6  # 100 MHz
    duration = 1e-6    # 1 μs
    t, test_signal = adc.generate_test_signal(test_freq, 0.8, duration, noise_level=0.01)
    
    # Sample with timing skews
    samples_with_skew = adc.sample_with_skew(t, test_signal)
    
    print(f"\nTest signal: {test_freq/1e6:.0f} MHz, {duration*1e6:.1f} μs duration")
    print(f"Number of samples: {len(samples_with_skew)}")
    
    # Perform calibration using different methods
    print("\nCalibration Methods:")
    print("-" * 20)
    
    # Method 1: Correlation-based
    skews_corr = adc.correlation_based_calibration(samples_with_skew, test_freq)
    corrected_samples_corr = adc.apply_skew_correction(samples_with_skew, skews_corr)
    perf_corr = adc.analyze_performance(samples_with_skew, corrected_samples_corr, test_freq)
    
    print(f"1. Correlation-based:")
    print(f"   SNDR improvement: {perf_corr['sndr_improvement']:.1f} dB")
    print(f"   Skew estimation RMS error: {perf_corr['skew_error_rms']*1e12:.2f} ps")
    
    # Method 2: Frequency domain
    skews_freq = adc.frequency_domain_calibration(samples_with_skew, test_freq)
    corrected_samples_freq = adc.apply_skew_correction(samples_with_skew, skews_freq)
    perf_freq = adc.analyze_performance(samples_with_skew, corrected_samples_freq, test_freq)
    
    print(f"2. Frequency domain:")
    print(f"   SNDR improvement: {perf_freq['sndr_improvement']:.1f} dB")
    print(f"   Skew estimation RMS error: {perf_freq['skew_error_rms']*1e12:.2f} ps")
    
    # Method 3: Adaptive calibration
    skews_adaptive = adc.adaptive_calibration(samples_with_skew, test_freq)
    corrected_samples_adaptive = adc.apply_skew_correction(samples_with_skew, skews_adaptive)
    perf_adaptive = adc.analyze_performance(samples_with_skew, corrected_samples_adaptive, test_freq)
    
    print(f"3. Adaptive calibration:")
    print(f"   SNDR improvement: {perf_adaptive['sndr_improvement']:.1f} dB")
    print(f"   Skew estimation RMS error: {perf_adaptive['skew_error_rms']*1e12:.2f} ps")
    
    # Plot results for the best performing method
    best_method = max([perf_corr, perf_freq, perf_adaptive], 
                     key=lambda x: x['sndr_improvement'])
    
    if best_method == perf_adaptive:
        print(f"\nBest method: Adaptive calibration")
        adc.plot_results(samples_with_skew, corrected_samples_adaptive, test_freq)
    elif best_method == perf_freq:
        print(f"\nBest method: Frequency domain")
        adc.plot_results(samples_with_skew, corrected_samples_freq, test_freq)
    else:
        print(f"\nBest method: Correlation-based")
        adc.plot_results(samples_with_skew, corrected_samples_corr, test_freq)

if __name__ == "__main__":
    main()