import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize
from scipy.fft import fft, fftfreq, ifft
import time

class TimeInterleavedSARADC:
    """
    64x Time-Interleaved SAR ADC with Timing Skew Calibration.
    Implements multiple calibration algorithms for high-speed ADC systems.
    """
    
    def __init__(self, num_channels=64, fs_total=10e9, resolution=12, 
                 nominal_delay=0):
        """
        Initialize TI-ADC system.
        
        Parameters:
        -----------
        num_channels : int
            Number of interleaved channels (64 default)
        fs_total : float
            Total sampling rate (10 GSps default)
        resolution : int
            ADC resolution in bits
        nominal_delay : float
            Nominal sampling delay between channels (seconds)
        """
        self.M = num_channels
        self.fs_total = fs_total
        self.fs_channel = fs_total / num_channels  # Per-channel rate
        self.Ts = 1 / fs_total  # Total sampling period
        self.resolution = resolution
        self.nominal_delay = nominal_delay
        
        # Initialize timing skews (in seconds)
        self.timing_skews = np.zeros(self.M)
        self.calibrated_skews = np.zeros(self.M)
        
        # Calibration parameters
        self.calibration_history = {
            'skew_estimates': [],
            'spurious_power': [],
            'snr': [],
            'sfdr': []
        }
        
        # ADC non-idealities
        self.gain_mismatches = np.ones(self.M)
        self.offset_mismatches = np.zeros(self.M)
        
    def add_timing_skews(self, skew_std=5e-12, deterministic_pattern=None):
        """
        Add timing skews to ADC channels.
        
        Parameters:
        -----------
        skew_std : float
            Standard deviation of random skews (5ps default)
        deterministic_pattern : array or None
            Specific skew pattern to apply
        """
        if deterministic_pattern is not None:
            self.timing_skews = deterministic_pattern
        else:
            # Random Gaussian skews with first channel as reference
            self.timing_skews = np.random.normal(0, skew_std, self.M)
            self.timing_skews[0] = 0  # Reference channel
            
        print(f"Added timing skews - RMS: {np.std(self.timing_skews)*1e12:.2f} ps")
        print(f"Max skew: {np.max(np.abs(self.timing_skews))*1e12:.2f} ps")
        
    def add_gain_offset_mismatch(self, gain_std=0.01, offset_std=0.001):
        """Add gain and offset mismatches."""
        self.gain_mismatches = 1 + np.random.normal(0, gain_std, self.M)
        self.offset_mismatches = np.random.normal(0, offset_std, self.M)
        
    def sample_signal(self, t, signal_func, include_skews=True):
        """
        Sample input signal with TI-ADC including timing skews.
        
        Parameters:
        -----------
        t : array
            Time vector
        signal_func : callable
            Function that generates signal values
        include_skews : bool
            Whether to include timing skews
        
        Returns:
        --------
        samples : array
            Interleaved samples
        channel_samples : array
            Samples organized by channel
        """
        N = len(t)
        samples = np.zeros(N)
        channel_samples = np.zeros((self.M, N // self.M + 1))
        
        for i in range(N):
            ch = i % self.M  # Channel index
            
            # Apply timing skew
            if include_skews:
                t_sample = t[i] + self.timing_skews[ch]
            else:
                t_sample = t[i]
            
            # Sample signal
            sample = signal_func(t_sample)
            
            # Apply gain and offset
            sample = self.gain_mismatches[ch] * sample + self.offset_mismatches[ch]
            
            # Quantization
            sample = self.quantize(sample)
            
            samples[i] = sample
            channel_samples[ch, i // self.M] = sample
            
        return samples, channel_samples
    
    def quantize(self, signal):
        """Quantize signal to ADC resolution."""
        # Assume full-scale range of [-1, 1]
        levels = 2**self.resolution
        lsb = 2.0 / levels
        
        # Clip to range
        signal = np.clip(signal, -1, 1)
        
        # Quantize
        quantized = np.round(signal / lsb) * lsb
        
        return quantized
    
    def correlation_based_calibration(self, samples, ref_channel=0, 
                                    window_size=1024, overlap=0.5):
        """
        Correlation-based timing skew estimation.
        Uses cross-correlation between channels to estimate relative delays.
        
        Parameters:
        -----------
        samples : array
            Input samples (interleaved)
        ref_channel : int
            Reference channel index
        window_size : int
            Window size for correlation
        overlap : float
            Overlap fraction between windows
        
        Returns:
        --------
        skew_estimates : array
            Estimated timing skews for each channel
        """
        print("\n=== Correlation-Based Calibration ===")
        
        # Reorganize samples by channel
        N = len(samples)
        channel_data = []
        for ch in range(self.M):
            ch_samples = samples[ch::self.M]
            channel_data.append(ch_samples)
        
        skew_estimates = np.zeros(self.M)
        
        # Reference channel data
        ref_data = channel_data[ref_channel]
        
        # Estimate skew for each channel relative to reference
        for ch in range(self.M):
            if ch == ref_channel:
                continue
                
            # Cross-correlation with reference
            correlation = signal.correlate(channel_data[ch][:window_size], 
                                         ref_data[:window_size], mode='full')
            
            # Find peak
            peak_idx = np.argmax(np.abs(correlation))
            delay_samples = peak_idx - (window_size - 1)
            
            # Convert to time skew
            # Account for the fact that channels sample at fs/M rate
            skew_estimates[ch] = delay_samples * self.M / self.fs_total
            
        print(f"Estimated skews - RMS: {np.std(skew_estimates)*1e12:.2f} ps")
        
        self.calibrated_skews = skew_estimates
        return skew_estimates
    
    def sine_fit_calibration(self, samples, signal_freq, num_periods=10):
        """
        Sine-fit based timing skew calibration.
        Fits sinusoidal model to each channel's data.
        
        Parameters:
        -----------
        samples : array
            Input samples (must be from a sine wave input)
        signal_freq : float
            Input sine wave frequency
        num_periods : int
            Number of periods to use for fitting
        
        Returns:
        --------
        skew_estimates : array
            Estimated timing skews
        """
        print("\n=== Sine-Fit Calibration ===")
        
        # Samples per period
        samples_per_period = int(self.fs_total / signal_freq)
        fit_length = num_periods * samples_per_period
        
        if fit_length > len(samples):
            fit_length = len(samples)
        
        skew_estimates = np.zeros(self.M)
        omega = 2 * np.pi * signal_freq
        
        # Fit each channel
        for ch in range(self.M):
            # Extract channel samples
            ch_samples = samples[ch:fit_length:self.M]
            ch_times = np.arange(len(ch_samples)) * self.M / self.fs_total
            
            # Initial parameter guess [amplitude, phase, offset]
            p0 = [np.std(ch_samples) * np.sqrt(2), 0, np.mean(ch_samples)]
            
            # Sine fitting function
            def sine_func(t, A, phi, C):
                return A * np.sin(omega * t + phi) + C
            
            try:
                # Least squares fit
                popt, _ = optimize.curve_fit(sine_func, ch_times, ch_samples, p0=p0)
                
                # Extract phase
                phase = popt[1]
                
                # Convert phase to timing skew
                skew_estimates[ch] = -phase / omega
                
            except:
                print(f"Fitting failed for channel {ch}")
                skew_estimates[ch] = 0
        
        # Remove common mode (relative to channel 0)
        skew_estimates -= skew_estimates[0]
        
        print(f"Estimated skews - RMS: {np.std(skew_estimates)*1e12:.2f} ps")
        
        self.calibrated_skews = skew_estimates
        return skew_estimates
    
    def spectral_calibration(self, samples, target_bin=None):
        """
        Spectral-based calibration using spurious tone minimization.
        
        Parameters:
        -----------
        samples : array
            Input samples
        target_bin : int or None
            Specific frequency bin to minimize (auto-detect if None)
        
        Returns:
        --------
        skew_estimates : array
            Estimated timing skews
        """
        print("\n=== Spectral Calibration ===")
        
        N = len(samples)
        
        # Compute spectrum
        spectrum = fft(samples * signal.windows.hann(N))
        freqs = fftfreq(N, 1/self.fs_total)
        
        # Find spurious tones at fs/M intervals
        if target_bin is None:
            # Auto-detect strongest spur
            spur_indices = []
            for k in range(1, self.M//2):
                idx = int(k * N / self.M)
                if idx < N//2:
                    spur_indices.append(idx)
            
            # Find strongest spur
            spur_powers = [np.abs(spectrum[idx])**2 for idx in spur_indices]
            target_bin = spur_indices[np.argmax(spur_powers)]
        
        print(f"Target spur at {freqs[target_bin]/1e6:.1f} MHz")
        
        # Gradient descent to minimize spur
        skew_estimates = np.zeros(self.M)
        learning_rate = 1e-15
        iterations = 100
        
        for iter in range(iterations):
            # Apply current skew estimates
            corrected_samples = self.apply_skew_correction(samples, skew_estimates)
            
            # Compute spectrum
            spectrum = fft(corrected_samples * signal.windows.hann(N))
            spur_power = np.abs(spectrum[target_bin])**2
            
            # Compute gradient numerically
            gradient = np.zeros(self.M)
            delta = 1e-13  # 0.1 ps perturbation
            
            for ch in range(1, self.M):  # Skip reference channel
                # Perturb skew
                skew_estimates[ch] += delta
                perturbed_samples = self.apply_skew_correction(samples, skew_estimates)
                perturbed_spectrum = fft(perturbed_samples * signal.windows.hann(N))
                perturbed_power = np.abs(perturbed_spectrum[target_bin])**2
                
                # Gradient
                gradient[ch] = (perturbed_power - spur_power) / delta
                
                # Restore
                skew_estimates[ch] -= delta
            
            # Update skews
            skew_estimates -= learning_rate * gradient
            
            if iter % 20 == 0:
                print(f"Iteration {iter}: Spur power = {10*np.log10(spur_power):.1f} dB")
        
        print(f"Final skews - RMS: {np.std(skew_estimates)*1e12:.2f} ps")
        
        self.calibrated_skews = skew_estimates
        return skew_estimates
    
    def blind_adaptive_calibration(self, samples, mu=1e-13, num_iterations=1000):
        """
        Blind adaptive calibration using output spectrum minimization.
        
        Parameters:
        -----------
        samples : array
            Input samples
        mu : float
            Adaptation step size
        num_iterations : int
            Number of adaptation iterations
        
        Returns:
        --------
        skew_estimates : array
            Estimated timing skews
        """
        print("\n=== Blind Adaptive Calibration ===")
        
        N = len(samples)
        skew_estimates = np.zeros(self.M)
        
        # Cost function: spurious power at fs/M harmonics
        def compute_cost(corrected_samples):
            spectrum = np.abs(fft(corrected_samples))
            cost = 0
            
            # Sum spurious power at fs/M intervals
            for k in range(1, self.M//2):
                idx = int(k * N / self.M)
                if idx < N//2:
                    cost += spectrum[idx]**2
                    
            return cost
        
        # Adaptive loop
        cost_history = []
        
        for iter in range(num_iterations):
            # Current cost
            corrected_samples = self.apply_skew_correction(samples, skew_estimates)
            current_cost = compute_cost(corrected_samples)
            cost_history.append(current_cost)
            
            # Stochastic gradient estimation
            gradient = np.zeros(self.M)
            
            for ch in range(1, self.M):  # Skip reference
                # Random perturbation
                delta = np.random.normal(0, 1e-13)
                skew_estimates[ch] += delta
                
                # Evaluate cost change
                perturbed_samples = self.apply_skew_correction(samples, skew_estimates)
                perturbed_cost = compute_cost(perturbed_samples)
                
                # Gradient estimate
                gradient[ch] = (perturbed_cost - current_cost) / delta
                
                # Restore
                skew_estimates[ch] -= delta
            
            # Update with momentum
            momentum = 0.9
            if iter == 0:
                velocity = -mu * gradient
            else:
                velocity = momentum * velocity - mu * gradient
            
            skew_estimates += velocity
            
            if iter % 100 == 0:
                print(f"Iteration {iter}: Cost = {10*np.log10(current_cost):.1f} dB")
        
        print(f"Final skews - RMS: {np.std(skew_estimates)*1e12:.2f} ps")
        
        self.calibrated_skews = skew_estimates
        return skew_estimates, cost_history
    
    def apply_skew_correction(self, samples, skew_corrections):
        """
        Apply timing skew corrections using fractional delay filters.
        
        Parameters:
        -----------
        samples : array
            Input samples
        skew_corrections : array
            Timing corrections for each channel
        
        Returns:
        --------
        corrected_samples : array
            Samples with skew correction applied
        """
        N = len(samples)
        corrected_samples = np.zeros(N)
        
        # Design fractional delay filters for each channel
        for ch in range(self.M):
            if skew_corrections[ch] == 0:
                # No correction needed
                corrected_samples[ch::self.M] = samples[ch::self.M]
                continue
            
            # Fractional delay in samples
            delay_samples = skew_corrections[ch] * self.fs_total
            
            # Extract channel samples
            ch_samples = samples[ch::self.M]
            
            # Apply fractional delay using sinc interpolation
            corrected_ch = self.fractional_delay(ch_samples, delay_samples)
            
            # Insert back
            corrected_samples[ch::self.M] = corrected_ch[:len(ch_samples)]
        
        return corrected_samples
    
    def fractional_delay(self, input_signal, delay_samples, filter_length=64):
        """
        Apply fractional delay using sinc interpolation.
        
        Parameters:
        -----------
        input_signal : array
            Input signal
        delay_samples : float
            Delay in fractional samples
        filter_length : int
            Length of sinc filter
        
        Returns:
        --------
        delayed_signal : array
            Delayed signal
        """
        # Sinc interpolation filter
        n = np.arange(-filter_length//2, filter_length//2)
        h = np.sinc(n - delay_samples)
        h *= signal.windows.hamming(filter_length)
        h /= np.sum(h)
        
        # Apply filter
        delayed_signal = signal.convolve(input_signal, h, mode='same')
        
        return delayed_signal
    
    def measure_performance(self, samples, plot=True):
        """
        Measure ADC performance metrics.
        
        Parameters:
        -----------
        samples : array
            ADC output samples
        plot : bool
            Whether to plot spectrum
        
        Returns:
        --------
        metrics : dict
            Performance metrics (SNDR, SFDR, etc.)
        """
        N = len(samples)
        
        # Apply window
        window = signal.windows.hann(N)
        windowed = samples * window
        
        # Compute spectrum
        spectrum = fft(windowed)
        spectrum_positive = spectrum[:N//2]  # Only positive frequencies
        spectrum_db = 20 * np.log10(np.abs(spectrum_positive) + 1e-15)
        freqs = fftfreq(N, 1/self.fs_total)[:N//2]
        
        # Find signal bin (highest peak)
        signal_bin = np.argmax(np.abs(spectrum_positive))
        signal_power = np.abs(spectrum_positive[signal_bin])**2
        
        # Noise and distortion power
        noise_bins = np.ones(N//2, dtype=bool)
        noise_bins[signal_bin-3:signal_bin+4] = False  # Exclude signal
        noise_power = np.sum(np.abs(spectrum_positive[noise_bins])**2)
        
        # SNDR
        sndr = 10 * np.log10(signal_power / noise_power)
        
        # SFDR - find largest spur
        spur_spectrum = np.abs(spectrum_positive).copy()
        spur_spectrum[signal_bin-3:signal_bin+4] = 0
        max_spur = np.max(spur_spectrum)
        sfdr = 20 * np.log10(np.abs(spectrum_positive[signal_bin]) / max_spur)
        
        # Find timing spurs at fs/M intervals
        timing_spurs = []
        for k in range(1, self.M//2):
            idx = int(k * N / self.M)
            if idx < N//2:
                spur_power = spectrum_db[idx]
                timing_spurs.append((freqs[idx], spur_power))
        
        metrics = {
            'sndr': sndr,
            'sfdr': sfdr,
            'timing_spurs': timing_spurs,
            'signal_freq': freqs[signal_bin],
            'spectrum': spectrum_db,
            'frequencies': freqs
        }
        
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(freqs/1e6, spectrum_db, 'b-', linewidth=0.5)
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Magnitude (dB)')
            plt.title(f'ADC Output Spectrum - SNDR: {sndr:.1f} dB, SFDR: {sfdr:.1f} dB')
            plt.grid(True, alpha=0.3)
            plt.xlim([0, self.fs_total/2e6])
            plt.ylim([-120, 0])
            
            # Mark timing spurs
            for freq, power in timing_spurs:
                plt.plot(freq/1e6, power, 'ro', markersize=6)
            
        return metrics
    
    def compare_calibration_methods(self, signal_type='sine', duration=1e-6):
        """
        Compare different calibration methods.
        
        Parameters:
        -----------
        signal_type : str
            Type of test signal ('sine', 'multitone', 'noise')
        duration : float
            Signal duration in seconds
        
        Returns:
        --------
        results : dict
            Comparison results
        """
        print("\n" + "="*60)
        print("TIMING SKEW CALIBRATION COMPARISON")
        print("="*60)
        
        # Generate test signal
        t = np.arange(0, duration, 1/self.fs_total)
        
        if signal_type == 'sine':
            # Single tone at ~0.45*fs
            f_in = 0.45 * self.fs_total
            signal_func = lambda t: 0.9 * np.sin(2*np.pi*f_in*t)
        elif signal_type == 'multitone':
            # Multiple tones
            f1, f2, f3 = 0.1*self.fs_total, 0.25*self.fs_total, 0.4*self.fs_total
            signal_func = lambda t: 0.3*(np.sin(2*np.pi*f1*t) + 
                                       np.sin(2*np.pi*f2*t) + 
                                       np.sin(2*np.pi*f3*t))
        
        # Sample with skews
        samples_skewed, _ = self.sample_signal(t, signal_func, include_skews=True)
        
        # Measure initial performance
        print("\n### Initial Performance (with skews) ###")
        metrics_initial = self.measure_performance(samples_skewed, plot=True)
        plt.title('Initial Spectrum (with timing skews)')
        
        results = {
            'initial': metrics_initial,
            'methods': {}
        }
        
        # Method 1: Correlation-based
        skews_corr = self.correlation_based_calibration(samples_skewed)
        samples_corr = self.apply_skew_correction(samples_skewed, -skews_corr)
        metrics_corr = self.measure_performance(samples_corr, plot=True)
        plt.title('After Correlation-based Calibration')
        
        results['methods']['correlation'] = {
            'skew_estimates': skews_corr,
            'metrics': metrics_corr,
            'error_rms': np.std(skews_corr - self.timing_skews)*1e12
        }
        
        # Method 2: Sine-fit (if using sine input)
        if signal_type == 'sine':
            skews_sine = self.sine_fit_calibration(samples_skewed, f_in)
            samples_sine = self.apply_skew_correction(samples_skewed, -skews_sine)
            metrics_sine = self.measure_performance(samples_sine, plot=True)
            plt.title('After Sine-fit Calibration')
            
            results['methods']['sine_fit'] = {
                'skew_estimates': skews_sine,
                'metrics': metrics_sine,
                'error_rms': np.std(skews_sine - self.timing_skews)*1e12
            }
        
        # Method 3: Spectral
        skews_spec = self.spectral_calibration(samples_skewed)
        samples_spec = self.apply_skew_correction(samples_skewed, -skews_spec)
        metrics_spec = self.measure_performance(samples_spec, plot=True)
        plt.title('After Spectral Calibration')
        
        results['methods']['spectral'] = {
            'skew_estimates': skews_spec,
            'metrics': metrics_spec,
            'error_rms': np.std(skews_spec - self.timing_skews)*1e12
        }
        
        # Method 4: Blind adaptive
        skews_blind, cost_history = self.blind_adaptive_calibration(
            samples_skewed, num_iterations=500)
        samples_blind = self.apply_skew_correction(samples_skewed, -skews_blind)
        metrics_blind = self.measure_performance(samples_blind, plot=True)
        plt.title('After Blind Adaptive Calibration')
        
        results['methods']['blind_adaptive'] = {
            'skew_estimates': skews_blind,
            'metrics': metrics_blind,
            'error_rms': np.std(skews_blind - self.timing_skews)*1e12,
            'cost_history': cost_history
        }
        
        # Plot calibration results summary
        self.plot_calibration_summary(results)
        
        return results
    
    def plot_calibration_summary(self, results):
        """Plot comprehensive calibration results."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Skew estimation accuracy
        ax = axes[0, 0]
        methods = list(results['methods'].keys())
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, method in enumerate(methods):
            estimates = results['methods'][method]['skew_estimates']
            error = (estimates - self.timing_skews) * 1e12  # Convert to ps
            ax.plot(error, 'o-', color=colors[i], label=method, alpha=0.7)
        
        ax.set_xlabel('Channel Index')
        ax.set_ylabel('Estimation Error (ps)')
        ax.set_title('Skew Estimation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. SNDR improvement
        ax = axes[0, 1]
        sndr_values = [results['initial']['sndr']]
        labels = ['Initial']
        
        for method in methods:
            sndr_values.append(results['methods'][method]['metrics']['sndr'])
            labels.append(method)
        
        bars = ax.bar(labels, sndr_values, alpha=0.7)
        bars[0].set_color('gray')
        
        ax.set_ylabel('SNDR (dB)')
        ax.set_title('SNDR Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. SFDR improvement
        ax = axes[0, 2]
        sfdr_values = [results['initial']['sfdr']]
        
        for method in methods:
            sfdr_values.append(results['methods'][method]['metrics']['sfdr'])
        
        bars = ax.bar(labels, sfdr_values, alpha=0.7)
        bars[0].set_color('gray')
        
        ax.set_ylabel('SFDR (dB)')
        ax.set_title('SFDR Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Estimation error RMS
        ax = axes[1, 0]
        error_rms = []
        method_names = []
        
        for method in methods:
            if 'error_rms' in results['methods'][method]:
                error_rms.append(results['methods'][method]['error_rms'])
                method_names.append(method)
        
        ax.bar(method_names, error_rms, alpha=0.7)
        ax.set_ylabel('RMS Error (ps)')
        ax.set_title('Skew Estimation RMS Error')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 5. Convergence plot (for adaptive method)
        ax = axes[1, 1]
        if 'blind_adaptive' in results['methods']:
            cost_history = results['methods']['blind_adaptive']['cost_history']
            ax.semilogy(cost_history, 'b-', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Cost Function')
            ax.set_title('Blind Adaptive Convergence')
            ax.grid(True, alpha=0.3)
        
        # 6. Summary table
        ax = axes[1, 2]
        ax.axis('tight')
        ax.axis('off')
        
        # Create summary data
        summary_data = []
        summary_data.append(['Method', 'SNDR (dB)', 'SFDR (dB)', 'RMS Error (ps)'])
        summary_data.append(['Initial', f"{results['initial']['sndr']:.1f}", 
                           f"{results['initial']['sfdr']:.1f}", '-'])
        
        for method in methods:
            m_data = results['methods'][method]
            error_rms = m_data.get('error_rms', '-')
            if isinstance(error_rms, float):
                error_rms = f"{error_rms:.2f}"
            
            summary_data.append([
                method,
                f"{m_data['metrics']['sndr']:.1f}",
                f"{m_data['metrics']['sfdr']:.1f}",
                error_rms
            ])
        
        table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        ax.set_title('Performance Summary', pad=20)
        
        plt.tight_layout()
        plt.show()
    
    def monte_carlo_analysis(self, num_trials=100, skew_range=(1e-12, 20e-12)):
        """
        Monte Carlo analysis of calibration methods.
        
        Parameters:
        -----------
        num_trials : int
            Number of Monte Carlo trials
        skew_range : tuple
            Range of timing skew standard deviations to test
        
        Returns:
        --------
        mc_results : dict
            Monte Carlo analysis results
        """
        print("\n=== Monte Carlo Analysis ===")
        print(f"Running {num_trials} trials...")
        
        skew_stds = np.linspace(skew_range[0], skew_range[1], 5)
        
        mc_results = {
            'skew_stds': skew_stds,
            'methods': {
                'correlation': {'sndr': [], 'sfdr': [], 'error_rms': []},
                'spectral': {'sndr': [], 'sfdr': [], 'error_rms': []},
                'blind_adaptive': {'sndr': [], 'sfdr': [], 'error_rms': []}
            }
        }
        
        # Test signal parameters
        duration = 0.5e-6  # 0.5 µs
        t = np.arange(0, duration, 1/self.fs_total)
        f_in = 0.45 * self.fs_total
        signal_func = lambda t: 0.9 * np.sin(2*np.pi*f_in*t)
        
        for skew_std in skew_stds:
            print(f"\nTesting skew std = {skew_std*1e12:.1f} ps")
            
            for method in mc_results['methods']:
                sndr_trials = []
                sfdr_trials = []
                error_trials = []
                
                for trial in range(num_trials):
                    # Add random skews
                    self.add_timing_skews(skew_std=skew_std)
                    
                    # Sample signal
                    samples, _ = self.sample_signal(t, signal_func, include_skews=True)
                    
                    # Apply calibration
                    if method == 'correlation':
                        skew_est = self.correlation_based_calibration(samples)
                    elif method == 'spectral':
                        skew_est = self.spectral_calibration(samples)
                    elif method == 'blind_adaptive':
                        skew_est, _ = self.blind_adaptive_calibration(samples, 
                                                                     num_iterations=200)
                    
                    # Apply correction
                    corrected = self.apply_skew_correction(samples, -skew_est)
                    
                    # Measure performance
                    metrics = self.measure_performance(corrected, plot=False)
                    
                    sndr_trials.append(metrics['sndr'])
                    sfdr_trials.append(metrics['sfdr'])
                    error_trials.append(np.std(skew_est - self.timing_skews)*1e12)
                
                # Store statistics
                mc_results['methods'][method]['sndr'].append({
                    'mean': np.mean(sndr_trials),
                    'std': np.std(sndr_trials)
                })
                mc_results['methods'][method]['sfdr'].append({
                    'mean': np.mean(sfdr_trials),
                    'std': np.std(sfdr_trials)
                })
                mc_results['methods'][method]['error_rms'].append({
                    'mean': np.mean(error_trials),
                    'std': np.std(error_trials)
                })
        
        # Plot Monte Carlo results
        self.plot_monte_carlo_results(mc_results)
        
        return mc_results
    
    def plot_monte_carlo_results(self, mc_results):
        """Plot Monte Carlo analysis results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        skew_stds_ps = mc_results['skew_stds'] * 1e12
        colors = {'correlation': 'blue', 'spectral': 'red', 'blind_adaptive': 'green'}
        
        # 1. SNDR vs skew
        ax = axes[0]
        for method, color in colors.items():
            means = [d['mean'] for d in mc_results['methods'][method]['sndr']]
            stds = [d['std'] for d in mc_results['methods'][method]['sndr']]
            
            ax.errorbar(skew_stds_ps, means, yerr=stds, 
                       color=color, marker='o', label=method, capsize=5)
        
        ax.set_xlabel('Timing Skew RMS (ps)')
        ax.set_ylabel('SNDR (dB)')
        ax.set_title('SNDR vs Timing Skew')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. SFDR vs skew
        ax = axes[1]
        for method, color in colors.items():
            means = [d['mean'] for d in mc_results['methods'][method]['sfdr']]
            stds = [d['std'] for d in mc_results['methods'][method]['sfdr']]
            
            ax.errorbar(skew_stds_ps, means, yerr=stds, 
                       color=color, marker='o', label=method, capsize=5)
        
        ax.set_xlabel('Timing Skew RMS (ps)')
        ax.set_ylabel('SFDR (dB)')
        ax.set_title('SFDR vs Timing Skew')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Estimation error vs skew
        ax = axes[2]
        for method, color in colors.items():
            means = [d['mean'] for d in mc_results['methods'][method]['error_rms']]
            stds = [d['std'] for d in mc_results['methods'][method]['error_rms']]
            
            ax.errorbar(skew_stds_ps, means, yerr=stds, 
                       color=color, marker='o', label=method, capsize=5)
        
        ax.set_xlabel('Timing Skew RMS (ps)')
        ax.set_ylabel('Estimation Error RMS (ps)')
        ax.set_title('Calibration Accuracy vs Timing Skew')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def hardware_implementation_analysis(self):
        """
        Analyze hardware implementation aspects of calibration methods.
        """
        print("\n" + "="*60)
        print("HARDWARE IMPLEMENTATION ANALYSIS")
        print("="*60)
        
        # Implementation metrics
        implementations = {
            'Correlation-based': {
                'complexity': 'O(N log N)',
                'memory': f'{self.M} × buffer_size',
                'latency': 'High (FFT required)',
                'precision': '16-24 bits',
                'hardware': 'DSP blocks + RAM',
                'power': 'Medium',
                'area': 'Medium'
            },
            'Sine-fit': {
                'complexity': 'O(N)',
                'memory': f'{self.M} × fit_samples',
                'latency': 'Medium',
                'precision': '20-24 bits',
                'hardware': 'Multipliers + Accumulators',
                'power': 'Low-Medium',
                'area': 'Small-Medium'
            },
            'Spectral': {
                'complexity': 'O(N log N) × iterations',
                'memory': 'FFT buffer',
                'latency': 'Very High',
                'precision': '16-20 bits',
                'hardware': 'FFT engine + Control',
                'power': 'High',
                'area': 'Large'
            },
            'Blind Adaptive': {
                'complexity': 'O(N) per iteration',
                'memory': 'Minimal',
                'latency': 'Continuous',
                'precision': '12-16 bits',
                'hardware': 'Simple MAC units',
                'power': 'Low',
                'area': 'Small'
            }
        }
        
        # Print comparison table
        print("\n┌─────────────────┬────────────────┬─────────────┬──────────┬────────────┬───────┬──────┐")
        print("│ Method          │ Complexity     │ Memory      │ Latency  │ Hardware   │ Power │ Area │")
        print("├─────────────────┼────────────────┼─────────────┼──────────┼────────────┼───────┼──────┤")
        
        for method, specs in implementations.items():
            print(f"│ {method:<15} │ {specs['complexity']:<14} │ "
                  f"{specs['memory']:<11} │ {specs['latency']:<8} │ "
                  f"{specs['hardware']:<10} │ {specs['power']:<5} │ {specs['area']:<4} │")
        
        print("└─────────────────┴────────────────┴─────────────┴──────────┴────────────┴───────┴──────┘")
        
        # Resource estimation for 64-channel ADC at 10 GSps
        print("\n### Resource Estimation (64-ch @ 10 GSps) ###")
        
        # Correlation method
        fft_size = 1024
        fft_ops = fft_size * np.log2(fft_size) * 5  # Complex multiplies
        correlation_rate = self.fs_total / fft_size
        correlation_gops = fft_ops * correlation_rate / 1e9
        
        print(f"\nCorrelation-based:")
        print(f"  - FFT operations: {correlation_gops:.1f} GOPS")
        print(f"  - Memory: {self.M * fft_size * 2 * 2 / 1024:.1f} KB (16-bit)")
        print(f"  - Update rate: {correlation_rate/1e3:.1f} kHz")
        
        # Adaptive method
        adaptive_ops = self.M * 4  # Per sample: mult, add, compare, update
        adaptive_gops = adaptive_ops * self.fs_total / 1e9
        
        print(f"\nBlind Adaptive:")
        print(f"  - Operations: {adaptive_gops:.1f} GOPS")
        print(f"  - Memory: {self.M * 4 * 2 / 1024:.1f} KB (coefficients + buffers)")
        print(f"  - Update rate: {self.fs_total/1e6:.0f} MHz (continuous)")
        
        return implementations


# Example usage and demonstration
if __name__ == "__main__":
    # Create 64x TI-SAR ADC
    adc = TimeInterleavedSARADC(
        num_channels=64,
        fs_total=10e9,  # 10 GSps total
        resolution=12
    )
    
    # Add realistic timing skews (5ps RMS)
    adc.add_timing_skews(skew_std=5e-12)
    
    # Add small gain/offset mismatches
    adc.add_gain_offset_mismatch(gain_std=0.005, offset_std=0.0005)
    
    # Compare calibration methods
    results = adc.compare_calibration_methods(signal_type='sine', duration=1e-6)
    
    # Run Monte Carlo analysis
    mc_results = adc.monte_carlo_analysis(num_trials=50, skew_range=(1e-12, 15e-12))
    
    # Hardware implementation analysis
    hw_analysis = adc.hardware_implementation_analysis()
    
    # Print final recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR 64x TI-SAR ADC")
    print("="*60)
    
    print("\n1. For Static/Lab Calibration:")
    print("   - Use Sine-fit or Correlation-based methods")
    print("   - High accuracy, one-time calibration")
    print("   - Can achieve <1ps RMS error")
    
    print("\n2. For Production/Field Use:")
    print("   - Implement Blind Adaptive calibration")
    print("   - Continuous tracking of drift")
    print("   - Low power and area overhead")
    
    print("\n3. For High-Performance Systems:")
    print("   - Hybrid approach: Sine-fit initialization + Adaptive tracking")
    print("   - Spectral monitoring for diagnostics")
    print("   - Digital correction with fractional delay filters")
    
    print("\n4. Key Design Considerations:")
    print("   - Fractional delay filter design critical for correction")
    print("   - Consider bandwidth vs skew correction trade-off")
    print("   - Monitor temperature for drift compensation")
    print("   - Implement background calibration for mission-critical apps")