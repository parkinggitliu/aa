import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from collections import deque
import math

class AllDigitalPLL:
    def __init__(self, 
                 ref_freq=100e6,        # Reference frequency (Hz)
                 target_freq=1e9,       # Target output frequency (Hz)
                 kp=0.1,                # Proportional gain
                 ki=0.001,              # Integral gain
                 kd=0.01,               # Derivative gain
                 dco_resolution=32,     # DCO frequency resolution bits
                 tdc_resolution=10,     # TDC time resolution bits
                 duty_cycle_target=0.5, # Target duty cycle
                 duty_correction_gain=0.01):
        
        # PLL parameters
        self.ref_freq = ref_freq
        self.target_freq = target_freq
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # DCO parameters
        self.dco_resolution = dco_resolution
        self.dco_max_code = 2**dco_resolution - 1
        self.dco_freq_range = target_freq * 0.2  # ±10% tuning range
        self.dco_freq_step = self.dco_freq_range / self.dco_max_code
        
        # TDC parameters  
        self.tdc_resolution = tdc_resolution
        self.tdc_max_code = 2**tdc_resolution - 1
        self.tdc_time_step = 1.0 / (target_freq * self.tdc_max_code)
        
        # Duty cycle correction parameters
        self.duty_cycle_target = duty_cycle_target
        self.duty_correction_gain = duty_correction_gain
        
        # State variables
        self.phase_error_integral = 0.0
        self.prev_phase_error = 0.0
        self.dco_code = self.dco_max_code // 2  # Start at center frequency
        self.duty_correction_code = 0
        
        # Frequency divider ratio
        self.divider_ratio = int(target_freq / ref_freq)
        
        # History for analysis
        self.phase_error_history = deque(maxlen=10000)
        self.frequency_error_history = deque(maxlen=10000)
        self.duty_cycle_history = deque(maxlen=10000)
        self.dco_code_history = deque(maxlen=10000)
        self.duty_correction_history = deque(maxlen=10000)
        
        # Duty cycle measurement variables
        self.duty_cycle_buffer = deque(maxlen=100)
        self.rising_edge_times = deque(maxlen=50)
        self.falling_edge_times = deque(maxlen=50)
        
    def time_to_digital_converter(self, time_difference):
        """Convert time difference to digital code (TDC)"""
        # Quantize time difference to TDC resolution
        normalized_time = time_difference / self.tdc_time_step
        tdc_code = np.clip(int(normalized_time), 0, self.tdc_max_code)
        return tdc_code
    
    def digital_controlled_oscillator(self, control_code, duty_correction=0):
        """Generate DCO output with duty cycle control"""
        # Convert control code to frequency
        freq_offset = (control_code - self.dco_max_code // 2) * self.dco_freq_step
        actual_freq = self.target_freq + freq_offset
        
        # Generate clock signal with duty cycle correction
        period = 1.0 / actual_freq
        duty_cycle = self.duty_cycle_target + duty_correction * 0.01  # 1% per correction code
        duty_cycle = np.clip(duty_cycle, 0.1, 0.9)  # Limit duty cycle range
        
        return actual_freq, period, duty_cycle
    
    def phase_frequency_detector(self, ref_edge, fb_edge, ref_time, fb_time):
        """Digital phase-frequency detector"""
        # Calculate phase difference
        time_diff = ref_time - fb_time
        
        # Convert to phase (normalized to 2π)
        phase_diff = 2 * np.pi * time_diff * self.ref_freq
        
        # Wrap phase to [-π, π]
        while phase_diff > np.pi:
            phase_diff -= 2 * np.pi
        while phase_diff < -np.pi:
            phase_diff += 2 * np.pi
            
        # Generate UP/DOWN signals for charge pump equivalent
        if phase_diff > 0:
            up_signal = 1
            down_signal = 0
        else:
            up_signal = 0
            down_signal = 1
            
        return phase_diff, up_signal, down_signal
    
    def digital_loop_filter(self, phase_error):
        """Digital PID loop filter"""
        # Integral term
        self.phase_error_integral += phase_error
        
        # Derivative term
        phase_error_derivative = phase_error - self.prev_phase_error
        
        # PID control output
        control_signal = (self.kp * phase_error + 
                         self.ki * self.phase_error_integral +
                         self.kd * phase_error_derivative)
        
        self.prev_phase_error = phase_error
        
        return control_signal
    
    def measure_duty_cycle(self, clock_signal, sample_times):
        """Measure duty cycle using digital correlation"""
        if len(clock_signal) < 10:
            return self.duty_cycle_target
        
        # Find rising and falling edges
        rising_edges = []
        falling_edges = []
        
        for i in range(1, len(clock_signal)):
            if clock_signal[i-1] == 0 and clock_signal[i] == 1:
                rising_edges.append(sample_times[i])
            elif clock_signal[i-1] == 1 and clock_signal[i] == 0:
                falling_edges.append(sample_times[i])
        
        # Calculate duty cycle from edge times
        if len(rising_edges) > 0 and len(falling_edges) > 0:
            # Align edges for measurement
            min_pairs = min(len(rising_edges), len(falling_edges))
            
            duty_cycles = []
            for i in range(min_pairs):
                if i < len(falling_edges) and rising_edges[i] < falling_edges[i]:
                    high_time = falling_edges[i] - rising_edges[i]
                    
                    # Find next rising edge for period calculation
                    if i + 1 < len(rising_edges):
                        period = rising_edges[i+1] - rising_edges[i]
                        if period > 0:
                            duty_cycle = high_time / period
                            duty_cycles.append(duty_cycle)
            
            if duty_cycles:
                measured_duty = np.mean(duty_cycles)
                self.duty_cycle_buffer.append(measured_duty)
                return measured_duty
        
        return self.duty_cycle_target
    
    def duty_cycle_correlator(self, measured_duty):
        """Correlate duty cycle error and generate correction"""
        duty_error = measured_duty - self.duty_cycle_target
        
        # Simple integrating correlator
        correction = -self.duty_correction_gain * duty_error
        
        # Quantize correction to digital codes
        correction_code = int(correction * 100)  # Scale for digital implementation
        correction_code = np.clip(correction_code, -50, 50)  # Limit correction range
        
        return duty_error, correction_code
    
    def frequency_divider(self, dco_freq, sample_time):
        """Frequency divider for feedback"""
        # Simple divider implementation
        divided_freq = dco_freq / self.divider_ratio
        divided_period = 1.0 / divided_freq
        
        # Generate divided clock edge timing
        edge_time = sample_time % divided_period
        fb_edge = 1 if edge_time < divided_period * 0.5 else 0
        
        return fb_edge, edge_time
    
    def process_sample(self, ref_edge, sample_time):
        """Process one sample through the ADPLL"""
        # Generate DCO output
        dco_freq, dco_period, current_duty = self.digital_controlled_oscillator(
            self.dco_code, self.duty_correction_code)
        
        # Generate feedback signal through divider
        fb_edge, fb_time = self.frequency_divider(dco_freq, sample_time)
        
        # Phase-frequency detection
        phase_error, up_sig, down_sig = self.phase_frequency_detector(
            ref_edge, fb_edge, sample_time, fb_time)
        
        # Digital loop filter (PID control)
        control_signal = self.digital_loop_filter(phase_error)
        
        # Update DCO control code
        dco_code_change = int(control_signal * 1000)  # Scale for digital implementation
        self.dco_code += dco_code_change
        self.dco_code = np.clip(self.dco_code, 0, self.dco_max_code)
        
        # Generate clock signal for duty cycle measurement
        clock_phase = (sample_time * dco_freq) % 1.0
        clock_signal = 1 if clock_phase < current_duty else 0
        
        # Measure duty cycle (every few samples for efficiency)
        if len(self.duty_cycle_history) % 10 == 0:
            # Create a short clock history for measurement
            clock_history = [clock_signal] * 20  # Simplified for this sample
            time_history = [sample_time + i * 1e-12 for i in range(20)]
            measured_duty = self.measure_duty_cycle(clock_history, time_history)
            
            # Duty cycle correction
            duty_error, correction_code = self.duty_cycle_correlator(measured_duty)
            self.duty_correction_code += correction_code
            self.duty_correction_code = np.clip(self.duty_correction_code, -100, 100)
        else:
            measured_duty = self.duty_cycle_target
            duty_error = 0
        
        # Calculate frequency error
        freq_error = (dco_freq - self.target_freq) / self.target_freq
        
        # Store history
        self.phase_error_history.append(phase_error)
        self.frequency_error_history.append(freq_error)
        self.duty_cycle_history.append(measured_duty)
        self.dco_code_history.append(self.dco_code)
        self.duty_correction_history.append(self.duty_correction_code)
        
        return {
            'dco_freq': dco_freq,
            'phase_error': phase_error,
            'freq_error': freq_error,
            'dco_code': self.dco_code,
            'duty_cycle': measured_duty,
            'duty_error': duty_error,
            'duty_correction': self.duty_correction_code,
            'clock_output': clock_signal,
            'up_signal': up_sig,
            'down_signal': down_sig
        }

class DutyCycleCorrector:
    """Dedicated duty cycle correction module"""
    
    def __init__(self, target_duty=0.5, correction_bw=1000, sample_rate=1e9):
        self.target_duty = target_duty
        self.correction_bw = correction_bw
        self.sample_rate = sample_rate
        
        # Correlation filters for duty cycle detection
        self.integrator_state = 0.0
        self.alpha = 2 * np.pi * correction_bw / sample_rate
        
        # Duty cycle measurement using correlation
        self.correlation_buffer = deque(maxlen=1000)
        self.clock_buffer = deque(maxlen=1000)
        
    def correlate_duty_cycle(self, clock_signal):
        """Correlate clock signal to extract duty cycle information"""
        self.clock_buffer.append(clock_signal)
        
        if len(self.clock_buffer) < 100:
            return self.target_duty, 0
        
        # Convert to array for processing
        clock_array = np.array(list(self.clock_buffer))
        
        # Calculate duty cycle using temporal correlation
        # Method 1: Average of high states
        duty_cycle_avg = np.mean(clock_array)
        
        # Method 2: Autocorrelation-based measurement
        if len(clock_array) >= 200:
            # Find fundamental period through autocorrelation
            autocorr = np.correlate(clock_array, clock_array, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find period (skip DC component)
            peaks, _ = signal.find_peaks(autocorr[10:], height=np.max(autocorr) * 0.5)
            if len(peaks) > 0:
                period = peaks[0] + 10
                
                # Measure duty cycle over one period
                if period < len(clock_array):
                    period_signal = clock_array[-period:]
                    duty_cycle_corr = np.mean(period_signal)
                else:
                    duty_cycle_corr = duty_cycle_avg
            else:
                duty_cycle_corr = duty_cycle_avg
        else:
            duty_cycle_corr = duty_cycle_avg
        
        # Weighted combination of measurements
        measured_duty = 0.7 * duty_cycle_avg + 0.3 * duty_cycle_corr
        
        # Calculate error
        duty_error = measured_duty - self.target_duty
        
        # Integrate error for correction signal
        self.integrator_state += self.alpha * duty_error
        correction_signal = self.integrator_state
        
        return measured_duty, correction_signal

def simulate_adpll_with_duty_correction():
    """Simulate ADPLL with duty cycle correction"""
    
    # Simulation parameters
    sim_time = 1e-3  # 1 ms simulation
    sample_rate = 10e9  # 10 GHz sampling rate
    samples = int(sim_time * sample_rate)
    time_vector = np.linspace(0, sim_time, samples)
    
    # Create ADPLL
    adpll = AllDigitalPLL(
        ref_freq=100e6,
        target_freq=1e9,
        kp=0.05,
        ki=0.001,
        kd=0.01,
        duty_cycle_target=0.5
    )
    
    # Create duty cycle corrector
    duty_corrector = DutyCycleCorrector(target_duty=0.5)
    
    # Reference clock generation
    ref_period = 1.0 / adpll.ref_freq
    ref_edges = []
    
    # Simulation results storage
    results = []
    
    # Add some disturbances
    duty_disturbance = 0.1 * np.sin(2 * np.pi * 1000 * time_vector)  # 1 kHz duty cycle variation
    freq_disturbance = 0.001 * np.sin(2 * np.pi * 100 * time_vector)  # 100 Hz frequency variation
    
    print("Running ADPLL simulation with duty cycle correction...")
    
    for i, t in enumerate(time_vector):
        # Generate reference edges
        ref_edge = 1 if (t % ref_period) < (ref_period * 0.5) else 0
        
        # Add disturbances
        adpll.duty_cycle_target = 0.5 + duty_disturbance[i] * 0.1
        adpll.target_freq = 1e9 + freq_disturbance[i] * 1e6
        
        # Process sample through ADPLL
        result = adpll.process_sample(ref_edge, t)
        
        # Additional duty cycle correction
        measured_duty, correction = duty_corrector.correlate_duty_cycle(result['clock_output'])
        result['enhanced_duty_correction'] = correction
        result['enhanced_duty_measurement'] = measured_duty
        
        results.append(result)
        
        # Progress indicator
        if i % (samples // 10) == 0:
            print(f"Progress: {100 * i / samples:.0f}%")
    
    return results, time_vector, adpll

def plot_adpll_results(results, time_vector, adpll):
    """Plot comprehensive ADPLL results"""
    
    # Extract data for plotting
    phase_errors = [r['phase_error'] for r in results]
    freq_errors = [r['freq_error'] for r in results]
    dco_codes = [r['dco_code'] for r in results]
    duty_cycles = [r['duty_cycle'] for r in results]
    duty_corrections = [r['duty_correction'] for r in results]
    clock_outputs = [r['clock_output'] for r in results]
    
    # Create comprehensive plot
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    # Plot 1: Phase error vs time
    axes[0,0].plot(time_vector * 1e6, np.array(phase_errors) * 180/np.pi)
    axes[0,0].set_title('Phase Error')
    axes[0,0].set_xlabel('Time (μs)')
    axes[0,0].set_ylabel('Phase Error (degrees)')
    axes[0,0].grid(True)
    
    # Plot 2: Frequency error vs time
    axes[0,1].plot(time_vector * 1e6, np.array(freq_errors) * 1e6)
    axes[0,1].set_title('Frequency Error')
    axes[0,1].set_xlabel('Time (μs)')
    axes[0,1].set_ylabel('Frequency Error (ppm)')
    axes[0,1].grid(True)
    
    # Plot 3: DCO control code
    axes[0,2].plot(time_vector * 1e6, dco_codes)
    axes[0,2].set_title('DCO Control Code')
    axes[0,2].set_xlabel('Time (μs)')
    axes[0,2].set_ylabel('DCO Code')
    axes[0,2].grid(True)
    
    # Plot 4: Duty cycle measurement
    axes[1,0].plot(time_vector * 1e6, np.array(duty_cycles) * 100)
    axes[1,0].axhline(y=50, color='r', linestyle='--', label='Target')
    axes[1,0].set_title('Measured Duty Cycle')
    axes[1,0].set_xlabel('Time (μs)')
    axes[1,0].set_ylabel('Duty Cycle (%)')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Plot 5: Duty cycle correction
    axes[1,1].plot(time_vector * 1e6, duty_corrections)
    axes[1,1].set_title('Duty Cycle Correction Code')
    axes[1,1].set_xlabel('Time (μs)')
    axes[1,1].set_ylabel('Correction Code')
    axes[1,1].grid(True)
    
    # Plot 6: Clock output (zoomed)
    zoom_samples = 1000
    axes[1,2].plot(time_vector[:zoom_samples] * 1e9, clock_outputs[:zoom_samples])
    axes[1,2].set_title('Clock Output (Zoomed)')
    axes[1,2].set_xlabel('Time (ns)')
    axes[1,2].set_ylabel('Clock')
    axes[1,2].grid(True)
    
    # Plot 7: Phase error histogram
    axes[2,0].hist(np.array(phase_errors) * 180/np.pi, bins=50, alpha=0.7)
    axes[2,0].set_title('Phase Error Distribution')
    axes[2,0].set_xlabel('Phase Error (degrees)')
    axes[2,0].set_ylabel('Count')
    axes[2,0].grid(True)
    
    # Plot 8: Duty cycle error histogram
    duty_errors = np.array(duty_cycles) - 0.5
    axes[2,1].hist(duty_errors * 100, bins=50, alpha=0.7)
    axes[2,1].set_title('Duty Cycle Error Distribution')
    axes[2,1].set_xlabel('Duty Cycle Error (%)')
    axes[2,1].set_ylabel('Count')
    axes[2,1].grid(True)
    
    # Plot 9: PLL settling behavior
    axes[2,2].semilogy(time_vector * 1e6, np.abs(phase_errors))
    axes[2,2].set_title('Phase Error Convergence')
    axes[2,2].set_xlabel('Time (μs)')
    axes[2,2].set_ylabel('|Phase Error| (rad)')
    axes[2,2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n=== ADPLL Performance Statistics ===")
    print(f"RMS Phase Error: {np.sqrt(np.mean(np.array(phase_errors)**2)) * 180/np.pi:.3f} degrees")
    print(f"RMS Frequency Error: {np.sqrt(np.mean(np.array(freq_errors)**2)) * 1e6:.3f} ppm")
    print(f"RMS Duty Cycle Error: {np.sqrt(np.mean(duty_errors**2)) * 100:.3f} %")
    print(f"Final DCO Code: {dco_codes[-1]}")
    print(f"Final Duty Correction: {duty_corrections[-1]}")
    print(f"Settling Time (90% of final): {estimate_settling_time(phase_errors, time_vector):.1f} μs")

def estimate_settling_time(phase_errors, time_vector):
    """Estimate PLL settling time"""
    final_error = np.mean(phase_errors[-100:])  # Average of last 100 samples
    threshold = 0.1 * abs(final_error) + abs(final_error)  # 90% settling criterion
    
    for i in range(len(phase_errors)//2, len(phase_errors)):
        if abs(phase_errors[i] - final_error) < threshold:
            return time_vector[i] * 1e6  # Convert to microseconds
    
    return time_vector[-1] * 1e6

# Main execution
if __name__ == "__main__":
    # Run simulation
    results, time_vector, adpll = simulate_adpll_with_duty_correction()
    
    # Plot results
    plot_adpll_results(results, time_vector, adpll)
    
    # Additional analysis
    print("\n=== ADPLL Configuration ===")
    print(f"Reference Frequency: {adpll.ref_freq/1e6:.1f} MHz")
    print(f"Target Frequency: {adpll.target_freq/1e9:.1f} GHz")
    print(f"Division Ratio: {adpll.divider_ratio}")
    print(f"DCO Resolution: {adpll.dco_resolution} bits")
    print(f"TDC Resolution: {adpll.tdc_resolution} bits")
    print(f"Loop Gains - Kp: {adpll.kp}, Ki: {adpll.ki}, Kd: {adpll.kd}")