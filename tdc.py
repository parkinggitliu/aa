import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from collections import deque
import random

class TDCCore:
    """Base Time-to-Digital Converter implementation"""
    
    def __init__(self, resolution_bits=10, max_time_range=10e-9, architecture='flash'):
        self.resolution_bits = resolution_bits
        self.max_time_range = max_time_range
        self.max_code = 2**resolution_bits - 1
        self.nominal_lsb = max_time_range / (2**resolution_bits)
        self.architecture = architecture
        
        # Non-linearity parameters
        self.dnl = np.zeros(2**resolution_bits)  # Differential Non-Linearity
        self.inl = np.zeros(2**resolution_bits)  # Integral Non-Linearity
        
        # Calibration data
        self.calibration_table = None
        self.is_calibrated = False
        
        # Statistics
        self.conversion_count = 0
        self.measurement_history = deque(maxlen=10000)

class FlashTDC(TDCCore):
    """Flash/Parallel TDC Implementation"""
    
    def __init__(self, resolution_bits=10, max_time_range=10e-9):
        super().__init__(resolution_bits, max_time_range, 'flash')
        
        # Delay line parameters for flash TDC
        self.num_stages = 2**resolution_bits
        self.nominal_delay_per_stage = max_time_range / self.num_stages
        
        # Generate realistic delay variations (process variations)
        self.generate_delay_variations()
        
    def generate_delay_variations(self):
        """Generate realistic delay variations for TDC stages"""
        # Random process variations (Gaussian distribution)
        process_sigma = 0.05  # 5% sigma variation
        process_variations = np.random.normal(1.0, process_sigma, self.num_stages)
        
        # Systematic variations (gradient across chip)
        gradient = np.linspace(0.95, 1.05, self.num_stages)
        
        # Temperature variations (sinusoidal across chip)
        temp_variation = 0.02 * np.sin(2 * np.pi * np.arange(self.num_stages) / self.num_stages)
        
        # Combined delay variations
        self.actual_delays = self.nominal_delay_per_stage * process_variations * gradient * (1 + temp_variation)
        
        # Calculate cumulative delays
        self.cumulative_delays = np.cumsum(self.actual_delays)
        
        # Calculate DNL and INL
        self.calculate_nonlinearity()
    
    def calculate_nonlinearity(self):
        """Calculate DNL and INL from delay variations"""
        # DNL: deviation of actual step from ideal step
        ideal_step = self.nominal_delay_per_stage
        actual_steps = self.actual_delays
        self.dnl = (actual_steps - ideal_step) / ideal_step
        
        # INL: cumulative effect of DNL
        ideal_cumulative = np.arange(len(self.cumulative_delays)) * ideal_step
        self.inl = (self.cumulative_delays - ideal_cumulative) / ideal_step
    
    def convert_time_to_code(self, time_interval):
        """Convert time interval to digital code"""
        if time_interval < 0:
            return 0
        if time_interval > self.max_time_range:
            return self.max_code
        
        # Find which delay stage the time falls into
        code = np.searchsorted(self.cumulative_delays, time_interval)
        code = min(code, self.max_code)
        
        self.conversion_count += 1
        self.measurement_history.append((time_interval, code))
        
        return code
    
    def convert_code_to_time(self, code):
        """Convert digital code back to time (for calibration)"""
        if code >= len(self.cumulative_delays):
            return self.max_time_range
        return self.cumulative_delays[code]

class VernerTDC(TDCCore):
    """Vernier TDC Implementation"""
    
    def __init__(self, resolution_bits=10, max_time_range=10e-9):
        super().__init__(resolution_bits, max_time_range, 'vernier')
        
        # Vernier parameters
        self.fast_delay = 100e-12  # 100 ps
        self.slow_delay = 101e-12  # 101 ps (1 ps difference)
        self.vernier_resolution = self.slow_delay - self.fast_delay
        
        # Number of stages needed
        self.num_stages = int(max_time_range / self.vernier_resolution)
        
        # Generate delay variations
        self.generate_vernier_variations()
    
    def generate_vernier_variations(self):
        """Generate delay variations for Vernier TDC"""
        # Fast chain variations
        fast_sigma = 0.02  # 2% variation
        self.fast_delays = np.random.normal(self.fast_delay, 
                                          fast_sigma * self.fast_delay, 
                                          self.num_stages)
        
        # Slow chain variations
        slow_sigma = 0.02
        self.slow_delays = np.random.normal(self.slow_delay,
                                          slow_sigma * self.slow_delay,
                                          self.num_stages)
        
        # Calculate effective resolution at each stage
        self.effective_resolution = self.slow_delays - self.fast_delays
        self.cumulative_resolution = np.cumsum(self.effective_resolution)
        
        # Calculate DNL and INL
        ideal_resolution = self.vernier_resolution
        self.dnl = (self.effective_resolution - ideal_resolution) / ideal_resolution
        self.inl = (self.cumulative_resolution - 
                   np.arange(len(self.cumulative_resolution)) * ideal_resolution) / ideal_resolution
    
    def convert_time_to_code(self, time_interval):
        """Convert time interval to digital code using Vernier principle"""
        if time_interval < 0:
            return 0
        if time_interval > self.max_time_range:
            return self.max_code
        
        # Simulate Vernier operation
        code = np.searchsorted(self.cumulative_resolution, time_interval)
        code = min(code, self.max_code)
        
        self.conversion_count += 1
        self.measurement_history.append((time_interval, code))
        
        return code

class PipelinedTDC(TDCCore):
    """Pipelined/Cascaded TDC Implementation"""
    
    def __init__(self, resolution_bits=10, max_time_range=10e-9, num_stages=4):
        super().__init__(resolution_bits, max_time_range, 'pipelined')
        
        self.num_stages = num_stages
        self.bits_per_stage = resolution_bits // num_stages
        self.residual_bits = resolution_bits % num_stages
        
        # Generate stage-specific parameters
        self.generate_pipeline_stages()
    
    def generate_pipeline_stages(self):
        """Generate parameters for each pipeline stage"""
        self.stages = []
        current_range = self.max_time_range
        
        for i in range(self.num_stages):
            # Calculate bits for this stage
            stage_bits = self.bits_per_stage
            if i < self.residual_bits:
                stage_bits += 1
            
            # Stage parameters
            stage_range = current_range
            stage_resolution = stage_range / (2**stage_bits)
            
            # Add non-linearity to each stage
            stage_dnl = np.random.normal(0, 0.1, 2**stage_bits)
            stage_inl = np.cumsum(stage_dnl)
            
            stage_params = {
                'bits': stage_bits,
                'range': stage_range,
                'resolution': stage_resolution,
                'dnl': stage_dnl,
                'inl': stage_inl
            }
            
            self.stages.append(stage_params)
            current_range = stage_resolution  # Next stage handles residual
    
    def convert_time_to_code(self, time_interval):
        """Convert time using pipelined approach"""
        if time_interval < 0:
            return 0
        if time_interval > self.max_time_range:
            return self.max_code
        
        total_code = 0
        remaining_time = time_interval
        
        for i, stage in enumerate(self.stages):
            # Quantize in this stage
            stage_code = int(remaining_time / stage['resolution'])
            stage_code = min(stage_code, 2**stage['bits'] - 1)
            
            # Add non-linearity effect
            if stage_code < len(stage['inl']):
                actual_time = stage_code * stage['resolution'] * (1 + stage['inl'][stage_code])
            else:
                actual_time = stage_code * stage['resolution']
            
            # Update total code
            total_code = (total_code << stage['bits']) + stage_code
            
            # Calculate residual for next stage
            remaining_time = remaining_time - actual_time
            if remaining_time < 0:
                remaining_time = 0
        
        self.conversion_count += 1
        self.measurement_history.append((time_interval, total_code))
        
        return total_code

class TDCNonLinearityAnalyzer:
    """Advanced TDC Non-linearity Analysis and Correction"""
    
    def __init__(self, tdc):
        self.tdc = tdc
        self.measurement_data = []
        self.calibration_methods = ['histogram', 'code_density', 'least_squares']
    
    def perform_linearity_test(self, num_measurements=10000, test_type='ramp'):
        """Perform comprehensive linearity testing"""
        print(f"Performing {test_type} linearity test with {num_measurements} measurements...")
        
        measurements = []
        
        if test_type == 'ramp':
            # Linear ramp test
            test_times = np.linspace(0, self.tdc.max_time_range * 0.95, num_measurements)
            
        elif test_type == 'random':
            # Random input test
            test_times = np.random.uniform(0, self.tdc.max_time_range * 0.95, num_measurements)
            
        elif test_type == 'sine':
            # Sinusoidal test
            t = np.linspace(0, 10 * 2 * np.pi, num_measurements)
            test_times = (self.tdc.max_time_range * 0.4) * (1 + 0.9 * np.sin(t))
            
        elif test_type == 'step':
            # Step response test
            test_times = []
            for code in range(0, self.tdc.max_code, max(1, self.tdc.max_code // num_measurements)):
                ideal_time = code * self.tdc.nominal_lsb
                test_times.extend([ideal_time] * 10)  # Multiple measurements per code
            test_times = np.array(test_times[:num_measurements])
        
        # Perform measurements
        for time_input in test_times:
            code_output = self.tdc.convert_time_to_code(time_input)
            measurements.append((time_input, code_output))
        
        self.measurement_data = measurements
        return measurements
    
    def calculate_dnl_inl_from_measurements(self):
        """Calculate DNL/INL from measurement data"""
        if not self.measurement_data:
            print("No measurement data available. Run linearity test first.")
            return None, None
        
        # Create code histogram
        codes = [m[1] for m in self.measurement_data]
        code_counts = np.bincount(codes, minlength=self.tdc.max_code + 1)
        
        # Calculate expected count per code (for uniform input)
        total_measurements = len(self.measurement_data)
        expected_count = total_measurements / (self.tdc.max_code + 1)
        
        # DNL calculation from code density
        dnl_measured = np.zeros(self.tdc.max_code + 1)
        for i in range(len(code_counts)):
            if expected_count > 0:
                dnl_measured[i] = (code_counts[i] - expected_count) / expected_count
        
        # INL calculation (cumulative DNL)
        inl_measured = np.cumsum(dnl_measured)
        
        return dnl_measured, inl_measured
    
    def histogram_calibration(self):
        """Histogram-based calibration method"""
        if not self.measurement_data:
            return None
        
        # Build histogram
        codes = [m[1] for m in self.measurement_data]
        times = [m[0] for m in self.measurement_data]
        
        # Create calibration lookup table
        calibration_table = {}
        
        for code in range(self.tdc.max_code + 1):
            code_times = [t for t, c in zip(times, codes) if c == code]
            if code_times:
                calibration_table[code] = np.mean(code_times)
            else:
                # Interpolate for missing codes
                calibration_table[code] = code * self.tdc.nominal_lsb
        
        return calibration_table
    
    def least_squares_calibration(self, polynomial_order=3):
        """Least squares polynomial calibration"""
        if not self.measurement_data:
            return None
        
        codes = np.array([m[1] for m in self.measurement_data])
        times = np.array([m[0] for m in self.measurement_data])
        
        # Fit polynomial
        coefficients = np.polyfit(codes, times, polynomial_order)
        
        # Create calibration function
        def calibration_func(code):
            return np.polyval(coefficients, code)
        
        return calibration_func, coefficients
    
    def advanced_calibration(self, method='spline'):
        """Advanced calibration using interpolation methods"""
        if not self.measurement_data:
            return None
        
        # Extract unique code-time pairs
        code_time_pairs = {}
        for time_input, code_output in self.measurement_data:
            if code_output not in code_time_pairs:
                code_time_pairs[code_output] = []
            code_time_pairs[code_output].append(time_input)
        
        # Average multiple measurements per code
        codes = []
        avg_times = []
        for code in sorted(code_time_pairs.keys()):
            codes.append(code)
            avg_times.append(np.mean(code_time_pairs[code]))
        
        codes = np.array(codes)
        avg_times = np.array(avg_times)
        
        if method == 'spline':
            # Cubic spline interpolation
            calibration_spline = interpolate.CubicSpline(codes, avg_times)
            return calibration_spline
        
        elif method == 'rbf':
            # Radial basis function interpolation
            from scipy.interpolate import Rbf
            calibration_rbf = Rbf(codes, avg_times, function='multiquadric')
            return calibration_rbf
    
    def apply_calibration(self, calibration_table, input_code):
        """Apply calibration to convert code to calibrated time"""
        if isinstance(calibration_table, dict):
            return calibration_table.get(input_code, input_code * self.tdc.nominal_lsb)
        elif callable(calibration_table):
            return calibration_table(input_code)
        else:
            return input_code * self.tdc.nominal_lsb
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive TDC performance metrics"""
        if not self.measurement_data:
            return {}
        
        times = np.array([m[0] for m in self.measurement_data])
        codes = np.array([m[1] for m in self.measurement_data])
        
        # Calculate metrics
        metrics = {}
        
        # Resolution
        unique_codes = np.unique(codes)
        if len(unique_codes) > 1:
            metrics['effective_resolution'] = np.mean(np.diff(unique_codes)) * self.tdc.nominal_lsb
        
        # Missing codes
        all_codes = set(range(max(codes) + 1))
        present_codes = set(unique_codes)
        missing_codes = all_codes - present_codes
        metrics['missing_codes'] = len(missing_codes)
        metrics['missing_code_rate'] = len(missing_codes) / len(all_codes)
        
        # DNL/INL statistics
        dnl, inl = self.calculate_dnl_inl_from_measurements()
        if dnl is not None:
            metrics['max_dnl'] = np.max(np.abs(dnl))
            metrics['rms_dnl'] = np.sqrt(np.mean(dnl**2))
            metrics['max_inl'] = np.max(np.abs(inl))
            metrics['rms_inl'] = np.sqrt(np.mean(inl**2))
        
        # Noise analysis
        code_std = {}
        for code in unique_codes:
            code_times = times[codes == code]
            if len(code_times) > 1:
                code_std[code] = np.std(code_times)
        
        if code_std:
            metrics['avg_noise'] = np.mean(list(code_std.values()))
            metrics['max_noise'] = np.max(list(code_std.values()))
        
        return metrics

def simulate_tdc_comparison():
    """Compare different TDC architectures"""
    
    # Create different TDC types
    tdcs = {
        'Flash TDC': FlashTDC(resolution_bits=8, max_time_range=5e-9),
        'Vernier TDC': VernerTDC(resolution_bits=8, max_time_range=5e-9),
        'Pipelined TDC': PipelinedTDC(resolution_bits=8, max_time_range=5e-9, num_stages=4)
    }
    
    results = {}
    
    print("Comparing TDC Architectures...")
    
    for name, tdc in tdcs.items():
        print(f"\nAnalyzing {name}...")
        
        # Create analyzer
        analyzer = TDCNonLinearityAnalyzer(tdc)
        
        # Perform tests
        measurements = analyzer.perform_linearity_test(num_measurements=5000, test_type='ramp')
        
        # Calculate metrics
        metrics = analyzer.calculate_performance_metrics()
        
        # Calculate DNL/INL
        dnl, inl = analyzer.calculate_dnl_inl_from_measurements()
        
        # Perform calibration
        calibration_table = analyzer.histogram_calibration()
        poly_cal, poly_coeffs = analyzer.least_squares_calibration()
        
        results[name] = {
            'tdc': tdc,
            'analyzer': analyzer,
            'measurements': measurements,
            'metrics': metrics,
            'dnl': dnl,
            'inl': inl,
            'calibration_table': calibration_table,
            'poly_calibration': poly_cal,
            'poly_coeffs': poly_coeffs
        }
    
    return results

def plot_tdc_analysis(results):
    """Plot comprehensive TDC analysis results"""
    
    num_tdcs = len(results)
    fig, axes = plt.subplots(4, num_tdcs, figsize=(6*num_tdcs, 16))
    
    if num_tdcs == 1:
        axes = axes.reshape(-1, 1)
    
    for col, (tdc_name, data) in enumerate(results.items()):
        tdc = data['tdc']
        dnl = data['dnl']
        inl = data['inl']
        measurements = data['measurements']
        
        # Plot 1: DNL
        axes[0, col].plot(dnl[:len(dnl)//2])  # Plot first half for clarity
        axes[0, col].set_title(f'{tdc_name} - DNL')
        axes[0, col].set_xlabel('Code')
        axes[0, col].set_ylabel('DNL (LSB)')
        axes[0, col].grid(True)
        axes[0, col].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Plot 2: INL
        axes[1, col].plot(inl[:len(inl)//2])
        axes[1, col].set_title(f'{tdc_name} - INL')
        axes[1, col].set_xlabel('Code')
        axes[1, col].set_ylabel('INL (LSB)')
        axes[1, col].grid(True)
        axes[1, col].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Plot 3: Transfer function
        times = [m[0] for m in measurements]
        codes = [m[1] for m in measurements]
        
        # Ideal transfer function
        ideal_times = np.linspace(0, max(times), 100)
        ideal_codes = ideal_times / tdc.nominal_lsb
        
        axes[2, col].scatter(times, codes, alpha=0.5, s=1, label='Measured')
        axes[2, col].plot(ideal_times, ideal_codes, 'r--', label='Ideal')
        axes[2, col].set_title(f'{tdc_name} - Transfer Function')
        axes[2, col].set_xlabel('Input Time (s)')
        axes[2, col].set_ylabel('Output Code')
        axes[2, col].legend()
        axes[2, col].grid(True)
        
        # Plot 4: Code histogram
        code_counts = np.bincount(codes)
        axes[3, col].bar(range(len(code_counts)), code_counts)
        axes[3, col].set_title(f'{tdc_name} - Code Histogram')
        axes[3, col].set_xlabel('Code')
        axes[3, col].set_ylabel('Count')
        axes[3, col].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison table
    print("\n=== TDC Performance Comparison ===")
    print(f"{'Metric':<20} " + " ".join(f"{name:<15}" for name in results.keys()))
    print("-" * (20 + 16 * len(results)))
    
    metrics_to_compare = ['max_dnl', 'rms_dnl', 'max_inl', 'rms_inl', 
                         'missing_code_rate', 'avg_noise']
    
    for metric in metrics_to_compare:
        row = f"{metric:<20} "
        for name, data in results.items():
            value = data['metrics'].get(metric, 'N/A')
            if isinstance(value, float):
                row += f"{value:<15.4f} "
            else:
                row += f"{str(value):<15} "
        print(row)

def demonstrate_calibration_effectiveness(tdc_type='flash'):
    """Demonstrate calibration effectiveness"""
    
    print(f"\nDemonstrating calibration for {tdc_type} TDC...")
    
    # Create TDC
    if tdc_type == 'flash':
        tdc = FlashTDC(resolution_bits=10, max_time_range=10e-9)
    elif tdc_type == 'vernier':
        tdc = VernerTDC(resolution_bits=10, max_time_range=10e-9)
    else:
        tdc = PipelinedTDC(resolution_bits=10, max_time_range=10e-9)
    
    # Create analyzer
    analyzer = TDCNonLinearityAnalyzer(tdc)
    
    # Test without calibration
    test_times = np.linspace(0, tdc.max_time_range * 0.9, 1000)
    uncalibrated_errors = []
    
    for time_input in test_times:
        code = tdc.convert_time_to_code(time_input)
        measured_time = code * tdc.nominal_lsb
        error = measured_time - time_input
        uncalibrated_errors.append(error)
    
    # Perform calibration
    analyzer.perform_linearity_test(num_measurements=5000, test_type='ramp')
    calibration_table = analyzer.histogram_calibration()
    poly_cal, _ = analyzer.least_squares_calibration()
    
    # Test with calibration
    calibrated_errors_hist = []
    calibrated_errors_poly = []
    
    for time_input in test_times:
        code = tdc.convert_time_to_code(time_input)
        
        # Histogram calibration
        cal_time_hist = analyzer.apply_calibration(calibration_table, code)
        error_hist = cal_time_hist - time_input
        calibrated_errors_hist.append(error_hist)
        
        # Polynomial calibration
        cal_time_poly = analyzer.apply_calibration(poly_cal, code)
        error_poly = cal_time_poly - time_input
        calibrated_errors_poly.append(error_poly)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(test_times * 1e12, np.array(uncalibrated_errors) * 1e12)
    plt.title('Uncalibrated Errors')
    plt.xlabel('Input Time (ps)')
    plt.ylabel('Error (ps)')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(test_times * 1e12, np.array(calibrated_errors_hist) * 1e12)
    plt.title('Histogram Calibrated Errors')
    plt.xlabel('Input Time (ps)')
    plt.ylabel('Error (ps)')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(test_times * 1e12, np.array(calibrated_errors_poly) * 1e12)
    plt.title('Polynomial Calibrated Errors')
    plt.xlabel('Input Time (ps)')
    plt.ylabel('Error (ps)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print improvement statistics
    rms_uncal = np.sqrt(np.mean(np.array(uncalibrated_errors)**2))
    rms_cal_hist = np.sqrt(np.mean(np.array(calibrated_errors_hist)**2))
    rms_cal_poly = np.sqrt(np.mean(np.array(calibrated_errors_poly)**2))
    
    print(f"RMS Error Uncalibrated: {rms_uncal * 1e12:.2f} ps")
    print(f"RMS Error Histogram Cal: {rms_cal_hist * 1e12:.2f} ps")
    print(f"RMS Error Polynomial Cal: {rms_cal_poly * 1e12:.2f} ps")
    print(f"Improvement Factor (Histogram): {rms_uncal / rms_cal_hist:.1f}x")
    print(f"Improvement Factor (Polynomial): {rms_uncal / rms_cal_poly:.1f}x")

# Main execution
if __name__ == "__main__":
    # Run TDC comparison
    print("Starting TDC Non-linearity Analysis...")
    results = simulate_tdc_comparison()
    
    # Plot results
    plot_tdc_analysis(results)
    
    # Demonstrate calibration
    demonstrate_calibration_effectiveness('flash')
    
    print("\nTDC Analysis Complete!")