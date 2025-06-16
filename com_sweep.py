import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate, optimize
from scipy.fft import fft, ifft, fftfreq
import pandas as pd

class ERLCOMAnalyzer:
    """
    ERL (Effective Return Loss) and COM (Channel Operating Margin) analyzer
    for high-speed serial links (PCIe, Ethernet, etc.)
    """
    
    def __init__(self, data_rate=25e9, modulation='NRZ', standard='IEEE_802.3'):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        data_rate : float
            Data rate in bits/second (e.g., 25e9 for 25Gbps)
        modulation : str
            'NRZ' or 'PAM4'
        standard : str
            Communication standard (affects masks and requirements)
        """
        self.data_rate = data_rate
        self.symbol_rate = data_rate if modulation == 'NRZ' else data_rate / 2
        self.modulation = modulation
        self.standard = standard
        
        # Nyquist frequency
        self.f_nyquist = self.symbol_rate / 2
        
        # Standard-specific parameters
        self.setup_standard_parameters()
        
        # Analysis results storage
        self.results = {
            'erl': {},
            'com': {},
            'sweep_data': []
        }
        
    def setup_standard_parameters(self):
        """Set up standard-specific parameters and masks."""
        if self.standard == 'IEEE_802.3':
            # IEEE 802.3 parameters for different data rates
            if self.data_rate == 10e9:  # 10GBASE
                self.erl_mask_db = 15  # Minimum ERL
                self.il_fit_db = 2.5   # Insertion loss at Nyquist
                self.com_target = 3     # Target COM in dB
            elif self.data_rate == 25e9:  # 25GBASE
                self.erl_mask_db = 13
                self.il_fit_db = 4.5
                self.com_target = 3
            elif self.data_rate == 50e9:  # 50GBASE
                self.erl_mask_db = 10
                self.il_fit_db = 7
                self.com_target = 3
            elif self.data_rate == 100e9:  # 100GBASE
                self.erl_mask_db = 8
                self.il_fit_db = 10
                self.com_target = 3
        elif self.standard == 'PCIe':
            # PCIe Gen 5/6 parameters
            self.erl_mask_db = 12
            self.il_fit_db = 6
            self.com_target = 3
            
    def calculate_erl(self, s_params, freq, fit_range=(0.1, 0.5)):
        """
        Calculate Effective Return Loss (ERL).
        
        Parameters:
        -----------
        s_params : dict
            S-parameters with keys 'S11', 'S21', etc.
        freq : array
            Frequency points in Hz
        fit_range : tuple
            Frequency range for IL fitting (as fraction of Nyquist)
        
        Returns:
        --------
        erl : array
            ERL values in dB
        erl_fit : array
            Fitted ERL curve
        """
        # Input return loss
        rl = -20 * np.log10(np.abs(s_params['S11']) + 1e-12)
        
        # Insertion loss
        il = -20 * np.log10(np.abs(s_params['S21']) + 1e-12)
        
        # Fit IL in specified range
        fit_mask = (freq >= fit_range[0] * self.f_nyquist) & \
                   (freq <= fit_range[1] * self.f_nyquist)
        
        # Linear fit for IL (in dB)
        fit_coeffs = np.polyfit(freq[fit_mask], il[fit_mask], 1)
        il_fit = np.polyval(fit_coeffs, freq)
        
        # ERL = RL - IL_fit
        erl = rl - il_fit
        
        # Smooth ERL fit
        from scipy.ndimage import gaussian_filter1d
        erl_fit = gaussian_filter1d(erl, sigma=len(freq)//100)
        
        return erl, erl_fit, il_fit
    
    def calculate_com(self, s_params, freq, package_params=None, 
                     noise_params=None, eq_params=None):
        """
        Calculate Channel Operating Margin (COM).
        
        Parameters:
        -----------
        s_params : dict
            Channel S-parameters
        freq : array
            Frequency points
        package_params : dict
            Package model parameters
        noise_params : dict
            Noise specifications
        eq_params : dict
            Equalizer parameters
        
        Returns:
        --------
        com_value : float
            COM in dB
        com_report : dict
            Detailed COM calculation report
        """
        # Default parameters if not provided
        if package_params is None:
            package_params = {
                'loss_db': 1.5,
                'reflection_db': -20,
                'length_mm': 10
            }
            
        if noise_params is None:
            noise_params = {
                'rj_rms': 0.5e-12,  # 0.5ps RMS jitter
                'sj_pp': 0.1,       # 0.1 UI sinusoidal jitter
                'dj_pp': 0.1,       # 0.1 UI duty cycle distortion
                'noise_rms': 5e-3   # 5mV RMS noise
            }
            
        if eq_params is None:
            eq_params = {
                'tx_taps': 3,      # 3-tap TX FFE
                'rx_taps': 5,      # 5-tap RX DFE
                'ctle_poles': 2    # 2-pole CTLE
            }
        
        # Step 1: Combine channel with package model
        combined_loss = self.add_package_effects(s_params, package_params, freq)
        
        # Step 2: Calculate pulse response
        pulse_response = self.calculate_pulse_response(combined_loss, freq)
        
        # Step 3: Apply equalization
        eq_pulse = self.apply_equalization(pulse_response, eq_params)
        
        # Step 4: Calculate eye metrics
        eye_height, eye_width = self.calculate_eye_metrics(eq_pulse, noise_params)
        
        # Step 5: Calculate signal amplitude
        if self.modulation == 'PAM4':
            signal_levels = 4
            voltage_margin_factor = 3  # Three eyes in PAM4
        else:
            signal_levels = 2
            voltage_margin_factor = 1
        
        # Step 6: Calculate COM
        As = 1.0  # Normalized signal amplitude
        An = noise_params['noise_rms'] * 6  # 6-sigma noise
        
        # COM formula
        com_linear = (eye_height * As) / (voltage_margin_factor * An)
        com_db = 20 * np.log10(com_linear)
        
        # Detailed report
        com_report = {
            'com_db': com_db,
            'eye_height': eye_height,
            'eye_width': eye_width,
            'signal_amplitude': As,
            'noise_amplitude': An,
            'eq_pulse_response': eq_pulse,
            'combined_channel_loss': combined_loss
        }
        
        return com_db, com_report
    
    def add_package_effects(self, s_params, package_params, freq):
        """Add package model effects to S-parameters."""
        # Simple package model (can be enhanced)
        pkg_loss = 10**(-package_params['loss_db']/20)
        
        # Frequency-dependent loss
        f_norm = freq / self.f_nyquist
        pkg_transfer = pkg_loss * np.exp(-f_norm * 0.5)  # Simple model
        
        # Combine with channel
        combined_s21 = s_params['S21'] * pkg_transfer
        
        return combined_s21
    
    def calculate_pulse_response(self, transfer_function, freq):
        """Calculate pulse response from transfer function."""
        # Create frequency domain pulse
        N = len(freq)
        df = freq[1] - freq[0]
        
        # Pulse spectrum (sinc function for NRZ)
        T = 1 / self.symbol_rate  # Symbol period
        pulse_spectrum = T * np.sinc(freq * T)
        
        # Apply channel transfer function
        output_spectrum = transfer_function * pulse_spectrum
        
        # Convert to time domain
        # Ensure conjugate symmetry for real output
        full_spectrum = np.zeros(2*N-2, dtype=complex)
        full_spectrum[:N] = output_spectrum
        full_spectrum[N:] = np.conj(output_spectrum[-2:0:-1])
        
        # IFFT
        pulse_response = np.real(ifft(full_spectrum))
        
        # Time vector
        dt = 1 / (2 * freq[-1])
        t = np.arange(len(pulse_response)) * dt
        
        # Extract relevant portion
        symbol_samples = int(T / dt)
        num_symbols = 10
        center = len(pulse_response) // 2
        
        start_idx = center - num_symbols * symbol_samples // 2
        end_idx = center + num_symbols * symbol_samples // 2
        
        pulse_response = pulse_response[start_idx:end_idx]
        t = t[start_idx:end_idx] - t[center]
        
        return pulse_response
    
    def apply_equalization(self, pulse_response, eq_params):
        """Apply TX and RX equalization to pulse response."""
        # Simplified equalization model
        # In practice, this would involve optimization
        
        # TX FFE (pre-emphasis)
        tx_taps = np.zeros(eq_params['tx_taps'])
        tx_taps[0] = 0.7   # Main tap
        tx_taps[1] = 0.3   # Post-cursor
        if eq_params['tx_taps'] > 2:
            tx_taps[2] = -0.1  # Pre-cursor
        
        # Apply TX FFE
        eq_pulse = signal.convolve(pulse_response, tx_taps, mode='same')
        
        # RX CTLE boost (simplified)
        # Boost high frequencies
        ctle_gain = 1 + 0.5 * np.linspace(0, 1, len(eq_pulse))
        
        # Simple frequency domain multiplication would be more accurate
        # but this is a simplified time-domain approximation
        
        return eq_pulse
    
    def calculate_eye_metrics(self, pulse_response, noise_params):
        """Calculate eye height and width from pulse response."""
        # Find peak
        peak_idx = np.argmax(np.abs(pulse_response))
        peak_value = pulse_response[peak_idx]
        
        # Sample at optimal point
        cursor = peak_value
        
        # Find ISI contributions (simplified)
        # In practice, would convolve with PRBS pattern
        pre_cursor_isi = np.sum(np.abs(pulse_response[:peak_idx-10]))
        post_cursor_isi = np.sum(np.abs(pulse_response[peak_idx+10:]))
        
        total_isi = pre_cursor_isi + post_cursor_isi
        
        # Eye height
        eye_height = cursor - total_isi
        
        # Eye width (simplified - based on jitter)
        total_jitter = np.sqrt(noise_params['rj_rms']**2 + 
                              (noise_params['sj_pp']/2)**2 +
                              (noise_params['dj_pp']/2)**2)
        
        eye_width = 1 - 2 * total_jitter  # In UI
        
        return eye_height, eye_width
    
    def sweep_analysis(self, s_param_files, parameter_name='trace_length',
                      parameter_values=None, plot_results=True):
        """
        Perform ERL and COM sweep analysis.
        
        Parameters:
        -----------
        s_param_files : list
            List of S-parameter file paths or S-parameter data
        parameter_name : str
            Name of swept parameter
        parameter_values : list
            Values of swept parameter
        plot_results : bool
            Whether to plot results
        
        Returns:
        --------
        sweep_results : dict
            Results of sweep analysis
        """
        if parameter_values is None:
            parameter_values = range(len(s_param_files))
        
        erl_values = []
        erl_min_values = []
        com_values = []
        
        for i, s_param_data in enumerate(s_param_files):
            # Load S-parameters (simplified - assumes data is provided)
            if isinstance(s_param_data, str):
                # Would load from file in practice
                s_params, freq = self.load_sparams(s_param_data)
            else:
                s_params, freq = s_param_data
            
            # Calculate ERL
            erl, erl_fit, il_fit = self.calculate_erl(s_params, freq)
            
            # Find minimum ERL in band of interest
            band_mask = freq <= self.f_nyquist
            erl_min = np.min(erl_fit[band_mask])
            
            # Calculate COM
            com_value, com_report = self.calculate_com(s_params, freq)
            
            # Store results
            erl_values.append(erl)
            erl_min_values.append(erl_min)
            com_values.append(com_value)
            
            # Store detailed results
            self.results['sweep_data'].append({
                'parameter_value': parameter_values[i],
                'erl': erl,
                'erl_fit': erl_fit,
                'erl_min': erl_min,
                'com': com_value,
                'com_report': com_report,
                'frequency': freq
            })
        
        # Summary results
        sweep_results = {
            'parameter_name': parameter_name,
            'parameter_values': parameter_values,
            'erl_min_values': erl_min_values,
            'com_values': com_values
        }
        
        if plot_results:
            self.plot_sweep_results(sweep_results)
        
        return sweep_results
    
    def plot_sweep_results(self, sweep_results):
        """Plot ERL and COM sweep results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        param_values = sweep_results['parameter_values']
        param_name = sweep_results['parameter_name']
        
        # 1. ERL vs swept parameter
        ax = axes[0, 0]
        ax.plot(param_values, sweep_results['erl_min_values'], 'bo-', 
                linewidth=2, markersize=8, label='Minimum ERL')
        ax.axhline(y=self.erl_mask_db, color='r', linestyle='--', 
                  label=f'Mask ({self.erl_mask_db} dB)')
        ax.set_xlabel(param_name)
        ax.set_ylabel('Minimum ERL (dB)')
        ax.set_title('ERL vs ' + param_name)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Check pass/fail
        passing = np.array(sweep_results['erl_min_values']) >= self.erl_mask_db
        if np.any(passing):
            ax.fill_between(param_values, 0, 1, where=passing, 
                          transform=ax.get_xaxis_transform(),
                          alpha=0.2, color='green', label='Pass region')
        
        # 2. COM vs swept parameter
        ax = axes[0, 1]
        ax.plot(param_values, sweep_results['com_values'], 'go-', 
                linewidth=2, markersize=8, label='COM')
        ax.axhline(y=self.com_target, color='r', linestyle='--', 
                  label=f'Target ({self.com_target} dB)')
        ax.set_xlabel(param_name)
        ax.set_ylabel('COM (dB)')
        ax.set_title('COM vs ' + param_name)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 3. ERL frequency response for all sweeps
        ax = axes[1, 0]
        cmap = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
        
        for i, data in enumerate(self.results['sweep_data']):
            freq_ghz = data['frequency'] / 1e9
            ax.plot(freq_ghz, data['erl_fit'], color=cmap[i], 
                   label=f'{param_name}={param_values[i]}')
        
        ax.axhline(y=self.erl_mask_db, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('ERL (dB)')
        ax.set_title('ERL Frequency Response')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, self.f_nyquist/1e9])
        if len(param_values) <= 6:
            ax.legend()
        
        # 4. Combined Pass/Fail analysis
        ax = axes[1, 1]
        
        # Create pass/fail matrix
        erl_pass = np.array(sweep_results['erl_min_values']) >= self.erl_mask_db
        com_pass = np.array(sweep_results['com_values']) >= self.com_target
        combined_pass = erl_pass & com_pass
        
        # Plot pass/fail regions
        x = np.array(param_values)
        ax.fill_between(x, 0, 1, where=erl_pass, alpha=0.3, 
                       color='blue', label='ERL Pass')
        ax.fill_between(x, 0, 1, where=com_pass, alpha=0.3, 
                       color='green', label='COM Pass')
        ax.fill_between(x, 0, 1, where=combined_pass, alpha=0.5, 
                       color='gold', label='Both Pass')
        
        ax.set_xlabel(param_name)
        ax.set_ylabel('Pass/Fail Status')
        ax.set_title('Combined Pass/Fail Analysis')
        ax.set_ylim([-0.1, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text annotations
        if np.any(combined_pass):
            pass_range = x[combined_pass]
            ax.text(np.mean(pass_range), 0.5, 
                   f'Operating Range:\n{pass_range[0]:.1f} to {pass_range[-1]:.1f}',
                   ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    def generate_test_sparams(self, trace_length_inch=10, num_points=1001):
        """
        Generate test S-parameters for demonstration.
        
        Parameters:
        -----------
        trace_length_inch : float
            PCB trace length in inches
        num_points : int
            Number of frequency points
        
        Returns:
        --------
        s_params : dict
            Generated S-parameters
        freq : array
            Frequency points
        """
        # Frequency points
        freq = np.linspace(0, 2 * self.f_nyquist, num_points)
        
        # Loss model (dB/inch/sqrt(GHz))
        loss_tangent = 0.02
        dk = 3.5  # Dielectric constant
        
        # Insertion loss
        alpha_dc = 0.1 * trace_length_inch  # DC loss
        alpha_skin = 0.2 * trace_length_inch * np.sqrt(freq/1e9)  # Skin effect
        alpha_diel = 0.1 * trace_length_inch * freq/1e9  # Dielectric loss
        
        il_db = alpha_dc + alpha_skin + alpha_diel
        s21 = 10**(-il_db/20) * np.exp(-1j * 2 * np.pi * freq * 
                                       trace_length_inch * 0.15e-9)  # Delay
        
        # Return loss (simplified model)
        # Impedance variations cause reflections
        z_var = 5  # ±5 ohm variation
        gamma = z_var / 100  # Reflection coefficient
        
        # Add some resonances
        resonance_freq = 5e9
        q_factor = 20
        resonance = 1 / (1 + 1j * q_factor * (freq/resonance_freq - 
                                             resonance_freq/freq))
        
        s11 = gamma * (0.1 + 0.9 * resonance)
        
        # Package resonance
        pkg_resonance_freq = 15e9
        pkg_resonance = 0.05 / (1 + 1j * 10 * (freq/pkg_resonance_freq - 
                                              pkg_resonance_freq/freq))
        s11 += pkg_resonance
        
        s_params = {
            'S11': s11,
            'S21': s21,
            'S12': s21,  # Reciprocal
            'S22': s11   # Symmetric
        }
        
        return s_params, freq
    
    def optimization_example(self, design_parameters):
        """
        Example optimization of design parameters for ERL/COM targets.
        
        Parameters:
        -----------
        design_parameters : dict
            Initial design parameters
        
        Returns:
        --------
        optimal_params : dict
            Optimized parameters
        """
        # Define cost function
        def cost_function(params):
            trace_length = params[0]
            via_count = int(params[1])
            
            # Generate S-parameters based on parameters
            s_params, freq = self.generate_test_sparams(trace_length)
            
            # Add via effects (simplified)
            via_cap = 0.1e-12 * via_count  # 0.1pF per via
            via_ind = 1e-9 * via_count     # 1nH per via
            
            # Calculate metrics
            erl, erl_fit, _ = self.calculate_erl(s_params, freq)
            com_value, _ = self.calculate_com(s_params, freq)
            
            # Cost: minimize if both pass, heavily penalize if fail
            erl_min = np.min(erl_fit[freq <= self.f_nyquist])
            
            cost = 0
            if erl_min < self.erl_mask_db:
                cost += 100 * (self.erl_mask_db - erl_min)**2
            if com_value < self.com_target:
                cost += 100 * (self.com_target - com_value)**2
                
            # Add preference for shorter traces
            cost += 0.1 * trace_length
            
            return cost
        
        # Initial guess
        x0 = [design_parameters['trace_length'], design_parameters['via_count']]
        
        # Bounds
        bounds = [(1, 30),   # Trace length 1-30 inches
                 (0, 10)]    # Via count 0-10
        
        # Optimize
        result = optimize.minimize(cost_function, x0, method='L-BFGS-B', 
                                 bounds=bounds)
        
        optimal_params = {
            'trace_length': result.x[0],
            'via_count': int(result.x[1]),
            'cost': result.fun,
            'success': result.success
        }
        
        return optimal_params


# Example usage
if __name__ == "__main__":
    # Create analyzer for 25Gbps NRZ
    analyzer = ERLCOMAnalyzer(data_rate=25e9, modulation='NRZ')
    
    # Generate test S-parameters for different trace lengths
    trace_lengths = np.linspace(5, 25, 9)  # 5 to 25 inches
    s_param_data = []
    
    print("Generating test S-parameters...")
    for length in trace_lengths:
        s_params, freq = analyzer.generate_test_sparams(trace_length_inch=length)
        s_param_data.append((s_params, freq))
    
    # Perform sweep analysis
    print("\nPerforming ERL and COM sweep analysis...")
    sweep_results = analyzer.sweep_analysis(
        s_param_files=s_param_data,
        parameter_name='Trace Length (inches)',
        parameter_values=trace_lengths,
        plot_results=True
    )
    
    # Print summary
    print("\n=== SWEEP ANALYSIS SUMMARY ===")
    print(f"Standard: {analyzer.standard}")
    print(f"Data Rate: {analyzer.data_rate/1e9:.0f} Gbps")
    print(f"ERL Mask: {analyzer.erl_mask_db} dB")
    print(f"COM Target: {analyzer.com_target} dB")
    
    print("\nResults:")
    for i, length in enumerate(trace_lengths):
        erl_pass = sweep_results['erl_min_values'][i] >= analyzer.erl_mask_db
        com_pass = sweep_results['com_values'][i] >= analyzer.com_target
        
        print(f"  {length:.1f} inches: ERL={sweep_results['erl_min_values'][i]:.1f} dB "
              f"{'(PASS)' if erl_pass else '(FAIL)'}, "
              f"COM={sweep_results['com_values'][i]:.1f} dB "
              f"{'(PASS)' if com_pass else '(FAIL)'}")
    
    # Find operating range
    erl_pass = np.array(sweep_results['erl_min_values']) >= analyzer.erl_mask_db
    com_pass = np.array(sweep_results['com_values']) >= analyzer.com_target
    both_pass = erl_pass & com_pass
    
    if np.any(both_pass):
        pass_indices = np.where(both_pass)[0]
        min_length = trace_lengths[pass_indices[0]]
        max_length = trace_lengths[pass_indices[-1]]
        print(f"\nOperating Range: {min_length:.1f} to {max_length:.1f} inches")
    else:
        print("\nNo operating range found - design does not meet requirements")
    
    # PAM4 analysis example
    print("\n\n=== PAM4 ANALYSIS ===")
    pam4_analyzer = ERLCOMAnalyzer(data_rate=50e9, modulation='PAM4')
    
    # Single point analysis
    s_params, freq = pam4_analyzer.generate_test_sparams(trace_length_inch=10)
    
    # Calculate metrics
    erl, erl_fit, il_fit = pam4_analyzer.calculate_erl(s_params, freq)
    com_value, com_report = pam4_analyzer.calculate_com(s_params, freq)
    
    print(f"PAM4 50Gbps @ 10 inches:")
    print(f"  Minimum ERL: {np.min(erl_fit[freq <= pam4_analyzer.f_nyquist]):.1f} dB")
    print(f"  COM: {com_value:.1f} dB")
    print(f"  Eye Height: {com_report['eye_height']:.3f}")
    print(f"  Eye Width: {com_report['eye_width']:.3f} UI")
    
    # Design optimization example
    print("\n\n=== DESIGN OPTIMIZATION ===")
    initial_design = {
        'trace_length': 15,  # inches
        'via_count': 4
    }
    
    print("Optimizing design parameters...")
    optimal = analyzer.optimization_example(initial_design)
    
    print(f"Initial: Length={initial_design['trace_length']}in, "
          f"Vias={initial_design['via_count']}")
    print(f"Optimal: Length={optimal['trace_length']:.1f}in, "
          f"Vias={optimal['via_count']}")
    print(f"Optimization {'succeeded' if optimal['success'] else 'failed'}")