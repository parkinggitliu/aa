import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class PAM4_TED_Comparison:
    def __init__(self, samples_per_symbol=8):
        """
        Initialize PAM4 timing error detector comparison
        
        Parameters:
        - samples_per_symbol: Oversampling ratio
        """
        self.sps = samples_per_symbol
        self.levels = [-3, -1, 1, 3]  # PAM4 levels
        
        # History buffers for each TED
        self.mm_buffer = np.zeros(4)  # Mueller-Muller needs 4 samples
        self.gardner_buffer = np.zeros(3)  # Gardner needs 3 samples
        self.alex_buffer = np.zeros(2)  # Alexander needs 2 samples
        self.mmse_buffer = np.zeros(5)  # MMSE needs more samples
        
        # For plotting results
        self.errors = {'MM': [], 'Gardner': [], 'Alexander': [], 'MMSE': []}
        self.true_phases = []
        
    def slice_pam4(self, sample):
        """Slice to nearest PAM4 level"""
        return self.levels[np.argmin(np.abs(np.array(self.levels) - sample))]
    
    def mueller_muller_ted(self):
        """
        PAM4 Mueller-Muller TED
        TED = y[n-1]*(x[n] - x[n-2]) - y[n]*(x[n-1] - x[n-3])
        """
        if len(self.mm_buffer) < 4:
            return 0
        return (self.mm_buffer[-2] * (self.mm_buffer[0] - self.mm_buffer[-4]) - 
                self.mm_buffer[0] * (self.mm_buffer[-1] - self.mm_buffer[-3]))
    
    def gardner_ted(self):
        """
        Gardner TED (modified for PAM4)
        TED = (x[n] - x[n-2]) * x[n-1]
        """
        if len(self.gardner_buffer) < 3:
            return 0
        return (self.gardner_buffer[0] - self.gardner_buffer[-2]) * self.gardner_buffer[-1]
    
    def alexander_ted(self):
        """
        Alexander TED (modified for PAM4)
        TED = y[n]*(x[n-1] - x[n+1])
        """
        if len(self.alex_buffer) < 2:
            return 0
        y = self.slice_pam4(self.alex_buffer[0])
        return y * (self.alex_buffer[-1] - self.alex_buffer[0])
    
    def mmse_ted(self, phase_estimate):
        """
        Minimum Mean Square Error (MMSE) TED
        TED = Σ (x[n] - ŷ[n]) * ∂ŷ[n]/∂τ
        Where ŷ[n] is the estimated sample at current phase
        """
        if len(self.mmse_buffer) < 5:
            return 0
            
        # Estimate the ideal sample (using interpolation)
        t = np.arange(-2, 3) - phase_estimate
        h = np.sinc(t) * np.hamming(5)  # Simple interpolator
        y_hat = np.dot(self.mmse_buffer, h)
        
        # Estimate derivative
        dh = np.cos(np.pi*t)/(np.pi*t) - np.sinc(t)  # Derivative of sinc
        dh[np.abs(t) < 1e-6] = 0  # Handle divide by zero
        dh *= np.hamming(5)  # Same window
        dy_dtau = np.dot(self.mmse_buffer, dh)
        
        # Calculate error
        y = self.slice_pam4(y_hat)
        return (y_hat - y) * dy_dtau
    
    def update_buffers(self, sample):
        """Update all TED buffers"""
        self.mm_buffer = np.roll(self.mm_buffer, -1)
        self.mm_buffer[-1] = sample
        
        self.gardner_buffer = np.roll(self.gardner_buffer, -1)
        self.gardner_buffer[-1] = sample
        
        self.alex_buffer = np.roll(self.alex_buffer, -1)
        self.alex_buffer[-1] = sample
        
        self.mmse_buffer = np.roll(self.mmse_buffer, -1)
        self.mmse_buffer[-1] = sample
    
    def evaluate_teds(self, phase_estimate):
        """Calculate all TED outputs"""
        mm_error = self.mueller_muller_ted()
        gardner_error = self.gardner_ted()
        alex_error = self.alexander_ted()
        mmse_error = self.mmse_ted(phase_estimate)
        
        return {
            'MM': mm_error,
            'Gardner': gardner_error,
            'Alexander': alex_error,
            'MMSE': mmse_error
        }
    
    def run_simulation(self, num_symbols=1000, snr=20, phase_offset=0.3):
        """Run comparison simulation"""
        # Generate PAM4 signal
        tx_symbols = np.random.choice(self.levels, num_symbols)
        
        # Create oversampled signal with timing offset
        samples = np.zeros(num_symbols * self.sps)
        for i, sym in enumerate(tx_symbols):
            pos = int(i*self.sps + phase_offset*self.sps)
            if pos < len(samples):
                samples[pos] = sym
        
        # Apply pulse shaping (rectangular)
        pulse = np.ones(self.sps)
        samples = np.convolve(samples, pulse, mode='same')
        
        # Add noise
        noise_power = 10**(-snr/10) * np.var(self.levels)
        samples += np.random.normal(0, np.sqrt(noise_power), len(samples))
        
        # Process samples
        for i, sample in enumerate(samples):
            self.update_buffers(sample)
            
            # Only evaluate at symbol centers (for comparison)
            if i % self.sps == int(self.sps/2):
                phase_est = (i % self.sps) / self.sps
                errors = self.evaluate_teds(phase_est)
                
                for k, v in errors.items():
                    self.errors[k].append(v)
                self.true_phases.append(phase_est - 0.5)  # Center at 0
        
        # Plot results
        self.plot_results()
    
    def plot_results(self):
        """Plot comparison of TED performances"""
        plt.figure(figsize=(12, 8))
        
        for i, (name, errors) in enumerate(self.errors.items()):
            plt.subplot(2, 2, i+1)
            plt.plot(self.true_phases, errors, '.', alpha=0.5)
            plt.title(f'{name} TED Response')
            plt.xlabel('True Phase Error')
            plt.ylabel('TED Output')
            plt.grid(True)
            
            # Fit linear curve to show S-curve
            if len(errors) > 10:
                coeff = np.polyfit(self.true_phases, errors, 1)
                x = np.linspace(min(self.true_phases), max(self.true_phases), 100)
                plt.plot(x, np.polyval(coeff, x), 'r-', linewidth=2)
        
        plt.tight_layout()
        plt.show()

# Run comparison
comparison = PAM4_TED_Comparison(samples_per_symbol=8)
comparison.run_simulation(num_symbols=500, snr=15, phase_offset=0.4)