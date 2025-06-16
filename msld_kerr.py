import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.special import erfc

class OpticalMLSDReceiver:
    """
    Maximum Likelihood Sequence Detection receiver for optical communication
    systems with non-linear effects.
    """
    
    def __init__(self, modulation_order=2, channel_memory=3, 
                 nonlinearity_coefficient=0.1, noise_variance=0.01):
        """
        Initialize MLSD receiver.
        
        Parameters:
        -----------
        modulation_order : int
            Number of modulation levels (2 for OOK, 4 for PAM-4, etc.)
        channel_memory : int
            Memory length of the channel (ISI span)
        nonlinearity_coefficient : float
            Strength of non-linear effects (γ parameter)
        noise_variance : float
            Variance of additive noise
        """
        self.M = modulation_order
        self.L = channel_memory
        self.gamma = nonlinearity_coefficient
        self.sigma2 = noise_variance
        
        # Generate constellation points
        if self.M == 2:  # OOK
            self.constellation = np.array([0, 1])
        else:  # PAM
            self.constellation = np.linspace(0, 1, self.M)
        
        # Number of states in trellis
        self.num_states = self.M ** self.L
        
    def optical_channel_model(self, symbols, h_linear):
        """
        Model optical channel with linear and non-linear effects.
        
        Parameters:
        -----------
        symbols : array
            Transmitted symbols
        h_linear : array
            Linear channel impulse response
        
        Returns:
        --------
        received : array
            Received signal after channel effects
        """
        # Linear convolution (ISI)
        linear_output = np.convolve(symbols, h_linear, mode='same')
        
        # Non-linear phase modulation (simplified Kerr effect)
        # In fiber: φ_NL = γ * P * L_eff
        power = np.abs(linear_output) ** 2
        nonlinear_phase = self.gamma * power
        
        # Apply non-linear phase rotation
        received = linear_output * np.exp(1j * nonlinear_phase)
        
        return received
    
    def compute_branch_metrics(self, received_sample, channel_response, 
                              current_state, next_symbol):
        """
        Compute branch metric for Viterbi algorithm.
        
        Parameters:
        -----------
        received_sample : complex
            Current received sample
        channel_response : array
            Channel impulse response
        current_state : int
            Current trellis state
        next_symbol : float
            Symbol associated with transition
        
        Returns:
        --------
        metric : float
            Branch metric (negative log-likelihood)
        """
        # Decode state to symbol history
        state_symbols = self._decode_state(current_state)
        
        # Append next symbol
        symbol_sequence = np.append(state_symbols, next_symbol)
        
        # Compute expected received value (with non-linearity)
        expected = self._compute_expected_value(symbol_sequence, channel_response)
        
        # Euclidean distance metric
        metric = np.abs(received_sample - expected) ** 2 / (2 * self.sigma2)
        
        return metric
    
    def _compute_expected_value(self, symbols, channel_response):
        """
        Compute expected received value including non-linear effects.
        """
        # Linear convolution
        linear_term = np.sum(symbols[-len(channel_response):] * channel_response[::-1])
        
        # Simplified non-linear term
        power = np.abs(linear_term) ** 2
        nonlinear_phase = self.gamma * power
        
        expected = linear_term * np.exp(1j * nonlinear_phase)
        
        return expected
    
    def viterbi_algorithm(self, received_signal, channel_response):
        """
        Perform Viterbi algorithm for MLSD.
        
        Parameters:
        -----------
        received_signal : array
            Received signal samples
        channel_response : array
            Estimated channel impulse response
        
        Returns:
        --------
        detected_symbols : array
            Detected symbol sequence
        """
        N = len(received_signal)
        
        # Initialize trellis
        path_metrics = np.full(self.num_states, np.inf)
        path_metrics[0] = 0  # Start from zero state
        
        # Store survivor paths
        survivors = np.zeros((N, self.num_states), dtype=int)
        
        # Forward recursion
        for n in range(N):
            new_metrics = np.full(self.num_states, np.inf)
            
            for current_state in range(self.num_states):
                if path_metrics[current_state] == np.inf:
                    continue
                
                # Try all possible next symbols
                for symbol_idx, symbol in enumerate(self.constellation):
                    # Compute next state
                    next_state = self._compute_next_state(current_state, symbol_idx)
                    
                    # Compute branch metric
                    branch_metric = self.compute_branch_metrics(
                        received_signal[n], channel_response, 
                        current_state, symbol
                    )
                    
                    # Update path metric
                    candidate_metric = path_metrics[current_state] + branch_metric
                    
                    if candidate_metric < new_metrics[next_state]:
                        new_metrics[next_state] = candidate_metric
                        survivors[n, next_state] = current_state
            
            path_metrics = new_metrics
        
        # Backward recursion (traceback)
        final_state = np.argmin(path_metrics)
        detected_sequence = self._traceback(survivors, final_state)
        
        return detected_sequence
    
    def _encode_state(self, symbols):
        """
        Encode symbol sequence into state number.
        """
        state = 0
        for i, symbol in enumerate(symbols):
            symbol_idx = np.argmin(np.abs(self.constellation - symbol))
            state += symbol_idx * (self.M ** i)
        return state
    
    def _decode_state(self, state):
        """
        Decode state number into symbol sequence.
        """
        symbols = []
        temp_state = state
        for _ in range(self.L):
            symbol_idx = temp_state % self.M
            symbols.append(self.constellation[symbol_idx])
            temp_state //= self.M
        return np.array(symbols)
    
    def _compute_next_state(self, current_state, new_symbol_idx):
        """
        Compute next state given current state and new symbol.
        """
        # Shift register operation
        next_state = (current_state // self.M) + new_symbol_idx * (self.M ** (self.L - 1))
        return next_state
    
    def _traceback(self, survivors, final_state):
        """
        Traceback through survivor paths to find detected sequence.
        """
        N = survivors.shape[0]
        detected_indices = np.zeros(N, dtype=int)
        
        state = final_state
        for n in range(N-1, -1, -1):
            # Extract symbol from state transition
            prev_state = survivors[n, state]
            symbol_idx = state // (self.M ** (self.L - 1))
            detected_indices[n] = symbol_idx
            state = prev_state
        
        # Convert indices to symbols
        detected_symbols = self.constellation[detected_indices]
        
        return detected_symbols
    
    def simulate_ber(self, num_bits=10000, snr_db_range=np.arange(0, 20, 2)):
        """
        Simulate bit error rate performance.
        
        Parameters:
        -----------
        num_bits : int
            Number of bits to simulate
        snr_db_range : array
            Range of SNR values in dB
        
        Returns:
        --------
        ber : array
            Bit error rates for each SNR
        """
        # Simple channel response (you can make this more realistic)
        h = np.array([0.8, 0.5, 0.3])  # Example dispersive channel
        
        ber_results = []
        
        for snr_db in snr_db_range:
            # Convert SNR to linear scale
            snr_linear = 10 ** (snr_db / 10)
            
            # Adjust noise variance
            self.sigma2 = 1 / (2 * snr_linear)
            
            errors = 0
            
            # Generate random bits
            if self.M == 2:
                bits = np.random.randint(0, 2, num_bits)
                tx_symbols = self.constellation[bits]
            else:
                # For higher order modulation
                bits_per_symbol = int(np.log2(self.M))
                num_symbols = num_bits // bits_per_symbol
                symbol_indices = np.random.randint(0, self.M, num_symbols)
                tx_symbols = self.constellation[symbol_indices]
            
            # Pass through channel
            rx_signal = self.optical_channel_model(tx_symbols, h)
            
            # Add noise
            noise = np.sqrt(self.sigma2/2) * (np.random.randn(len(rx_signal)) + 
                                              1j * np.random.randn(len(rx_signal)))
            rx_signal += noise
            
            # Detect using MLSD
            detected_symbols = self.viterbi_algorithm(np.real(rx_signal), h)
            
            # Count bit errors
            if self.M == 2:
                detected_bits = (detected_symbols > 0.5).astype(int)
                errors = np.sum(bits != detected_bits[:len(bits)])
            else:
                # Convert symbols back to bits for higher order modulation
                detected_indices = np.argmin(np.abs(detected_symbols[:, None] - 
                                                   self.constellation), axis=1)
                detected_bits = np.unpackbits(detected_indices.astype(np.uint8))
                errors = np.sum(bits != detected_bits[:num_bits])
            
            ber = errors / num_bits
            ber_results.append(ber)
            
            print(f"SNR: {snr_db} dB, BER: {ber:.6f}")
        
        return np.array(ber_results)


# Example usage and visualization
if __name__ == "__main__":
    # Create MLSD receiver
    receiver = OpticalMLSDReceiver(
        modulation_order=2,  # OOK modulation
        channel_memory=3,
        nonlinearity_coefficient=0.05,
        noise_variance=0.01
    )
    
    # Simulate BER performance
    print("Simulating BER performance...")
    snr_range = np.arange(0, 16, 2)
    ber_mlsd = receiver.simulate_ber(num_bits=5000, snr_db_range=snr_range)
    
    # Compare with simple threshold detection (no MLSD)
    ber_threshold = []
    for snr_db in snr_range:
        # Theoretical BER for OOK with threshold detection
        snr_linear = 10 ** (snr_db / 10)
        ber_theory = 0.5 * erfc(np.sqrt(snr_linear/2))
        ber_threshold.append(ber_theory)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_mlsd, 'bo-', label='MLSD with Non-linearity', 
                 linewidth=2, markersize=8)
    plt.semilogy(snr_range, ber_threshold, 'rs--', label='Threshold Detection (Theory)', 
                 linewidth=2, markersize=6)
    
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('MLSD Performance in Optical System with Non-linearity', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim([1e-6, 1])
    
    plt.tight_layout()
    plt.show()
    
    # Visualize trellis structure
    plt.figure(figsize=(12, 8))
    
    # Simple trellis diagram for 2-PAM with memory L=2
    states = ['00', '01', '10', '11']
    time_steps = 5
    
    for t in range(time_steps):
        for i, state in enumerate(states):
            plt.plot(t, i, 'ko', markersize=10)
            plt.text(t, i+0.1, state, ha='center', fontsize=9)
            
            if t < time_steps - 1:
                # Draw transitions
                for next_symbol in [0, 1]:
                    next_state_idx = (i // 2) + next_symbol * 2
                    plt.arrow(t, i, 0.9, next_state_idx - i, 
                             head_width=0.05, head_length=0.05, 
                             fc='blue' if next_symbol == 0 else 'red', 
                             ec='blue' if next_symbol == 0 else 'red',
                             alpha=0.6)
    
    plt.xlim(-0.5, time_steps - 0.5)
    plt.ylim(-0.5, len(states) - 0.5)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('State', fontsize=12)
    plt.title('Trellis Structure for MLSD (Binary, L=2)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add legend
    blue_arrow = plt.Line2D([0], [0], color='blue', linewidth=2, label='Symbol 0')
    red_arrow = plt.Line2D([0], [0], color='red', linewidth=2, label='Symbol 1')
    plt.legend(handles=[blue_arrow, red_arrow], loc='upper right')
    
    plt.tight_layout()
    plt.show()