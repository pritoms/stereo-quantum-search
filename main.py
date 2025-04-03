import math
import random
import matplotlib.pyplot as plt
import numpy as np

class StereoQubit:
    def __init__(self, phase_precision=1.0):
        self.channelA = {"amplitude": 1, "phase": 0}
        self.channelB = {"amplitude": 1, "phase": 0}
        self.phase_precision = phase_precision  # Precision in degrees
        
    def applyPhaseShift(self, degrees):
        # Apply phase shift with limited precision and potential noise
        if self.phase_precision > 0:
            # Quantize to the nearest precision level
            quantized_degrees = round(degrees / self.phase_precision) * self.phase_precision
            self.channelB["phase"] = (self.channelB["phase"] + quantized_degrees) % 360
        return self
        
    def applyPhaseShiftWithNoise(self, degrees, noise_level=0):
        # Add random noise to the phase shift
        if noise_level > 0:
            noise = random.uniform(-noise_level, noise_level)
            degrees += noise
        self.applyPhaseShift(degrees)
        return self

    def applySuperposition(self):
        self.applyPhaseShift(90)
        return self

    def getPhantomCenter(self):
        phaseDiff = (self.channelB["phase"] - self.channelA["phase"]) % 360
        if phaseDiff > 180:
            phaseDiff -= 360
        return phaseDiff / 180

    def getProbability(self):
        phaseDiff = abs((self.channelB["phase"] - self.channelA["phase"]) % 360)
        if phaseDiff > 180:
            phaseDiff = 360 - phaseDiff
        return phaseDiff / 180

    def __str__(self):
        return (
            f"Channel A: {self.channelA['amplitude']}∠{self.channelA['phase']:.1f}° | "
            f"Channel B: {self.channelB['amplitude']}∠{self.channelB['phase']:.1f}° | "
            f"Phantom: {self.getPhantomCenter():.2f} | "
            f"P(|1⟩): {self.getProbability() * 100:.1f}%"
        )

class StereoRegister:
    def __init__(self, size, phase_precision=1.0, error_correction=None):
        self.qubits = [StereoQubit(phase_precision) for _ in range(size)]
        self.size = size
        self.phase_precision = phase_precision
        self.error_correction = error_correction
        self.phase_history = []  # Track phase changes for analysis

    def initialize(self):
        for q in self.qubits:
            q.applySuperposition()
        self.recordState()
        return self

    def recordState(self):
        # Record current phase states for later analysis
        state = []
        for q in self.qubits:
            state.append({
                'channelA_phase': q.channelA["phase"],
                'channelB_phase': q.channelB["phase"],
                'phantom': q.getPhantomCenter(),
                'probability': q.getProbability()
            })
        self.phase_history.append(state)

    def applyOracle(self, targetIndex, noise_level=0):
        if 0 <= targetIndex < self.size:
            if noise_level > 0:
                self.qubits[targetIndex].applyPhaseShiftWithNoise(180, noise_level)
            else:
                self.qubits[targetIndex].applyPhaseShift(180)
        
        if self.error_correction == 'phase_tracking':
            self.applyPhaseTracking()
            
        self.recordState()
        return self

    def applyPhaseTracking(self):
        # Simple phase tracking and correction
        # Uses a reference signal to detect and correct drift
        reference_phase = 0  # Ideal reference phase
        for q in self.qubits:
            channelA_drift = q.channelA["phase"] - reference_phase
            if abs(channelA_drift) > self.phase_precision:
                # Correct drift in channel A
                q.channelA["phase"] = reference_phase

    def applyDiffusion(self, noise_level=0):
        # Calculate average phantom center
        avgCenter = sum(q.getPhantomCenter() for q in self.qubits) / self.size
        
        for q in self.qubits:
            currentCenter = q.getPhantomCenter()
            diffFromAvg = currentCenter - avgCenter
            
            # Apply diffusion with possible noise
            if noise_level > 0:
                q.applyPhaseShiftWithNoise(-diffFromAvg * 90, noise_level)
            else:
                q.applyPhaseShift(-diffFromAvg * 90)
                
        if self.error_correction == 'redundancy':
            self.applyRedundancyCorrection()
            
        self.recordState()
        return self
        
    def applyRedundancyCorrection(self):
        # Simple redundancy-based error correction
        # For demonstration - in a real system this would use multiple physical channels
        # Here we just smooth out extreme values
        phantom_centers = [q.getPhantomCenter() for q in self.qubits]
        median_center = sorted(phantom_centers)[len(phantom_centers)//2]
        
        for i, q in enumerate(self.qubits):
            # If a phantom center is very far from the median, adjust it
            if abs(q.getPhantomCenter() - median_center) > 0.5:
                # Move it slightly toward the median
                current = q.getPhantomCenter()
                correction = (median_center - current) * 0.3  # Partial correction
                q.applyPhaseShift(correction * 90)

    def measure(self):
        probabilities = [q.getProbability() for q in self.qubits]
        return probabilities.index(max(probabilities))

    def display(self):
        return "\n".join(f"Qubit {i}: {q}" for i, q in enumerate(self.qubits))
        
    def plotPhaseHistory(self):
        # Plot the phase history for analysis
        plt.figure(figsize=(12, 8))
        
        # Phantom center history
        plt.subplot(2, 1, 1)
        for i in range(self.size):
            phantom_values = [state[i]['phantom'] for state in self.phase_history]
            plt.plot(phantom_values, label=f'Qubit {i}')
            
        plt.title('Phantom Center Evolution')
        plt.xlabel('Algorithm Step')
        plt.ylabel('Phantom Center Value')
        plt.legend()
        plt.grid(True)
        
        # Probability history
        plt.subplot(2, 1, 2)
        for i in range(self.size):
            prob_values = [state[i]['probability'] for state in self.phase_history]
            plt.plot(prob_values, label=f'Qubit {i}')
            
        plt.title('Probability Evolution')
        plt.xlabel('Algorithm Step')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        return plt

def runStereoSearch(registerSize, targetIndex, iterations, phase_precision=1.0, noise_level=0, 
                   error_correction=None, verbose=True):
    if verbose:
        print(f"\n=== STEREO QUANTUM SEARCH SIMULATION ===")
        print(f"Database size: {registerSize}, Target index: {targetIndex}")
        print(f"Phase precision: {phase_precision}°, Noise level: {noise_level}°")
        print(f"Error correction: {error_correction if error_correction else 'None'}")
        print(f"Optimal iterations: ~{math.floor(math.pi/4 * math.sqrt(registerSize))}")
        print(f"Requested iterations: {iterations}")

    register = StereoRegister(registerSize, phase_precision, error_correction).initialize()

    if verbose:
        print(f"\n--- Initial State (after initialization) ---")
        print(register.display())

    for i in range(iterations):
        register.applyOracle(targetIndex, noise_level)

        if verbose:
            print(f"\n--- After Oracle (iteration {i+1}) ---")
            print(register.display())

        register.applyDiffusion(noise_level)

        if verbose:
            print(f"\n--- After Diffusion (iteration {i+1}) ---")
            print(register.display())

    result = register.measure()

    if verbose:
        print(f"\n--- MEASUREMENT RESULT ---")
        print(f"Measured index: {result}")
        print(f"Target index: {targetIndex}")
        print(f"Success: {'YES' if result == targetIndex else 'NO'}")

    return {
        'result': result,
        'success': result == targetIndex,
        'register': register
    }

def analyzePhaseRequirements(max_register_size=10, trials_per_config=10):
    """
    Test different phase precisions to determine requirements for success
    """
    results = []
    register_sizes = [2**i for i in range(1, int(math.log2(max_register_size))+1)]
    precisions = [10.0, 5.0, 1.0, 0.5, 0.1, 0.05, 0.01]
    
    for size in register_sizes:
        for precision in precisions:
            successes = 0
            optimal_iterations = math.floor(math.pi/4 * math.sqrt(size))
            
            for _ in range(trials_per_config):
                target = random.randint(0, size-1)
                result = runStereoSearch(size, target, optimal_iterations, 
                                        phase_precision=precision, verbose=False)
                if result['success']:
                    successes += 1
                    
            results.append({
                'register_size': size,
                'precision': precision,
                'success_rate': successes / trials_per_config
            })
            
    return results

def analyzeNoiseResilience(register_size=8, trials_per_config=10):
    """
    Test how different noise levels and error correction methods affect success
    """
    results = []
    noise_levels = [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    correction_methods = [None, 'phase_tracking', 'redundancy']
    
    optimal_iterations = math.floor(math.pi/4 * math.sqrt(register_size))
    
    for noise in noise_levels:
        for correction in correction_methods:
            successes = 0
            
            for _ in range(trials_per_config):
                target = random.randint(0, register_size-1)
                result = runStereoSearch(register_size, target, optimal_iterations,
                                        phase_precision=0.1, noise_level=noise,
                                        error_correction=correction, verbose=False)
                if result['success']:
                    successes += 1
                    
            results.append({
                'noise_level': noise,
                'error_correction': correction,
                'success_rate': successes / trials_per_config
            })
            
    return results

def plotPrecisionRequirements(results):
    """
    Plot the success rate vs precision for different register sizes
    """
    plt.figure(figsize=(10, 6))
    register_sizes = sorted(list(set([r['register_size'] for r in results])))
    
    for size in register_sizes:
        size_results = [r for r in results if r['register_size'] == size]
        precisions = [r['precision'] for r in size_results]
        success_rates = [r['success_rate'] for r in size_results]
        
        plt.plot(precisions, success_rates, 'o-', label=f'Size={size}')
    
    plt.xscale('log')
    plt.title('Phase Precision Requirements for Stereo Search')
    plt.xlabel('Phase Precision (degrees, lower is better)')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.legend()
    return plt
    
def plotNoiseResilience(results):
    """
    Plot the success rate vs noise level for different error correction methods
    """
    plt.figure(figsize=(10, 6))
    correction_methods = sorted(list(set([r['error_correction'] for r in results])))
    
    for method in correction_methods:
        method_name = 'None' if method is None else method
        method_results = [r for r in results if r['error_correction'] == method]
        noise_levels = [r['noise_level'] for r in method_results]
        success_rates = [r['success_rate'] for r in method_results]
        
        plt.plot(noise_levels, success_rates, 'o-', label=f'Correction={method_name}')
    
    plt.title('Noise Resilience with Different Error Correction Methods')
    plt.xlabel('Noise Level (degrees)')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.legend()
    return plt

# Example usage with visualization
if __name__ == "__main__":
    # Basic demonstration
    result = runStereoSearch(4, 2, 2, phase_precision=0.1)
    result['register'].plotPhaseHistory()
    
    # Phase precision analysis
    precision_results = analyzePhaseRequirements(max_register_size=8)
    plotPrecisionRequirements(precision_results)
    
    # Noise resilience analysis
    noise_results = analyzeNoiseResilience(register_size=4)
    plotNoiseResilience(noise_results)
