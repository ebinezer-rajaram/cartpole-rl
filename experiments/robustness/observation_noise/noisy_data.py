import numpy as np
import os

def main():
    np.random.seed(42)
    
    noise_std = 0.05

    os.makedirs("data/robustness/observation_noise", exist_ok=True)
    

    print("\n=== Processing state-transition data (linear model) ===")
    try:
        X_linear = np.load("data/state_transitions/X.npy")
        Y_linear = np.load("data/state_transitions/Y.npy")
        
        Y_linear_noisy = Y_linear + np.random.normal(0, noise_std, size=Y_linear.shape)
        
        np.savez("data/robustness/observation_noise/noisy_dataset.npz", X=X_linear, Y=Y_linear_noisy)
        
        print(f"Linear noisy dataset saved to data/robustness/observation_noise/noisy_dataset.npz")
        print(f"  - Shapes: X={X_linear.shape}, Y={Y_linear_noisy.shape}")
        print(f"  - Added noise with std={noise_std}")
    except FileNotFoundError:
        print("Error: Could not find state-transition data files (X.npy and Y.npy)")
    
    print("\n=== Processing state-action data (nonlinear model) ===")
    try:
        X_nonlinear = np.load("data/state_action_transitions/X.npy")
        Y_nonlinear = np.load("data/state_action_transitions/Y.npy")
        
        Y_nonlinear_noisy = Y_nonlinear + np.random.normal(0, noise_std, size=Y_nonlinear.shape)
        
        np.savez("data/robustness/observation_noise/noisy_nonlinear_dataset.npz", X=X_nonlinear, Y=Y_nonlinear_noisy)
        
        print(f"Nonlinear noisy dataset saved to data/robustness/observation_noise/noisy_nonlinear_dataset.npz")
        print(f"  - Shapes: X={X_nonlinear.shape}, Y={Y_nonlinear_noisy.shape}")
        print(f"  - Added noise with std={noise_std}")
    except FileNotFoundError:
        print("Error: Could not find state-action dataset file")
    
if __name__ == "__main__":
    main()