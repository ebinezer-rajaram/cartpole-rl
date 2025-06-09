import numpy as np
import os

def main():
    np.random.seed(42)
    
    noise_std = 0.05  

    os.makedirs("data/task_4.1", exist_ok=True)
    

    print("\n=== Processing task 1.3 data (linear model) ===")
    try:
        X_linear = np.load("data/task_1.3/X.npy")
        Y_linear = np.load("data/task_1.3/Y.npy")
        
        Y_linear_noisy = Y_linear + np.random.normal(0, noise_std, size=Y_linear.shape)
        
        np.savez("data/task_4.1/noisy_dataset.npz", X=X_linear, Y=Y_linear_noisy)
        
        print(f"Linear noisy dataset saved to data/task_4.1/noisy_dataset.npz")
        print(f"  - Shapes: X={X_linear.shape}, Y={Y_linear_noisy.shape}")
        print(f"  - Added noise with std={noise_std}")
    except FileNotFoundError:
        print("Error: Could not find task 1.3 data files (X.npy and Y.npy)")
    
    print("\n=== Processing task 3.1 data (nonlinear model) ===")
    try:
        X_nonlinear = np.load("data/task_3.1/X.npy")
        Y_nonlinear = np.load("data/task_3.1/Y.npy")
        
        Y_nonlinear_noisy = Y_nonlinear + np.random.normal(0, noise_std, size=Y_nonlinear.shape)
        
        np.savez("data/task_4.1/noisy_nonlinear_dataset.npz", X=X_nonlinear, Y=Y_nonlinear_noisy)
        
        print(f"Nonlinear noisy dataset saved to data/task_4.1/noisy_nonlinear_dataset.npz")
        print(f"  - Shapes: X={X_nonlinear.shape}, Y={Y_nonlinear_noisy.shape}")
        print(f"  - Added noise with std={noise_std}")
    except FileNotFoundError:
        print("Error: Could not find task 3.1 dataset file")
    
if __name__ == "__main__":
    main()