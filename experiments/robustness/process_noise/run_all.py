import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Process-noise pipeline: noisy dynamics, data, model and policy stability")
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4], default=0, 
                        help="Run specific step: 1=test dynamics, 2=generate data, 3=train model, 4=evaluate stability")
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs("figures/robustness/process_noise", exist_ok=True)
    os.makedirs("models/robustness/process_noise", exist_ok=True)
    os.makedirs("data/robustness/process_noise", exist_ok=True)
    
    # Determine which steps to run
    run_all = args.step == 0
    
    if run_all or args.step == 1:
        print("\n=== Step 1: Test Noisy Dynamics ===")
        # Import and run noisy dynamics test
        from experiments.robustness.process_noise.noisy_dynamics import test_noisy_dynamics
        test_noisy_dynamics()
    
    if run_all or args.step == 2:
        print("\n=== Step 2: Generate Data with Noisy Dynamics ===")
        # Import and run data generation
        from experiments.robustness.process_noise.noisy_data import main as generate_data
        generate_data()
    
    if run_all or args.step == 3:
        print("\n=== Step 3: Train Nonlinear Model ===")
        # Import and run nonlinear model training
        from experiments.robustness.process_noise.kernel_model import main as train_nonlinear
        train_nonlinear()
    
    if run_all or args.step == 4:
        print("\n=== Step 4: Policy Stability Analysis ===")
        # Import and run stability analysis
        from experiments.robustness.process_noise.policy_stability import main as run_stability_analysis
        run_stability_analysis()
    
    print("\nProcess-noise pipeline complete.")

if __name__ == "__main__":
    main()
