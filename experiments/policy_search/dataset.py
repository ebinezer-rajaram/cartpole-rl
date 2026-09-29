import os
import numpy as np
from cartpole.data import collect_dataset

def main():
    np.random.seed(5)
    X, Y = collect_dataset(n_samples=50000, with_action=True)
    os.makedirs("data/state_action_transitions", exist_ok=True)
    np.save("data/state_action_transitions/X.npy", X)
    np.save("data/state_action_transitions/Y.npy", Y)
    print("Saved dataset to data/state_action_transitions/")

if __name__ == "__main__":
    main()
