import os
import numpy as np
from cartpole.data import collect_dataset

def main():
    X, Y = collect_dataset(n_samples=500)
    os.makedirs("data/state_transitions", exist_ok=True)
    np.save("data/state_transitions/X.npy", X)
    np.save("data/state_transitions/Y.npy", Y)
    print("Saved dataset to data/state_transitions/")

if __name__ == "__main__":
    main()
