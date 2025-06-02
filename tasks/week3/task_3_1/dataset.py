import os
import numpy as np
from cartpole.data import collect_dataset

def main():
    X, Y = collect_dataset(n_samples=1000, with_action=True)
    os.makedirs("data/task_3.1", exist_ok=True)
    np.save("data/task_3.1/X.npy", X)
    np.save("data/task_3.1/Y.npy", Y)
    print("Saved dataset to data/task_3.1/")

if __name__ == "__main__":
    main()
