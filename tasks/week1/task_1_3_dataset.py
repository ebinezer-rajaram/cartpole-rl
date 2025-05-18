import os
import numpy as np
from cartpole.data import collect_dataset

def main():
    X, Y = collect_dataset(n_samples=500)
    os.makedirs("data/task_1.3", exist_ok=True)
    np.save("data/task_1.3/X.npy", X)
    np.save("data/task_1.3/Y.npy", Y)
    print("Saved dataset to data/task_1.3/")

if __name__ == "__main__":
    main()
