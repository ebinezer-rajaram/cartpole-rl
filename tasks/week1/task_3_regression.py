import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    X = np.load("data/task_1.3/X.npy")  # shape (N, d)
    Y = np.load("data/task_1.3/Y.npy")  # shape (N, 4)
    d = X.shape[1]
    labels = ["x", "x_dot", "theta", "theta_dot"]

    # --- Fit C via least squares: Y ≈ C X
    C = np.linalg.lstsq(X, Y, rcond=None)[0].T  # shape (4, d)
    print("Fitted C:\n", C)

    Y_pred = X @ C.T

    # --- Save figures
    pdf_dir = "figures/task_1.3/pdf"
    png_dir = "figures/task_1.3/png"
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    for j in range(4):  # output dimensions
        # --- Y_true vs Y_pred
        plt.figure()
        plt.scatter(Y[:, j], Y_pred[:, j], alpha=0.6)
        plt.plot([Y[:, j].min(), Y[:, j].max()],
                 [Y[:, j].min(), Y[:, j].max()], 'k--')
        plt.xlabel(f"True Δ{labels[j]}")
        plt.ylabel(f"Predicted Δ{labels[j]}")
        plt.title(f"Δ{labels[j]}: predicted vs true")
        plt.grid(True)
        fname = f"pred_vs_true_delta_{labels[j]}"
        plt.savefig(f"{pdf_dir}/{fname}.pdf")
        plt.savefig(f"{png_dir}/{fname}.png")
        plt.close()

        # --- X_i vs Y_j overlays
        for i in range(d):
            plt.figure()
            plt.plot(X[:, i], Y[:, j], 'o', label="true", alpha=0.5)
            plt.plot(X[:, i], Y_pred[:, j], '.', label="pred", alpha=0.5)
            plt.xlabel(f"X[{i}]")
            plt.ylabel(f"Δ{labels[j]}")
            plt.title(f"Δ{labels[j]} vs X[{i}] (true vs pred)")
            plt.legend()
            plt.grid(True)
            fname = f"delta_{labels[j]}_vs_X{i}"
            plt.savefig(f"{pdf_dir}/{fname}.pdf")
            plt.savefig(f"{png_dir}/{fname}.png")
            plt.close()

if __name__ == "__main__":
    main()
