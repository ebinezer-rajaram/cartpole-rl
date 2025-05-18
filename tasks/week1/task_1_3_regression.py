import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    X = np.load("data/task_1.3/X.npy")  # shape (N, d)
    Y = np.load("data/task_1.3/Y.npy")  # shape (N, 4)
    d = X.shape[1]
    labels = ["x", "x_dot", "theta", "theta_dot"]

    C = np.linalg.lstsq(X, Y, rcond=None)[0].T  # shape (4, d)
    print("Fitted C:\n", C)

    Y_pred = X @ C.T

    base_dir = "figures/task_1.3/regression"
    pdf_dir = os.path.join(base_dir, "pdf")
    png_dir = os.path.join(base_dir, "png")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    for j in range(4): 
        # --- Predicted vs True scatter ---
        plt.figure()
        plt.scatter(Y[:, j], Y_pred[:, j], alpha=0.6)
        plt.plot([Y[:, j].min(), Y[:, j].max()],
                 [Y[:, j].min(), Y[:, j].max()], 'k--')
        var = labels[j]
        plt.xlabel(f"True Δ{var}")
        plt.ylabel(f"Predicted Δ{var}")
        plt.title(f"Δ{var}: predicted vs true")
        plt.grid(True)

        fname = f"predicted_vs_true_delta_{var}"
        plt.savefig(f"{pdf_dir}/{fname}.pdf")
        plt.savefig(f"{png_dir}/{fname}.png")
        plt.close()

        # --- Input X_i vs Δ_j line plot ---
        for i in range(d):
            input_label = labels[i] if i < 4 else f"action"
            output_label = labels[j]

            plt.figure()
            plt.plot(X[:, i], Y[:, j], 'o', label="true", alpha=0.5)
            plt.plot(X[:, i], Y_pred[:, j], '.', label="pred", alpha=0.5)
            plt.xlabel(f"{input_label}")
            plt.ylabel(f"Δ{output_label}")
            plt.title(f"Δ{output_label} vs {input_label} (true vs pred)")
            plt.legend()
            plt.grid(True)

            fname = f"delta_{output_label}_vs_{input_label}_true_vs_pred"
            plt.savefig(f"{pdf_dir}/{fname}.pdf")
            plt.savefig(f"{png_dir}/{fname}.png")
            plt.close()


if __name__ == "__main__":
    main()
