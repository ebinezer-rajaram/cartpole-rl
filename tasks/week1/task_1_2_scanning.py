import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations
from matplotlib.tri import Triangulation

from cartpole.scanning import scan_1d, scan_2d

def main():
    labels = ["x", "x_dot", "theta", "theta_dot"]

    # TOGGLE: True for Y = X' - X; False for Y = X'
    return_delta = False

    scan_ranges = {
        0: np.linspace(-5, 5, 100),
        1: np.linspace(-10, 10, 100),
        2: np.linspace(-np.pi, np.pi, 100),
        3: np.linspace(-15, 15, 100),
    }

    base_state = np.random.uniform(
        low=[-5, -10, -np.pi, -15],
        high=[5, 10, np.pi, 15],
        size=4
    )

    tag = "delta" if return_delta else "next"
    base_dir = f"figures/task_1.2/{tag}"
    pdf_dir = os.path.join(base_dir, "pdf")
    png_dir = os.path.join(base_dir, "png")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    # --- 1D scans ---
    # for i in range(4):
    #     X_vals, Y_vals = scan_1d(i, scan_ranges[i], base_state, return_delta)
    #     for j in range(4):
    #         plt.figure()
    #         plt.plot(X_vals, Y_vals[:, j])
    #         plt.xlabel(labels[i])
    #         ylabel = f"Δ{labels[j]}" if return_delta else f"{labels[j]} (next)"
    #         plt.ylabel(ylabel)
    #         plt.title(f"{ylabel} vs {labels[i]}")
    #         plt.grid(True)

    #         fname = f"{ylabel.replace(' ', '_')}_vs_{labels[i]}"
    #         plt.savefig(os.path.join(pdf_dir, f"{fname}.pdf"))
    #         plt.savefig(os.path.join(png_dir, f"{fname}.png"))
    #         plt.show()
    #         plt.close()

    # Combined 1D plots 
    for i in range(4):
        X_vals, Y_vals = scan_1d(i, scan_ranges[i], base_state, return_delta)
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axs = axs.flatten()

        for j in range(4):
            axs[j].plot(X_vals, Y_vals[:, j])
            axs[j].set_xlabel(labels[i])
            ylabel = f"Δ{labels[j]}" if return_delta else f"{labels[j]} (next)"
            axs[j].set_ylabel(ylabel)
            axs[j].set_title(f"{ylabel} vs {labels[i]}")
            axs[j].grid(True)

        fig.suptitle(f"All Outputs vs {labels[i]}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        fname = f"combined_outputs_vs_{labels[i]}"
        fig.savefig(os.path.join(pdf_dir, f"{fname}.pdf"))
        fig.savefig(os.path.join(png_dir, f"{fname}.png"))
        plt.show()
        plt.close(fig)

    # 2D slices (only for delta mode) 
    if return_delta:
        print("Generating 2D contour plots...")

        contour_base_dir = f"figures/task_1.2/{tag}_contour"
        contour_pdf = os.path.join(contour_base_dir, "pdf")
        contour_png = os.path.join(contour_base_dir, "png")
        os.makedirs(contour_pdf, exist_ok=True)
        os.makedirs(contour_png, exist_ok=True)

        pairs = list(combinations(range(4), 2))

        for i, j in pairs:
            X_coords, Z_outputs = scan_2d(
                i, j,
                scan_ranges[i][[0, -1]], scan_ranges[j][[0, -1]],
                base_state, return_delta,
                grid_resolution=30
            )

            for k in range(4):
                Z = Z_outputs[:, k]
                plt.figure()
                triang = Triangulation(X_coords[:, 0], X_coords[:, 1])
                contour = plt.tricontourf(triang, Z, levels=20, cmap='viridis')
                plt.colorbar(contour)

                label_i = labels[i]
                label_j = labels[j]
                label_k = f"Δ{labels[k]}"
                plt.xlabel(label_i)
                plt.ylabel(label_j)
                plt.title(f"{label_k} over ({label_i}, {label_j})")

                fname = f"{label_k.replace(' ', '_')}_vs_{label_i}_{label_j}"
                plt.savefig(os.path.join(contour_pdf, f"{fname}.pdf"))
                plt.savefig(os.path.join(contour_png, f"{fname}.png"))
                plt.show()
                plt.close()


if __name__ == "__main__":
    main()
