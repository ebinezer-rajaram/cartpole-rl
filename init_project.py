import os

BASE_DIR = "cartpole-rl"

STRUCTURE = [
    "cartpole",
    "tasks",
    "tests",
    "scripts",
    "data/rollouts",
    "figures/pdf",
    "figures/png",
    "animations",
    "reports",
    "notebooks",
]

FILES = {
    ".gitignore": """
__pycache__/
*.pyc
.ipynb_checkpoints/
*.npy
*.mp4
*.pdf
*.png
.DS_Store
""",
    "requirements.txt": """
numpy
matplotlib
pytest
""",
    "README.md": "# CartPole RL Project\n",
    "main.py": "# Optional script entry point\n",
    "reports/interim_report.tex": "% LaTeX file for interim report\n",
    "notebooks/dev_playground.ipynb": "",  # leave empty for now
    "tasks/__init__.py": "",
    "tests/test_cartpole.py": "# Placeholder for CartPole unit tests\n",
    "tasks/task_1_rollout.py": "# Task 1.1 rollout simulation\n",
    "tasks/task_1_plotting.py": "# Task 1.1 plotting functions\n",
    "scripts/run_rollout.py": "# CLI script to generate rollouts\n",
}

def create_structure():
    for path in STRUCTURE:
        full_path = os.path.join(BASE_DIR, path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")

    for file_path, content in FILES.items():
        full_path = os.path.join(BASE_DIR, file_path)
        with open(full_path, "w") as f:
            f.write(content.strip() + "\n")
        print(f"Created file: {full_path}")

if __name__ == "__main__":
    create_structure()
