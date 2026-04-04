import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PIPELINE_SCRIPTS = [
    "make_h5.py",
    "preprocess.py",
    "visualize.py",
    "Feature_extraction.py",
    "Training.py",
]


def implement_pipeline():
    for script in PIPELINE_SCRIPTS:
        path = ROOT / script
        print(f"Running {script}\n")
        subprocess.run([sys.executable, str(path)], cwd=ROOT, check=False)
    print("Pipeline completed.")


if __name__ == "__main__":
    implement_pipeline()
