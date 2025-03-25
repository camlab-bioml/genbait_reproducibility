import platform
import sys
from setuptools import setup, find_packages

# Platform-specific warnings for compiler requirements
if platform.system() == "Windows":
    print(
        "\n[GENBAIT REPRODUCIBILITY INSTALL WARNING - Windows]\n"
        "Some dependencies (e.g., shap, xgboost) require Microsoft C++ Build Tools.\n"
        "To install them:\n"
        "1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
        "2. In the installer, select the **C++ build tools** workload.\n"
        "3. Ensure the following components are selected:\n"
        "   - MSVC v14+ (e.g., v142 or v143)\n"
        "   - Windows 10 or 11 SDK\n"
        "   - (Optional) C++ CMake tools\n"
        "4. Install and restart your terminal.\n",
        file=sys.stderr
    )
elif platform.system() == "Darwin":
    print(
        "\n[GENBAIT REPRODUCIBILITY INSTALL NOTE - macOS]\n"
        "If installation fails due to missing compiler tools or Python.h not found:\n"
        "Run the following command to install Xcode Command Line Tools:\n"
        "👉 xcode-select --install\n",
        file=sys.stderr
    )

# Package setup
setup(
    name="genbait_reproducibility",
    version="0.1.0",
    author="Your Name",
    author_email="vesal.kasmaeifar@mail.utoronto.ca",
    description="GENBAIT reproducibility: A bioinformatics tool for bait selection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vesalkasmaeifar/genbait_reproducibility",
    packages=find_packages(include=["src", "src.*"]),
    package_data={
        "": ["config/*.yaml", "data/**/*.txt", "data/**/*.csv", "Snakefile"],
    },
    include_package_data=True,
    install_requires=[
        "snakemake",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "argparse",
        "pyyaml",
        "gprofiler",
        "deap",
        "torch",
        "pytorch_lightning",
        "shap",
        "XGBoost",
    ],
    entry_points={
        "console_scripts": [
            "genbait=src.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
