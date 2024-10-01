from setuptools import setup, find_packages

setup(
    name="genbait_reproducibility",
    version="0.1.0",
    author="Your Name",
    author_email="vesal.kasmaeifar@mail.utoronto.ca",
    description="GENBAIT: A bioinformatics tool for bait selection",
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
        "deap",
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
    python_requires=">=3.6",
)
