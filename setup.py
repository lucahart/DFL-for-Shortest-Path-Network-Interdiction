# setup.py
#     author="Luca Hartmann",
#     author_email="lhartmann@ucsd.edu"

from setuptools import setup, find_packages

setup(
    name="shortest_path_indiction",         # project name
    version="0.1.0",
    package_dir={"": "src"},                # tells setuptools that packages live under src/
    packages=find_packages(where="src"),    # auto-discovers all packages under src/
    py_modules=["Main"],                     # include top-level modules
    install_requires=[
        "torch>=1.10.0",
    ],
    entry_points={
        "console_scripts": [
            "shortest-path=Main:main",
        ]
    },
)
