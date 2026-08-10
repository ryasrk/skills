from setuptools import setup, find_packages

setup(
    name="constraint-driven-ml-skills",
    version="0.1.0",
    description="Modular skill library for rigorous ML research under deployment constraints",
    author="Research methodology from face-detection-openvino-edge",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "openvino>=2024.1",
        "matplotlib",
        "numpy",
        "psutil",
        "opencv-python"
    ]
)
