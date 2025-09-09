from setuptools import setup, find_packages

setup(
    name="oscnet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pytest>=7.0.0",
        "click>=8.0.0",
        "equinox>=0.1.0",
        "jax>=0.3.25",
        "jaxlib>=0.3.25",
        "diffrax>=0.1.0",
        "optax>=0.1.0",
        "numpy>=1.24.0",
        "matplotlib>=3.5.0",
        "networkx>=2.7.0",
    ],
    python_requires=">=3.8",
    author="OscNet Team",
    author_email="example@example.com",
    description="Tools for oscillatory neural networks and nonlinear dynamics",
    keywords="oscillator, dynamics, neural networks, JAX",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ],
) 