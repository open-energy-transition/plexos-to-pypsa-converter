from setuptools import find_packages, setup

setup(
    name="plexos-pypsa",
    version="0.1.0",
    description="A Python package for interfacing PLEXOS with PyPSA.",
    author="Firstname Lastname",
    author_email="email@example.com",
    packages=find_packages(),
    install_requires=[
        # Add dependencies here, e.g., "numpy", "pandas"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
