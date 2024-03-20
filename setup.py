import io
import os
from setuptools import find_packages, setup


setuptools.setup(
    name="dlshm",
    version="0.1.0",
    description="dlshm",
    url="https://github.com/PTauzowski/DLSHM",
    author="Piotr Tauzowski",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=[]),
    classifiers=[
        "Development Status :: 1 - Planning",

        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",

        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # "tensorflow>=1.2.0",
        # "pyyaml",
    ],
    python_requires='>=3.8'
)

