#!/usr/bin/env python
# setup.py

import setuptools

setuptools.setup(
    name='infoseq',
    version='0.0.1',
    description='tools for information theory on sequences',
    author='A. Ray',
    packages=['infoseq'],
    requires=[
        'wheel',
        'numpy',
        'matplotlib',
        'jupyter',
        'pandas',
        'ipython',
        'ipdb',
        'mypy',
    ],
)
