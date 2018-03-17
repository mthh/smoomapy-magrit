#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from ast import parse
from os import path
try:
    from future_builtins import filter
except ImportError:
    pass

with open("requirements.txt") as f:
    requirements = f.read().split('\n')

with open(path.join('smoomacypy', '__init__.py')) as f:
    __version__ = parse(next(filter(lambda line: line.startswith('__version__'),
                                     f))).body[0].value.s

exts = [Extension("smoomacypy.compute",
                  ["smoomacypy/compute.pyx"], ["."],
                  extra_compile_args=["-march=native", "-Ofast", "-fno-signed-zeros"])]

setup(
    name='smoomacypy',
    version=__version__,
    author="Matthieu Viry",
    author_email="matthieu.viry@cnrs.fr",
    packages=find_packages(),
    ext_modules=exts,
    description="Brings smoothed maps through python",
    url='http://github.com/mthh/smoomacypy',
    license="MIT",
    test_suite="tests",
    install_requires=requirements,
    setup_requires=['setuptools>=25.1', 'Cython>=0.24'],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
        ],
    )
