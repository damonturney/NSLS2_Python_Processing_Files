#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:18:33 2020

@author: damon
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("image_cross_correlation.pyx")
)