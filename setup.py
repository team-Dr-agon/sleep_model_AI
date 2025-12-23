# setup.py
from setuptools import setup, Extension
import sys
import pybind11

if sys.platform == "win32":
    extra_compile_args = ["/O2", "/std:c++17"]
else:
    extra_compile_args = ["-O3", "-std=c++17"]

ext_modules = [
    Extension(
        "kalman_filter",                 # import kalman_filter になる
        ["kalman_filter.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="kalman_filter",
    version="0.0.1",
    ext_modules=ext_modules,
)
