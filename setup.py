from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="lr-model-cpp",
    version="0.1.0",
    ext_modules=[
        Pybind11Extension(
            "LR_Model_cpp", 
            ["LR_Model.cpp", "LR_Model_bindings.cpp"], 
            cxx_std=17
        ),
    ],
    cmdclass={"build_ext": build_ext},
)
