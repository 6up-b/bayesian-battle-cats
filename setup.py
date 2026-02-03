from setuptools import setup
from Cython.Build import cythonize

setup(
    name="seed_seeker",
    ext_modules=cythonize(
        "seed_seeker.pyx",
        compiler_directives={"language_level": "3"},
    ),
)

