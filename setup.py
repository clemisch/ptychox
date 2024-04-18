from setuptools import setup

setup(
    name="ptychox",
    version="1.0",
    description="Ptychographic reconstruction powered by JAX",
    author="Clemens Schmid",
    author_email="clemens.schmid@psi.ch",
    packages=["ptychox"],
    install_requires=["numpy", "scipy", "jax", "numexpr"],
    license="MIT",
)
