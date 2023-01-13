import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="describer_ml",
    version="0.22",
    description="A set of descriptive statistics and hypothesis tests",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/EricSchles/describer_ml",
    author="Eric Schles",
    author_email="ericschles@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["describer_ml", "describer_ml.timeseries", "describer_ml.numeric", "describer_ml.matching"],
    include_package_data=True,
    install_requires=["sklearn", "scipy", "numpy", "statsmodels", "pytest", "mlxtend", "ThinkBayes2"],
)
