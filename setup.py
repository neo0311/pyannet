from setuptools import setup, find_packages

setup(
    name="PYANNET",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["numpy", "matplotlib"],
    extras_require={"dev": ["pytest"]},
)
