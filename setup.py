import setuptools

with open("README.md", "r") as file_handle:
    long_description = file_handle.read()

setuptools.setup(
        name='mpstool',
        version='0.0.1dev',
        author="Przemyslaw Juda",
        description="Multiple-point statistics toolbox",
        long_description=long_description,
        packages=setuptools.find_packages()
        )
