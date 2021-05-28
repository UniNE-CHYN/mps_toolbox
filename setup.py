"""Setup script for mpstool"""

import setuptools

with open("README.md", "r") as file_handle:
    long_description = file_handle.read()

setuptools.setup(
    name='mpstool',
    version='1.1.0',
    author="randlab",
    description="Multiple-point statistics toolbox",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/UniNE-CHYN/mps_toolbox",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    packages=["mpstool"],
    install_requires=[
        'numpy',
        'scikit-image',
        'py-vox-io',
        'properscoring',
        'Pillow',
    ]
)
