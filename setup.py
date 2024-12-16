from setuptools import setup, find_packages

setup(
    name='prm2Gaussian',
    version='0.2',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'prm2Gaussian=prm2Gaussian:main',
        ],
    },
    author='tomaszborowski',
    author_email='tomasz.borowski@ikifp.edu.pl',
    description='prm2Gaussian: A tool for converting amber topology file to Gaussian-Oniom input files',
    url='https://github.com/tomaszborowski/prm2gaussian',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)