import subprocess

from setuptools import setup
from setuptools import find_packages


setup(
    name='dyda',
    version='1.40.2+20190911',
    description='DT42 dyda library and application',
    long_description=
        'TBD',
    url='https://gitlab.com/DT42/galaxy42/dt42-dyda',
    author='DT42 research team',
    author_email='research@dt42.io',
    license='DT42 License',
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords=['wheels'],
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'dt42lab',
        'matplotlib',
        'numpy',
        'Pillow',
        'requests'
    ],
    extras_require={
        'tf': ['tensorflow==1.1.0'],
        'tf_gpu': ['tensorflow-gpu==1.1.0'],
        'opencv': ['opencv-python']
    },
    python_requires='>=3',
    test_suite='tests'
)
