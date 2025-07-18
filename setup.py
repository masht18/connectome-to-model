#!/usr/bin/env python
"""Setup script for connectome-to-model package."""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements from requirements.txt if it exists
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'numpy>=1.19.0',
        'pandas>=1.3.0',
        'matplotlib>=3.3.0',
        'scipy>=1.7.0',
        'tqdm>=4.60.0',
    ]

setup(
    name='connectome-to-model',
    version='0.1.0',
    author='Mashbayar Tugsbayar',
    author_email='',  # Add email if available
    description='Transform biological connectome data into functional artificial neural networks with biologically-inspired top-down feedback mechanisms',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/masht18/convgru_feedback',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    python_requires='>=3.8',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.10',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.812',
        ],
        'notebooks': [
            'jupyter>=1.0.0',
            'ipykernel>=6.0.0',
        ],
        'audio': [
            'torchaudio>=0.9.0',
        ],
    },
    include_package_data=True,
    package_data={
        'connectome_to_model': [
            'graphs/*.csv',
            'examples/*.ipynb',
        ],
    },
    entry_points={
        'console_scripts': [
            # Add any command-line scripts here if needed
        ],
    },
    keywords='connectome, neural networks, neuroscience, machine learning, pytorch, top-down feedback, brain-inspired AI',
    project_urls={
        'Bug Reports': 'https://github.com/masht18/convgru_feedback/issues',
        'Source': 'https://github.com/masht18/convgru_feedback',
        'Documentation': 'https://github.com/masht18/convgru_feedback#readme',
    },
)