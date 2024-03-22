from setuptools import setup, find_packages

# Package metadata
NAME = 'my_project'
VERSION = '1.0.0'
DESCRIPTION = 'Description of your project'
AUTHOR = 'Your Name'
EMAIL = 'your@email.com'
URL = 'https://github.com/yourusername/my_project'
LICENSE = 'MIT'
PYTHON_VERSION = '>=3.6'

# Long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Required packages
with open('requirements.txt', 'r') as f:
    required_packages = f.read().splitlines()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    license=LICENSE,
    python_requires=PYTHON_VERSION,
    packages=find_packages(),
    install_requires=required_packages,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)