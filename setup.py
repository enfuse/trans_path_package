from setuptools import setup, find_packages

# Package metadata
NAME = 'trans_path_enfuse'
VERSION = '1.0.5'
DESCRIPTION = 'Enfuse implementation of TransPath model from AIRI'
AUTHOR = 'Enfuse.io'
EMAIL = 'jd@enfuse.io'
URL = 'https://github.com/enfuse/trans_path_package'
LICENSE = 'MIT'
PYTHON_VERSION = '>=3.11.0'

# Long description from README.md
with open('README.md', 'r', encoding = 'utf-8') as f:
    long_description = 'you do not need the whole README here'

# Required packages
with open('requirements.txt', 'r') as f:
    required_packages = f.read().splitlines()

setup(
    name = NAME,
    version = VERSION,
    description = DESCRIPTION,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author = AUTHOR,
    author_email = EMAIL,
    url = URL,
    license = LICENSE,
    python_requires = PYTHON_VERSION,
    packages = find_packages(),
    package_data = {
        'trans_path': ['models/weights/*.pth']
    },
    install_requires = required_packages,
    classifiers = [
        'Development Status :: 4 - Production/Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent'
    ],
)