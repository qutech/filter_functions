# -*- coding: utf-8 -*-
import os
import re
import sys

from setuptools import setup


def read(*args):
    return open(os.path.join(os.path.dirname(__file__), *args),
                encoding='utf8').read()


def extract_version(version_file):
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)

    raise RuntimeError("Unable to find version string.")


if sys.version_info < (3, 5):
    sys.stderr.write('ERROR: You need Python 3.5 or later '
                     'to install this package.\n')
    exit(1)

setup(name='filter_functions',
      version=extract_version(read('filter_functions', '__init__.py')),
      description='Package for efficient calculation of generalized filter functions',
      long_description=read('README.md'),
      long_description_content_type='text/markdown',
      url='https://github.com/qutech/filter_functions',
      author='Quantum Technology Group, RWTH Aachen University',
      author_email='tobias.hangleiter@rwth-aachen.de',
      packages=['filter_functions'],
      package_dir={'filter_functions': 'filter_functions'},
      install_requires=['numpy', 'scipy', 'matplotlib', 'qutip', 'opt_einsum',
                        'sparse'],
      extras_require={
          'fancy_progressbar': ['tqdm', 'requests'],
          'doc': ['ipython', 'ipykernel', 'nbsphinx', 'numpydoc', 'sphinx',
                  'jupyter_client', 'sphinx_rtd_theme'],
          'tests': ['pytest', 'coverage', 'coveralls'],
      },
      test_suite='tests',
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering :: Physics',
      ])
