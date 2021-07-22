#!/usr/bin/env python3
import sys

from setuptools import setup

import zimply


def read_file(fname):
    """
    Read file and decode in py2k
    """
    if sys.version_info < (3,):
        return open(fname).read().decode("utf-8")
    return open(fname).read()


dist_name = "zimply_core"
plugin_name = "zimply_core"
repo_url = "https://github.com/endlessm/kolibri-zim-plugin"

readme = read_file("README.md")

description = """Fork of ZIMply with only the zimply_core module."""


setup(
        name='zimply-core',
        version=zimply.__version__,
        description=description,
        long_description=readme,
        long_description_content_type="text/markdown",
        packages=['zimply_core'],
        package_dir={'zimply_core': 'zimply'},
        author="Dylan McCall",
        author_email="dylan@endlessos.org",
        license='MIT',
        url="https://github.com/dylanmccall/ZIMply-core",
        keywords=['zim', 'wiki', 'wikipedia'],
        install_requires=["zstandard>=0.14.1"],
        classifiers=[
            'Programming Language :: Python :: 3.4',
            'License :: OSI Approved :: MIT License',
            "Development Status :: 2 - Pre-Alpha",
        ],
        platforms='any',
)
