"""Package setup."""

import os
import re
import sys

from typing import Optional

from setuptools import find_packages, setup


PACKAGE_NAME: str = "deker_tools"


def get_version() -> str:
    """Get version from commit tag.

    Regexp reference:
    https://gitlab.openweathermap.org/help/user/packages/pypi_repository/index.md#ensure-your-version-string-is-valid
    """
    ci_commit_tag: Optional[str] = os.getenv("PACKAGE_VERSION", "1.0.0b-2")
    regex = (
        r"(?:"
        r"(?:([0-9]+)!)?"
        r"([0-9]+(?:\.[0-9]+)*)"
        r"([-_\.]?((a|b|c|rc|alpha|beta|pre|preview))[-_\.]?([0-9]+)?)?"
        r"((?:-([0-9]+))|(?:[-_\.]?(post|rev|r)[-_\.]?([0-9]+)?))?"
        r"([-_\.]?(dev)[-_\.]?([0-9]+)?)?"
        r"(?:\+([a-z0-9]+(?:[-_\.][a-z0-9]+)*))?"
        r")$"
    )
    try:
        return re.search(regex, ci_commit_tag, re.X + re.IGNORECASE).group()
    except Exception:
        sys.exit(f"No valid version could be found in PACKAGE_VERSION {ci_commit_tag}")


with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip("\n") for line in f if line.strip("\n") and not line.startswith(("#", "-i", "abstract"))]


setup_kwargs = dict(
    name=PACKAGE_NAME,
    version=get_version(),
    author="OpenWeatherMap",
    description="Tools for Deker management",
    packages=find_packages(exclude=["tests", "test*.*"]),
    author_email="info@openweathermap.org",
    maintainer_email="info@openweathermap.org",
    url="https://github.com/openweathermap/deker-tools",
    license="GPL-3.0-only",
    package_data={PACKAGE_NAME: ["py.typed"]},
    include_package_data=True,
    platforms="any",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: Unix",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)

setup(**setup_kwargs)
