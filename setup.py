from setuptools import setup, find_packages
import sys
import os

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import allennlp_semparse whilst setting up.
VERSION = {}
with open("allennlp_models/version.py") as version_file:
    exec(version_file.read(), VERSION)

# Load requirements.txt with a special case for allennlp so we can handle
# cross-library integration testing.
with open("requirements.txt") as requirements_file:
    import re

    def requirement_is_allennlp(req: str) -> bool:
        if req == "allennlp":
            return True
        if re.match(r"^allennlp[>=<]", req):
            return True
        if re.match(r"^(git\+)?(https|ssh)://(git@)?github\.com/allenai/allennlp\.git@", req):
            return True
        return False

    def fix_url_dependencies(req: str) -> str:
        """Pip and setuptools disagree about how URL dependencies should be handled."""
        m = re.match(r"^(git\+)?(https|ssh)://(git@)?github\.com/([\w-]+)/(?P<name>[\w-]+)\.git", req)
        if m is None:
            return req
        else:
            return f"{m.group('name')} @ {req}"

    override_allennlp = os.environ.get("OVERRIDE_ALLENNLP_IN_SETUP")
    install_requirements = (line.strip() for line in requirements_file)
    install_requirements = [
        fix_url_dependencies(line)
        for line in install_requirements
        if not line.startswith("#")
        if len(line) > 0
        if not override_allennlp or not requirement_is_allennlp(line)
    ]
    if override_allennlp:
        install_requirements.append(f"allennlp=={VERSION['VERSION']}")

# make pytest-runner a conditional requirement,
# per: https://github.com/pytest-dev/pytest-runner#considerations
needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

setup_requirements = [
    # add other setup requirements as necessary
] + pytest_runner

setup(
    name="allennlp_models",
    version=VERSION["VERSION"],
    description=("Officially supported models for the AllenNLP framework"),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="allennlp NLP deep learning machine reading semantic parsing parsers",
    url="https://github.com/allenai/allennlp-models",
    author="Allen Institute for Artificial Intelligence",
    author_email="allennlp@allenai.org",
    license="Apache",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"],),
    install_requires=install_requirements,
    setup_requires=setup_requirements,
    tests_require=["pytest", "flaky", "responses>=0.7"],
    include_package_data=True,
    python_requires=">=3.6.1",
    zip_safe=False,
)
