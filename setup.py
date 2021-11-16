from setuptools import setup

with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ahlive",
    description="animate your data to life",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ahuang11/ahlive",
    packages=["ahlive"],
    include_package_data=True,
    install_requires=install_requires,
    keywords=["ahlive", "xarray", "animation", "easing", "interp", "gif"],
    use_scm_version={
        "version_scheme": "post-release",
        "local_scheme": "dirty-tag",
    },
    setup_requires=[
        "setuptools_scm",
        "setuptools>=30.3.0",
        "setuptools_scm_git_archive",
    ],
    zip_safe=True,
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Framework :: Matplotlib",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries",
    ],
)
