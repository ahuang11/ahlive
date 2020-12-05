from setuptools import setup

setup(
    name="ahlive",
    version="0.0.1",
    description="Make your data alive through animations",
    url="https://github.com/ahuang11/ahlive",
    packages=["ahlive"],
    include_package_data=True,
    install_requires=[
        "pygifsicle",
        "matplotlib",
        "imageio",
        "pandas",
        "xarray",
        "numpy",
        "param",
        "dask",
    ],
    keywords=["ahlive", "xarray", "animation" "easing", "interp", "gif"],
    zip_safe=True,
)
