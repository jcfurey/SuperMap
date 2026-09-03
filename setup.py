import os
from glob import glob

from setuptools import find_packages, setup

package_name = "semantic_mapping"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*", "examples", "examples.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SuperX SLAM / AirLab, Carnegie Mellon University",
    maintainer_email="guofei@cmu.edu",
    description=(
        "SuperMap: a training-free spatio-temporal SLAM system producing a "
        "queryable 4D scene graph for language-guided navigation."
    ),
    license="MIT",
    extras_require={"test": ["pytest>=7.4,<8"]},
    entry_points={
        "console_scripts": [
            "semantic_mapping_node = semantic_mapping.node:main",
        ],
    },
)
