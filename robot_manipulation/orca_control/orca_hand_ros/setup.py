from setuptools import find_packages, setup

package_name = "orca_hand_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jzq",
    maintainer_email="jzq@todo.todo",
    description="ROS 2 wrapper node for commanding an OrcaHand from joint target topics.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "orca_hand_node = orca_hand_ros.orca_hand_node:main",
        ],
    },
)
