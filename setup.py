from setuptools import setup, find_packages


setup(
    name="mobipi",
    version="0.1.0",
    description="Mobi-pi plus the MMWAM-OBC-001 typed option evaluator",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Jingyun Yang",
    author_email="jingyuny@stanford.edu",
    url="https://github.com/yjy0625/mobipi",
    packages=find_packages() + find_packages("src"),
    package_dir={"mobiwam": "src/mobiwam"},
    install_requires=[
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="robotics",
    python_requires=">=3.7",
)
