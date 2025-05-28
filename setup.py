from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="laser-trim-ai-system",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@company.com",
    description="AI-powered quality analysis for laser trim testing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/laser-trim-ai-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Manufacturing",
        "Topic :: Scientific/Engineering :: Quality Control",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "openpyxl>=3.0.0",
        "scikit-learn>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "laser-trim-ai=src.cli:main",
        ],
    },
)
