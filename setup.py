from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="viral_master_system",
    version="1.0.0",
    author="TheAlchemist",
    author_email="alchemist@example.com",
    description="Advanced Viral Content Optimization and Distribution System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thealchemist/viral_master_system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-asyncio>=0.15.1",
            "pytest-cov>=2.12.1",
            "black>=21.7b0",
            "mypy>=0.910",
            "isort>=5.9.3",
            "pre-commit>=2.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "viral-system=viral_master_system.core.run_system:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Bug Tracker": "https://github.com/thealchemist/viral_master_system/issues",
        "Documentation": "https://github.com/thealchemist/viral_master_system/docs",
        "Source Code": "https://github.com/thealchemist/viral_master_system",
    },
)

