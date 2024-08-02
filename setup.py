from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the contents of your requirements file
with open("requirements/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="datum",
    version="0.1.0-dev",
    author="Md. Muklasur Rahman Bulbul",
    author_email="mmrbulbul@gmail.com",
    description="Data analysis toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/yourproject",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements
)
