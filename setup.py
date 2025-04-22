from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="instruction-data-generator",
    version="1.0.0",
    author="ThanhMV",
    author_email="thanhmv@example.com",
    description="A tool for generating high-quality instruction data from various document formats",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thanhmv/instruction-data-generator",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: General",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "black>=23.3.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
            "pytest-cov>=4.1.0",
        ],
        "doc": [
            "python-docx>=0.8.11",
            "PyPDF2>=3.0.0",
            "pdfplumber>=0.9.0",
            "pdf2image>=1.16.3",
            "python-pptx>=0.6.21",
            "openpyxl>=3.1.2",
            "markdown>=3.4.3",
            "mammoth>=1.6.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "accelerate>=0.20.0",
            "bitsandbytes>=0.41.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "generate-instructions=instruction_generator.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "instruction_generator": [
            "config/*.yaml",
            "prompts/*.txt",
        ],
    },
)