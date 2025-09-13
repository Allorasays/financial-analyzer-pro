from setuptools import setup, find_packages

setup(
    name="financial-analyzer-pro",
    version="1.0.0",
    description="Financial Analyzer Pro - Ultra Minimal Version",
    author="Financial Analyzer Team",
    packages=find_packages(),
    install_requires=[
        "streamlit==1.28.0",
        "pandas==2.0.3",
        "yfinance==0.2.18",
        "numpy==1.24.3",
    ],
    python_requires=">=3.8",
)
