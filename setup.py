from setuptools import setup, find_packages

setup(
    name="financial-analyzer-pro",
    version="1.0.0",
    description="Financial Analyzer Pro - Streamlit App",
    packages=find_packages(),
    install_requires=[
        "streamlit==1.28.0",
    ],
    python_requires=">=3.8",
)
