"""
Entry point.

Usage:
    python run.py extract-data store/train --version cl_v1
    python run.py build-index --version cl_v1
    python run.py evaluate store/test --version cl_v1
    python run.py classify some_document.pdf --version cl_v1
"""
from src.cli import app

if __name__ == "__main__":
    app()
