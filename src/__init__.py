# src/__init__.py
# Makes src/ a Python package so notebooks can do:
#   from src.utils import set_seeds
# without any sys.path manipulation, as long as the notebook is run
# from the project root.
