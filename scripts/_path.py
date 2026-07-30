"""Put the repository root on sys.path so `traffic_estimate` imports.

Every script imports this first, which keeps `python scripts/foo.py` working
without installing the package.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
