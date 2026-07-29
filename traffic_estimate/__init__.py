"""Traffic-Estimate: SUMO-based hourly traffic forecasting and OD estimation.

Layers, each usable on its own:

    network / taz / odmatrix / edgedata   the SUMO data model
    demand                                OD generators and land-use gravity
    services                              Overpass, routing costs, live traffic
    simulation                            the SUMO tool chain and calibration
    ml                                    forecaster, surrogate, inverse problem
    viz                                   figures and PDF reports

The scripts in scripts/ are thin CLI wrappers over these.
"""
from .edgedata import EdgeData, find_edgedata, scenario_dirs
from .network import STATIC_FEATURES, RoadNetwork
from .odmatrix import ODMatrix
from .taz import TazSet, Zone

__version__ = "2.0.0"

__all__ = ["EdgeData", "ODMatrix", "RoadNetwork", "STATIC_FEATURES", "TazSet",
           "Zone", "find_edgedata", "scenario_dirs"]
