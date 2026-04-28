"""Synthetic task modules."""
from .depo import DepoRefactored
from .bfs import BFSTaskGenerator
from .shortest_path import ShortestPathTaskGenerator
from .concomp_factor import ConCompFactorTaskGenerator

SYNTHETIC_TASKS = {
    DepoRefactored.name: DepoRefactored,
    BFSTaskGenerator.name: BFSTaskGenerator,
    ShortestPathTaskGenerator.name: ShortestPathTaskGenerator,
    ConCompFactorTaskGenerator.name: ConCompFactorTaskGenerator
}