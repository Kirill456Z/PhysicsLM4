"""Synthetic task modules."""
from .depo import DepoRefactored
from .bfs import BFSTaskGenerator

SYNTHETIC_TASKS = {
    DepoRefactored.name: DepoRefactored,
    BFSTaskGenerator.name: BFSTaskGenerator
}