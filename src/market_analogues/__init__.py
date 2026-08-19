"""Market-agnostic historical OHLCV analogue retrieval."""

from .config import AppConfig, DatasetSpec, load_config
from .types import EpisodeKey, InstrumentKey, SearchQuery, AnalogueMatch

__all__ = [
    "AppConfig", "DatasetSpec", "load_config", "EpisodeKey", "InstrumentKey",
    "SearchQuery", "AnalogueMatch",
]

__version__ = "0.1.0"

