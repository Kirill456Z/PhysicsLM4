from pydantic import BaseModel, ConfigDict, model_validator
from enum import Enum


class NodeWord(BaseModel):
    """Frozen so instances are hashable and can be dict / defaultdict keys."""

    model_config = ConfigDict(frozen=True)
    tokens: tuple[int, ...]

class SpecialToken(BaseModel):
    token: int

    @model_validator(mode="before")
    @classmethod
    def _coerce_from_int(cls, data):
        if isinstance(data, int):
            return {"token": data}
        return data

class EncodingFormat(Enum):
    EDGES_LIST = "edges_list"
    ADJACENCY_LIST = "adjacency_list"

class EncodingConfig(BaseModel):
    pair_sep: SpecialToken
    node_to_neighbors_sep: SpecialToken
    format: EncodingFormat = EncodingFormat.EDGES_LIST

class GraphGeneratorConfig(BaseModel): 
    base_vocab_size: int
    min_token_length: int
    max_token_length: int
    is_directed: bool
    encoding_config: EncodingConfig
    is_dag: bool
    edge_probability: float

