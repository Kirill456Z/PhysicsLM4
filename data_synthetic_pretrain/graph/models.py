from pydantic import BaseModel, ConfigDict, model_validator
from enum import Enum


class NodeWord(BaseModel):
    """Frozen so instances are hashable and can be dict / defaultdict keys."""

    model_config = ConfigDict(frozen=True)
    tokens: tuple[int, ...]

    @model_validator(mode="before")
    @classmethod
    def _coerce_from_legacy_key_string(cls, data):
        # During JSON round-trips dict keys become strings (e.g. "tokens=(1, 2, 3)").
        if isinstance(data, str) and data.startswith("tokens=(") and data.endswith(")"):
            tokens_str = data[len("tokens=(") : -1].strip()
            if not tokens_str:
                return {"tokens": tuple()}
            tokens = tuple(int(tok.strip()) for tok in tokens_str.split(","))
            return {"tokens": tokens}
        return data

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

