from data_synthetic_pretrain.graph.models import EncodingConfig, EncodingFormat, SpecialToken


def test_special_token_from_int():
    t = SpecialToken.model_validate(25)
    assert t.token == 25


def test_encoding_config_from_yaml_like_dict():
    cfg = EncodingConfig.model_validate(
        {
            "pair_sep": 25,
            "node_to_neighbors_sep": 26,
            "format": "edges_list",
        }
    )
    assert cfg.pair_sep.token == 25
    assert cfg.node_to_neighbors_sep.token == 26
    assert cfg.format == EncodingFormat.EDGES_LIST
