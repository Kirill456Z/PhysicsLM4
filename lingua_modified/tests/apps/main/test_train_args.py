from dataclasses import fields
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

from apps.main.train_args import (
    TrainArgs,
    _apply_data_to_synthetic_tasks_formatting_args,
    parse_data_mode,
    validate_train_args,
    prepare_train_args,
)
from data_synthetic_pretrain.dataloader.dataloader import build_dataloader


DEPO_DEBUG_YAML = (
    Path(__file__).resolve().parents[3]
    / "apps"
    / "main"
    / "configs"
    / "depo_debug.yaml"
)


def _merge_train_args_like_train_py(config_path: Path) -> TrainArgs:
    """Same OmegaConf merge as ``apps.main.train.main`` (file overrides defaults only)."""
    file_cfg = OmegaConf.load(config_path)
    default_cfg = OmegaConf.structured(TrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg)
    return OmegaConf.to_object(cfg)


class TestTrainArgs:
    def test_parse_data_mode(self):
        assert parse_data_mode("depo") == [("depo", 1.0)]
        assert parse_data_mode("depo:0.5|brevo:0.5") == [("depo", 0.5), ("brevo", 0.5)]
        assert parse_data_mode("depo:0.3|brevo:0.7") == [("depo", 0.3), ("brevo", 0.7)]

    def test_apply_data_to_synthetic_tasks_formatting_args(self):
        args = TrainArgs()
        args.data.batch_size = 128
        args.data.seq_len = 256
        args.data.pad_token = 7
        args.data.prefetch_size = 99
        args.data.n_workers = 3
        args.synthetic_tasks_formatting_args.batch_size = 1
        args.synthetic_tasks_formatting_args.seq_len = 2
        args.synthetic_tasks_formatting_args.pad_token = 3
        args.synthetic_tasks_formatting_args.prefetch_size = 4
        args.synthetic_tasks_formatting_args.n_workers = 5
        _apply_data_to_synthetic_tasks_formatting_args(args)
        fmt = args.synthetic_tasks_formatting_args
        assert fmt.batch_size == 128
        assert fmt.seq_len == 256
        assert fmt.pad_token == 7
        assert fmt.prefetch_size == 99
        assert fmt.n_workers == 3


class TestDepoDebugYamlAgainstTrainArgs:
    """``depo_debug.yaml`` must merge cleanly into ``TrainArgs`` (strict struct, no stray keys)."""

    def test_depo_debug_yaml_exists(self):
        assert DEPO_DEBUG_YAML.is_file(), f"missing {DEPO_DEBUG_YAML}"

    def test_depo_debug_yaml_merges_without_extra_or_invalid_keys(self):
        args = _merge_train_args_like_train_py(DEPO_DEBUG_YAML)
        assert isinstance(args, TrainArgs)

    def test_depo_debug_yaml_unknown_top_level_key_is_rejected(self):
        file_cfg = OmegaConf.load(DEPO_DEBUG_YAML)
        file_cfg["__not_a_train_arg__"] = True
        default_cfg = OmegaConf.structured(TrainArgs())
        with pytest.raises(ConfigKeyError):
            OmegaConf.merge(default_cfg, file_cfg)

    def test_depo_debug_yaml_top_level_keys_are_subset_of_train_args_fields(self):
        """Guards against silent typos if struct behavior ever changes for this merge path."""
        file_cfg = OmegaConf.load(DEPO_DEBUG_YAML)
        file_keys = set(OmegaConf.to_container(file_cfg, resolve=True).keys())
        allowed = {f.name for f in fields(TrainArgs)}
        assert file_keys <= allowed

    def test_prepare_train_args_does_not_raise(self):
        args = _merge_train_args_like_train_py(DEPO_DEBUG_YAML)
        prepare_train_args(args)

    def test_validate_train_args_passes(self):
        args = _merge_train_args_like_train_py(DEPO_DEBUG_YAML)
        prepare_train_args(args)
        validate_train_args(args, args.model.vocab_size)
    
    #@pytest.fixture(autouse=True)
    #def validated_train_args(self):
        #args = _merge_train_args_like_train_py(DEPO_DEBUG_YAML)
        #prepare_train_args(args)
        #validate_train_args(args, args.model.vocab_size)
        #args["eval"]["dump_dir"] = "/Users/kirillzemlanskij/Workspace/PhysicsLM4/test_dump"
        #return args
    
    #def test_validate_dataloader_args(self, validated_train_args):
        #args = validated_train_args
        #build_dataloader(args.synthetic_tasks_generation_args, args.synthetic_tasks_formatting_args)
