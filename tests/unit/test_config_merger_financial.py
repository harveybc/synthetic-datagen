"""Config-merger precedence tests (financial-mode-aware)."""
import sys

from app.config import DEFAULT_VALUES
from app.config_merger import merge_config, process_unknown_args, convert_type


def test_unknown_args_are_parsed():
    # process_unknown_args returns raw strings; convert_type runs later
    # inside merge_config so the merge step is the one that types values.
    ua = process_unknown_args(["--block_length_mean", "16", "--label", "foo"])
    assert ua == {"block_length_mean": "16", "label": "foo"}


def test_convert_type_priorities():
    assert convert_type("42") == 42
    assert convert_type("3.14") == 3.14
    assert convert_type("hello") == "hello"


def test_merge_precedence_plugin_lt_repo_lt_file_lt_cli(monkeypatch):
    # Simulate that "--seed 7" was actually passed on the CLI.
    monkeypatch.setattr(sys, "argv", ["sdg", "--seed", "7"])
    plugin_params = {"seed": 1, "extra_plugin_only": "P"}
    repo_defaults = {"seed": 2, "block_length_mean": 99}
    file_cfg = {"seed": 3, "block_length_mean": 50}
    cli_args = {"seed": 7, "block_length_mean": 32}  # 32 = argparse default
    cfg = merge_config(repo_defaults, plugin_params, {}, file_cfg, cli_args, {})
    assert cfg["seed"] == 7              # CLI explicit wins
    # block_length_mean was the argparse default (NOT explicit) -> file wins.
    assert cfg["block_length_mean"] == 50
    assert cfg["extra_plugin_only"] == "P"  # plugin defaults visible


def test_unknown_cli_overrides_top_priority(monkeypatch):
    # Unknown args only take effect if their key appears in sys.argv AND is
    # not an argparse-known cli_arg. Use a key that argparse would not know.
    monkeypatch.setattr(sys, "argv", ["sdg", "--brand_new_key", "99"])
    cfg = merge_config(
        {}, {}, {}, {"brand_new_key": 2}, {},
        {"brand_new_key": "99"},
    )
    assert cfg["brand_new_key"] == 99   # int-coerced via convert_type


def test_default_values_have_financial_keys():
    for k in (
        "financial_mode", "datetime_column", "primitive_columns",
        "heldout_boundary", "project3_mode", "allow_non_research_mode",
    ):
        assert k in DEFAULT_VALUES
