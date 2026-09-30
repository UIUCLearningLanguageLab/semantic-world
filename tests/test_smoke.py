"""Stage 1 smoke test: the Rust extension module builds, imports, and reports its version."""

import importlib.metadata

import semantic_world
from semantic_world import _core


def test_core_module_imports() -> None:
    assert _core.version() == semantic_world.__version__


def test_package_version_matches_rust_version() -> None:
    assert importlib.metadata.version("semantic_world") == semantic_world.__version__


def test_every_engine_crate_is_linked() -> None:
    assert _core.crates() == ["sw-schema", "sw-core", "sw-rules", "sw-sense", "sw-render", "sw-log"]


def test_rerun_flag_is_a_bool() -> None:
    assert isinstance(_core.rerun_enabled(), bool)
