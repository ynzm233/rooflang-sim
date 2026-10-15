"""Portable run configuration and provenance; no source checkout is required."""

from dataclasses import asdict, is_dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform

import rooflang


def _version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "source-checkout"


def provenance():
    root = Path(rooflang.__file__).parent
    digest = sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".json"):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return {
        "package_version": _version("rooflang-sim"),
        "python": platform.python_version(),
        "dependencies": {"networkx": _version("networkx")},
        "source_repositories": json.loads((root / "sources.json").read_text()),
        "package_sha256": digest.hexdigest(),
    }


def hardware_snapshot(hardware):
    if hardware is None:
        return None
    nodes = [{"type": type(node).__name__, **vars(node)}
             for node in sorted(hardware.nodes, key=lambda item: item.name)]
    fabrics = []
    for _, _, attributes in hardware._graph.edges(data=True):
        for edge in attributes["fabrics"]:
            fabrics.append({**vars(edge), "src": edge.src.name, "dst": edge.dst.name})
    return {"nodes": nodes, "fabrics": sorted(fabrics, key=lambda item: item["name"])}


def config_snapshot(workflow, *, model_name, hardware_name, stage, config, hardware):
    resolved = workflow.metadata.get("_resolved_config")
    if resolved is None:
        model_config = workflow.metadata.get("_model_config")
        resolved = asdict(model_config) if is_dataclass(model_config) else dict(config or {})
    return {
        "model_name": model_name, "hardware_name": hardware_name,
        "selected_stage": stage, "input_config": dict(config or {}),
        "resolved_config": resolved, "hardware": hardware_snapshot(hardware),
    }


