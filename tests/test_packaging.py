import tomllib
from pathlib import Path


def test_pyproject_exposes_only_canonical_package_entries():
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = payload["project"]["scripts"]

    assert scripts["ricevision-qi"] == "ricevision_qi.app.main:main"
    assert scripts["ricevision-qi-tune"].startswith("ricevision_qi.app.tools.")
    assert scripts["ricevision-qi-evaluate"].startswith("ricevision_qi.app.tools.")
