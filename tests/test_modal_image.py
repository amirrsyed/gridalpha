"""The Modal image must install exactly what requirements.txt lists (a missing scikit-learn broke training)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_modal_image_installs_requirements_txt():
    src = (ROOT / "gridalpha" / "cloud" / "modal_app.py").read_text()
    assert 'pip_install_from_requirements(str(_ROOT / "requirements.txt"))' in src
    assert ".pip_install(" not in src.replace("pip_install_from_requirements", "")


def test_requirements_cover_runtime_imports():
    req = (ROOT / "requirements.txt").read_text().lower()
    for pkg in ["torch", "lightgbm", "scikit-learn", "gridstatus", "pyarrow", "scipy", "pandas", "pyyaml", "modal"]:
        assert pkg in req, pkg
