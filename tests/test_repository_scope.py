import ast
from pathlib import Path


def test_removed_dependencies_and_layers_are_absent() -> None:
    root = Path(__file__).parents[1]
    pyproject = (root / "pyproject.toml").read_text()
    assert "torch-geometric" not in pyproject
    assert "torch-scatter" not in pyproject
    assert "torch-sparse" not in pyproject
    assert "torch-cluster" not in pyproject
    assert "src/gflownet" not in {
        str(path.relative_to(root)) for path in root.glob("src/gflownet")
    }
    assert not (root / "src/rxnflow/envs/action.py").exists()
    assert not (root / "src/rxnflow/envs/workflow.py").exists()
    package_text = "\n".join(
        path.read_text() for path in (root / "src/rxnflow").rglob("*.py")
    )
    for removed in (
        "torch_geometric",
        "torch_scatter",
        "KMeans",
        "PIGNet",
        "UniDock",
        "reward_server",
        "eMolecules",
        "TieredActionSpace",
        "SET_WORKFLOW",
    ):
        assert removed not in package_text


def test_no_module_level_torch_tensor_constants() -> None:
    root = Path(__file__).parents[1] / "src" / "rxnflow"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            uses_torch = any(
                isinstance(child, ast.Name) and child.id == "torch"
                for child in ast.walk(node.value)
            )
            assert not uses_torch, f"module-level tensor constant in {path}:{node.lineno}"
