from pathlib import Path

from rxnflow.data import prepare_all
from rxnflow.envs import SynthesisEnv
from rxnflow.types import ActionKind, RxnAction


def test_pipeline_is_resumable_aligned_and_cluster_free(prepared_env: Path) -> None:
    required = {
        "protocol.yaml",
        "workflow_map.csv",
        "bb_feature.pt",
        "prepare_manifest.json",
    }
    assert required <= {path.name for path in prepared_env.iterdir()}
    assert {path.name for path in (prepared_env / "smiles").glob("*.smi")} == {
        "amine.smi",
        "acid.smi",
    }
    assert not list(prepared_env.rglob("*cluster*"))

    fixtures = Path(__file__).parent / "fixtures"
    before = (prepared_env / "bb_feature.pt").stat().st_mtime_ns
    prepare_all(fixtures / "raw", prepared_env, [fixtures / "templates"])
    assert (prepared_env / "bb_feature.pt").stat().st_mtime_ns == before
    env = SynthesisEnv(prepared_env, max_atoms=20)
    assert set(env.blocks) == {"acid", "amine"}
    assert len(env.blocks["amine"].smiles) == env.blocks["amine"].properties.shape[0]


def test_all_synthesis_action_types(prepared_env: Path) -> None:
    env = SynthesisEnv(prepared_env, max_atoms=20)
    state = env.initial_state()
    assert env.next_action_kind(state) == ActionKind.SET_WORKFLOW
    state = env.step(state, RxnAction(ActionKind.SET_WORKFLOW, workflow_index=0))

    first = env.current_protocol(state)
    assert first.kind == ActionKind.FIRST_BLOCK
    state = env.step(
        state,
        RxnAction(first.kind, 0, state.protocol_order, first.block_type, 0),
    )
    second = env.current_protocol(state)
    assert second.kind == ActionKind.BI_REACTION
    state = env.step(
        state,
        RxnAction(second.kind, 0, state.protocol_order, second.block_type, 0),
    )
    third = env.current_protocol(state)
    assert third.kind == ActionKind.UNI_REACTION
    state = env.step(state, RxnAction(third.kind, 0, state.protocol_order))
    assert env.is_terminal(state)
    assert "At" not in state.smiles
